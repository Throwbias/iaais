"""Supervised exercise-window classification and cross-module integration."""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import replace
from datetime import datetime, timezone
from uuid import uuid4

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from ..knowledge_base import Fact, FactStatus, KnowledgeBase, Polarity
from ..uncertainty import ProbabilityDistribution, UncertaintyModule
from .features import KnowledgeBaseFeatureExtractor
from .models import (
    ClassifierEvaluation,
    ClassifierPrediction,
    FeatureContribution,
    FeatureFactSpec,
    FeatureVector,
    TrainingExample,
    validate_feature_mapping,
)


class Classifier:
    """Balanced, interpretable multiclass logistic-regression classifier.

    Training uses explicitly supplied, human-labeled exercise windows. The
    default task is a single label for one inertial sensor window, such as
    rest or a configured exercise class. The model does not estimate reps,
    sets, technique, injury, diagnosis, or treatment.
    """

    def __init__(
        self,
        *,
        model_name: str = "exercise_logistic_regression",
        review_threshold: float = 0.60,
        regularization_c: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 0,
        feature_specs: Iterable[FeatureFactSpec] | None = None,
    ) -> None:
        if not model_name or not model_name.strip():
            raise ValueError("model_name must not be empty")
        if isinstance(review_threshold, bool) or not math.isfinite(float(review_threshold)) or not 0.0 <= float(review_threshold) <= 1.0:
            raise ValueError("review_threshold must be between 0 and 1")
        if isinstance(regularization_c, bool) or not math.isfinite(float(regularization_c)) or float(regularization_c) <= 0.0:
            raise ValueError("regularization_c must be finite and greater than zero")
        if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")

        self.model_name = model_name
        self.review_threshold = float(review_threshold)
        self.regularization_c = float(regularization_c)
        self.max_iter = max_iter
        self.random_state = random_state
        self.feature_extractor = KnowledgeBaseFeatureExtractor(feature_specs)
        self._vectorizer: DictVectorizer | None = None
        self._scaler: StandardScaler | None = None
        self._estimator: LogisticRegression | None = None
        self._training_group_ids: frozenset[str] = frozenset()
        self._feature_names: tuple[str, ...] = ()

    @property
    def classes_(self) -> tuple[str, ...]:
        """The sorted outcome labels learned during fit."""

        self._require_fitted()
        return tuple(str(label) for label in self._estimator.classes_)

    @property
    def feature_names_(self) -> tuple[str, ...]:
        """Raw feature keys seen during training."""

        self._require_fitted()
        return self._feature_names

    def fit(self, examples: Iterable[TrainingExample]) -> Classifier:
        """Fit on user-supplied labeled examples with balanced class weights."""

        rows = tuple(examples)
        if len(rows) < 2:
            raise ValueError("At least two labeled training examples are required")
        if any(not isinstance(example, TrainingExample) for example in rows):
            raise TypeError("fit expects TrainingExample instances")
        labels = {example.label for example in rows}
        if len(labels) < 2:
            raise ValueError("Training requires at least two distinct class labels")

        vectorizer = DictVectorizer(sparse=True, sort=True)
        matrix = vectorizer.fit_transform([dict(example.features) for example in rows])
        scaler = StandardScaler(with_mean=False)
        scaled = scaler.fit_transform(matrix)
        estimator = LogisticRegression(
            C=self.regularization_c,
            class_weight="balanced",
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver="lbfgs",
        )
        estimator.fit(scaled, [example.label for example in rows])

        self._vectorizer = vectorizer
        self._scaler = scaler
        self._estimator = estimator
        self._training_group_ids = frozenset(
            example.group_id for example in rows if example.group_id is not None
        )
        self._feature_names = tuple(
            sorted({name for example in rows for name in example.features})
        )
        return self

    def fit_from_knowledge_base(
        self,
        knowledge_base: KnowledgeBase,
        entity_ids: Iterable[Hashable],
        *,
        label_predicate: str = "exercise_label",
        label_entity_argument_index: int = 0,
        label_argument_index: int = 1,
        group_ids: Mapping[Hashable, str] | None = None,
    ) -> Classifier:
        """Train from feature facts and confirmed exercise_label facts.

        Predicted labels and proposed labels are not used as ground truth.
        Each requested entity must have exactly one distinct confirmed label.
        """

        if not label_predicate or not label_predicate.strip():
            raise ValueError("label_predicate must not be empty")
        if min(label_entity_argument_index, label_argument_index) < 0:
            raise ValueError("Label argument indices must be non-negative")
        requested = tuple(entity_ids)
        if not requested:
            raise ValueError("At least one labeled entity ID is required")

        labels_by_entity: dict[Hashable, list[Fact]] = {}
        for fact in knowledge_base.facts:
            if (
                fact.predicate != label_predicate
                or fact.status is not FactStatus.CONFIRMED
                or fact.polarity is not Polarity.POSITIVE
            ):
                continue
            if len(fact.arguments) <= max(label_entity_argument_index, label_argument_index):
                continue
            labels_by_entity.setdefault(fact.arguments[label_entity_argument_index], []).append(fact)

        examples: list[TrainingExample] = []
        for entity_id in requested:
            matching = labels_by_entity.get(entity_id, [])
            distinct_labels = {fact.arguments[label_argument_index] for fact in matching}
            if not matching:
                raise KeyError(f"No confirmed {label_predicate} fact for {entity_id!r}")
            if len(distinct_labels) != 1:
                raise ValueError(f"Expected one confirmed class label for {entity_id!r}")
            label = next(iter(distinct_labels))
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"Confirmed class label for {entity_id!r} must be a non-empty string")
            vector = self.feature_extractor.extract(knowledge_base, entity_id)
            examples.append(
                TrainingExample(
                    features=vector.values,
                    label=label,
                    group_id=(group_ids or {}).get(entity_id),
                    label_fact_id=matching[0].fact_id or None,
                )
            )
        return self.fit(examples)

    def predict(
        self,
        features: FeatureVector | Mapping[str, object],
        *,
        entity_id: Hashable | None = None,
    ) -> ClassifierPrediction:
        """Return the winning label, full class distribution, and explanation."""

        self._require_fitted()
        if isinstance(features, FeatureVector):
            if entity_id is not None and entity_id != features.entity_id:
                raise ValueError("entity_id does not match the supplied FeatureVector")
            values = features.values
            entity_id = features.entity_id
            fact_ids = features.fact_ids
        else:
            values = validate_feature_mapping(features)
            fact_ids = ()

        row = self._scaled_row(values)
        raw_probabilities = self._estimator.predict_proba(row)[0]
        distribution = ProbabilityDistribution.from_weights(
            {
                str(label): float(probability)
                for label, probability in zip(self._estimator.classes_, raw_probabilities)
            }
        )
        label = str(distribution.most_likely)
        confidence = float(distribution.probabilities[label])
        contributions = self.explain(values, label=label)
        return ClassifierPrediction(
            entity_id=entity_id,
            prediction_id=uuid4().hex,
            label=label,
            confidence=confidence,
            probabilities=distribution,
            review_required=confidence < self.review_threshold,
            review_threshold=self.review_threshold,
            model_name=self.model_name,
            feature_contributions=contributions,
            feature_fact_ids=fact_ids,
        )

    def predict_from_knowledge_base(
        self,
        knowledge_base: KnowledgeBase,
        entity_id: Hashable,
        *,
        uncertainty_module: UncertaintyModule | None = None,
        observed_at: str | None = None,
        interval: tuple[float, float] | None = None,
    ) -> ClassifierPrediction:
        """Read an entity's features, predict, then write proposed result facts."""

        vector = self.feature_extractor.extract(knowledge_base, entity_id)
        prediction = self.predict(vector)
        prediction = self._record_prediction(
            knowledge_base,
            prediction,
            observed_at=observed_at,
            interval=interval,
        )
        if uncertainty_module is not None:
            variable = f"exercise_label:{entity_id!r}"
            report = uncertainty_module.observe_classifier_prediction(
                variable,
                prediction.probabilities,
                source=f"classifier:{self.model_name}",
                metadata={
                    "prediction_id": prediction.prediction_id,
                    "entity_id": repr(entity_id),
                    "feature_fact_ids": prediction.feature_fact_ids,
                    "review_required": prediction.review_required,
                },
            )
            prediction = replace(prediction, uncertainty_report=report)
        return prediction

    def evaluate(self, examples: Iterable[TrainingExample]) -> ClassifierEvaluation:
        """Evaluate held-out labeled windows; reject known recording leakage."""

        self._require_fitted()
        rows = tuple(examples)
        if not rows:
            raise ValueError("At least one holdout example is required")
        if any(not isinstance(example, TrainingExample) for example in rows):
            raise TypeError("evaluate expects TrainingExample instances")
        if self._training_group_ids:
            if any(example.group_id is None for example in rows):
                raise ValueError(
                    "Holdout group_id is required because training examples used group IDs"
                )
            overlap = self._training_group_ids.intersection(
                example.group_id for example in rows if example.group_id is not None
            )
            if overlap:
                raise ValueError(
                    "Holdout examples share recording group(s) with training: "
                    + ", ".join(sorted(overlap))
                )

        y_true = [example.label for example in rows]
        unknown = set(y_true).difference(self.classes_)
        if unknown:
            raise ValueError(f"Holdout labels were not learned during fit: {sorted(unknown)}")
        y_pred = [self.predict(example.features).label for example in rows]
        labels = self.classes_
        precision = precision_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )
        recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        matrix = confusion_matrix(y_true, y_pred, labels=labels)
        return ClassifierEvaluation(
            labels=labels,
            sample_count=len(rows),
            accuracy=float(accuracy_score(y_true, y_pred)),
            macro_f1=float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
            per_class_precision={label: float(value) for label, value in zip(labels, precision)},
            per_class_recall={label: float(value) for label, value in zip(labels, recall)},
            confusion_matrix=tuple(tuple(int(value) for value in row) for row in matrix),
        )

    def explain(
        self,
        features: Mapping[str, object],
        *,
        label: str,
        limit: int = 10,
    ) -> tuple[FeatureContribution, ...]:
        """Rank linear contributions to one class score, excluding its intercept.

        A contribution explains this model's score in this feature space; it is
        not a causal claim about exercise mechanics.
        """

        self._require_fitted()
        if label not in self.classes_:
            raise ValueError(f"Unknown class label {label!r}")
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        values = validate_feature_mapping(features)
        class_index = self.classes_.index(label)
        coefficients = self._estimator.coef_[class_index]
        grouped: list[FeatureContribution] = []
        for feature_name, value in values.items():
            encoded = self._vectorizer.transform([{feature_name: value}])
            scaled = self._scaler.transform(encoded).tocoo()
            contribution = float(
                sum(
                    scaled.data[position] * coefficients[scaled.col[position]]
                    for position in range(len(scaled.data))
                )
            )
            if contribution != 0.0:
                grouped.append(FeatureContribution(feature_name, contribution))
        return tuple(sorted(grouped, key=lambda item: abs(item.contribution), reverse=True)[:limit])

    def _scaled_row(self, features: Mapping[str, object]):
        checked = validate_feature_mapping(features)
        if not set(checked).intersection(self._feature_names):
            raise ValueError("Prediction features do not include any feature seen during training")
        vector = self._vectorizer.transform([dict(checked)])
        if vector.nnz == 0:
            raise ValueError("Prediction features produce no known model inputs")
        return self._scaler.transform(vector)

    def _record_prediction(
        self,
        knowledge_base: KnowledgeBase,
        prediction: ClassifierPrediction,
        *,
        observed_at: str | None,
        interval: tuple[float, float] | None,
    ) -> ClassifierPrediction:
        prediction_id = prediction.prediction_id or uuid4().hex
        timestamp = observed_at or datetime.now(timezone.utc).isoformat()
        distribution = dict(prediction.probabilities.probabilities)
        source = f"classifier:{self.model_name}"
        top_fact = knowledge_base.assert_fact(
            "exercise_prediction",
            (prediction.entity_id, prediction_id, prediction.label),
            confidence=prediction.confidence,
            status=FactStatus.PROPOSED,
            source=source,
            observed_at=timestamp,
            interval=interval,
            provenance=prediction.feature_fact_ids,
            metadata={
                "probabilities": distribution,
                "review_required": prediction.review_required,
                "review_threshold": prediction.review_threshold,
                "feature_names": tuple(self.feature_names_),
                "calibration_validated": False,
                "probability_note": prediction.probability_note,
            },
        )
        probability_facts = tuple(
            knowledge_base.assert_fact(
                "exercise_class_probability",
                (prediction.entity_id, prediction_id, label, float(probability)),
                confidence=float(probability),
                status=FactStatus.PROPOSED,
                source=source,
                observed_at=timestamp,
                interval=interval,
                provenance=tuple(
                    fact_id for fact_id in (top_fact.fact_id, *prediction.feature_fact_ids) if fact_id
                ),
                metadata={
                    "calibration_validated": False,
                    "probability_note": prediction.probability_note,
                },
            )
            for label, probability in distribution.items()
        )
        return replace(
            prediction,
            prediction_id=prediction_id,
            prediction_fact=top_fact,
            probability_facts=probability_facts,
        )

    def _require_fitted(self) -> None:
        if self._estimator is None or self._vectorizer is None or self._scaler is None:
            raise RuntimeError("Classifier must be fit before prediction or evaluation")
