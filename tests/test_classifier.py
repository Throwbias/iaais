import numpy as np
import pytest

from iaais.classifier import (
    Classifier,
    FeatureFactSpec,
    InertialFeatureEngineer,
    KnowledgeBaseFeatureExtractor,
    TrainingExample,
    write_sensor_features_to_knowledge_base,
)
from iaais.knowledge_base import FactStatus, KnowledgeBase
from iaais.uncertainty import UncertaintyModule


def training_examples():
    examples = []
    for index, energy in enumerate((0.05, 0.08, 0.10, 0.12, 0.15, 0.18)):
        examples.append(
            TrainingExample(
                {"movement_energy": energy, "sensor_profile": "wrist"},
                "rest",
                group_id=f"rest-session-{index}",
            )
        )
    for index, energy in enumerate((0.8, 0.9, 1.0)):
        examples.append(
            TrainingExample(
                {"movement_energy": energy, "sensor_profile": "wrist"},
                "squat",
                group_id=f"squat-session-{index}",
            )
        )
    for index, energy in enumerate((1.8, 2.0, 2.2)):
        examples.append(
            TrainingExample(
                {"movement_energy": energy, "sensor_profile": "wrist"},
                "press",
                group_id=f"press-session-{index}",
            )
        )
    return examples


def test_inertial_feature_engineer_produces_checked_statistics():
    acceleration = np.array([[1, 0, 0], [-1, 0, 0], [1, 0, 0]], dtype=float)
    gyroscope = np.array([[0, 1, 0], [0, 2, 0], [0, 3, 0]], dtype=float)

    features = InertialFeatureEngineer().transform(
        acceleration,
        gyroscope,
        sample_rate_hz=10,
        sensor_profile=" wrist ",
    )

    assert features["sample_count"] == 3.0
    assert features["window_duration_s"] == pytest.approx(0.2)
    assert features["accelerometer_x_rms"] == pytest.approx(1.0)
    assert features["accelerometer_magnitude_mean"] == pytest.approx(1.0)
    assert features["gyroscope_y_mean"] == pytest.approx(2.0)
    assert features["sensor_profile"] == "wrist"


def test_inertial_feature_engineer_rejects_malformed_windows():
    engineer = InertialFeatureEngineer()
    with pytest.raises(ValueError, match="shape"):
        engineer.transform([[0, 1]], [[0, 1, 2]], sample_rate_hz=10)
    with pytest.raises(ValueError, match="finite"):
        engineer.transform(
            [[0, 1, float("nan")], [0, 1, 2]],
            [[0, 1, 2], [0, 1, 2]],
            sample_rate_hz=10,
        )


def test_feature_facts_round_trip_and_conflicts_are_visible():
    kb = KnowledgeBase()
    facts = write_sensor_features_to_knowledge_base(
        kb,
        "window-1",
        {"movement_energy": 0.8, "sensor_profile": "wrist"},
        provenance=("measurement-17",),
    )

    vector = KnowledgeBaseFeatureExtractor().extract(kb, "window-1")
    assert vector.values == {"movement_energy": 0.8, "sensor_profile": "wrist"}
    assert vector.fact_ids == tuple(fact.fact_id for fact in facts)
    assert all(fact.status is FactStatus.PROPOSED for fact in facts)
    assert all(fact.provenance == ("measurement-17",) for fact in facts)

    kb.assert_fact(
        "sensor_feature",
        ("window-1", "movement_energy", 0.9),
        status=FactStatus.CONFIRMED,
    )
    with pytest.raises(ValueError, match="Conflicting"):
        KnowledgeBaseFeatureExtractor().extract(kb, "window-1")


def test_custom_feature_spec_reads_only_the_selected_kb_predicate():
    kb = KnowledgeBase()
    kb.assert_fact("window_stat", ("window-a", "energy", 0.4), status=FactStatus.CONFIRMED)
    kb.assert_fact("window_stat", ("window-b", "energy", 0.9), status=FactStatus.CONFIRMED)
    extractor = KnowledgeBaseFeatureExtractor(
        [FeatureFactSpec("window_stat", name_argument_index=1, value_argument_index=2)]
    )

    assert extractor.extract(kb, "window-a").values == {"energy": 0.4}


def test_classifier_returns_complete_distribution_and_explanations():
    classifier = Classifier().fit(training_examples())
    prediction = classifier.predict(
        {"movement_energy": 0.95, "sensor_profile": "wrist"},
        entity_id="window-predict",
    )

    assert prediction.entity_id == "window-predict"
    assert set(prediction.probabilities.probabilities) == {"rest", "squat", "press"}
    assert sum(prediction.probabilities.probabilities.values()) == pytest.approx(1.0)
    assert prediction.confidence == pytest.approx(
        prediction.probabilities.probabilities[prediction.label]
    )
    assert prediction.model_name == "exercise_logistic_regression"
    assert prediction.feature_contributions
    assert classifier._estimator.class_weight == "balanced"


def test_prediction_is_proposed_in_kb_and_full_distribution_reaches_uncertainty():
    classifier = Classifier().fit(training_examples())
    kb = KnowledgeBase()
    source_facts = write_sensor_features_to_knowledge_base(
        kb,
        "window-new",
        {"movement_energy": 1.0, "sensor_profile": "wrist"},
        provenance=("accelerometer-window-55",),
    )
    uncertainty = UncertaintyModule()

    prediction = classifier.predict_from_knowledge_base(
        kb,
        "window-new",
        uncertainty_module=uncertainty,
        interval=(10.0, 11.0),
    )

    assert prediction.prediction_fact.status is FactStatus.PROPOSED
    assert prediction.prediction_fact.predicate == "exercise_prediction"
    assert prediction.prediction_fact.provenance == tuple(f.fact_id for f in source_facts)
    assert prediction.prediction_fact.interval == (10.0, 11.0)
    assert len(prediction.probability_facts) == len(prediction.probabilities.probabilities)
    assert all(fact.status is FactStatus.PROPOSED for fact in prediction.probability_facts)
    assert {
        fact.arguments[2]: fact.arguments[3] for fact in prediction.probability_facts
    } == pytest.approx(dict(prediction.probabilities.probabilities))
    assert prediction.uncertainty_report.posterior == prediction.probabilities
    assert prediction.uncertainty_report.variable == "exercise_label:'window-new'"


def test_confirmed_kb_labels_can_train_and_proposed_labels_are_ignored():
    kb = KnowledgeBase()
    entities = []
    for entity_id, energy, label in (
        ("rest-1", 0.1, "rest"),
        ("rest-2", 0.2, "rest"),
        ("squat-1", 1.0, "squat"),
        ("squat-2", 1.2, "squat"),
    ):
        entities.append(entity_id)
        write_sensor_features_to_knowledge_base(kb, entity_id, {"energy": energy})
        kb.assert_fact(
            "exercise_label",
            (entity_id, label),
            status=FactStatus.CONFIRMED,
            source="reviewer",
        )
    kb.assert_fact(
        "exercise_label",
        ("rest-1", "incorrect-proposed-label"),
        status=FactStatus.PROPOSED,
        source="classifier",
    )

    classifier = Classifier().fit_from_knowledge_base(kb, entities)

    assert set(classifier.classes_) == {"rest", "squat"}
    assert classifier.predict({"energy": 0.15}).label == "rest"


def test_holdout_metrics_report_macro_f1_and_reject_recording_leakage():
    examples = training_examples()
    classifier = Classifier().fit(examples)
    holdout = [
        TrainingExample(
            {"movement_energy": 0.1, "sensor_profile": "wrist"},
            "rest",
            group_id="unseen-rest-session",
        ),
        TrainingExample(
            {"movement_energy": 0.95, "sensor_profile": "wrist"},
            "squat",
            group_id="unseen-squat-session",
        ),
        TrainingExample(
            {"movement_energy": 2.0, "sensor_profile": "wrist"},
            "press",
            group_id="unseen-press-session",
        ),
    ]

    report = classifier.evaluate(holdout)
    assert report.primary_metric == "macro_f1"
    assert report.sample_count == 3
    assert report.macro_f1 == pytest.approx(1.0)
    assert report.balanced_accuracy == pytest.approx(1.0)
    assert report.confusion_matrix == ((1, 0, 0), (0, 1, 0), (0, 0, 1))

    with pytest.raises(ValueError, match="share recording"):
        classifier.evaluate(
            [TrainingExample({"movement_energy": 0.1}, "rest", group_id="rest-session-0")]
        )
