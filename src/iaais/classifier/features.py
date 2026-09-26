"""Inertial feature engineering and Knowledge Base feature extraction."""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

import numpy as np

from ..knowledge_base import Fact, FactStatus, KnowledgeBase, Polarity
from .models import FeatureFactSpec, FeatureValue, FeatureVector, validate_feature_mapping


class InertialFeatureEngineer:
    """Build a compact, interpretable feature vector from one sensor window.

    The first version uses raw-window summaries. It does not filter gravity,
    estimate repetitions, or claim that a movement was performed correctly.
    """

    def transform(
        self,
        accelerometer: Sequence[Sequence[float]] | np.ndarray,
        gyroscope: Sequence[Sequence[float]] | np.ndarray,
        *,
        sample_rate_hz: float,
        sensor_profile: str | None = None,
    ) -> Mapping[str, FeatureValue]:
        """Return axis statistics and magnitude summaries for an N x 3 window."""

        acceleration = self._as_window(accelerometer, "accelerometer")
        rotation = self._as_window(gyroscope, "gyroscope")
        if len(acceleration) != len(rotation):
            raise ValueError("Accelerometer and gyroscope windows must have equal sample counts")
        if isinstance(sample_rate_hz, bool):
            raise TypeError("sample_rate_hz must be numeric")
        rate = float(sample_rate_hz)
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError("sample_rate_hz must be finite and greater than zero")
        if sensor_profile is not None and (not isinstance(sensor_profile, str) or not sensor_profile.strip()):
            raise ValueError("sensor_profile must be a non-empty string when supplied")

        features: dict[str, FeatureValue] = {
            "sample_count": len(acceleration),
            "sample_rate_hz": rate,
            "window_duration_s": (len(acceleration) - 1) / rate,
        }
        self._axis_features(features, "accelerometer", acceleration)
        self._axis_features(features, "gyroscope", rotation)
        self._magnitude_features(features, "accelerometer", acceleration)
        self._magnitude_features(features, "gyroscope", rotation)
        # Signal magnitude area is a common compact summary of 3-axis movement.
        features["accelerometer_sma"] = float(np.mean(np.sum(np.abs(acceleration), axis=1)))
        features["gyroscope_sma"] = float(np.mean(np.sum(np.abs(rotation), axis=1)))
        if sensor_profile is not None:
            features["sensor_profile"] = sensor_profile.strip()
        return validate_feature_mapping(features)

    @staticmethod
    def _as_window(values: Sequence[Sequence[float]] | np.ndarray, name: str) -> np.ndarray:
        try:
            array = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a numeric array with shape (samples, 3)") from exc
        if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] < 2:
            raise ValueError(f"{name} must have shape (samples, 3) with at least two samples")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} values must all be finite")
        return array

    @staticmethod
    def _axis_features(features: dict[str, FeatureValue], name: str, values: np.ndarray) -> None:
        for index, axis in enumerate(("x", "y", "z")):
            series = values[:, index]
            prefix = f"{name}_{axis}"
            features[f"{prefix}_mean"] = float(np.mean(series))
            features[f"{prefix}_std"] = float(np.std(series, ddof=0))
            features[f"{prefix}_rms"] = float(np.sqrt(np.mean(np.square(series))))
            features[f"{prefix}_min"] = float(np.min(series))
            features[f"{prefix}_max"] = float(np.max(series))
            features[f"{prefix}_range"] = float(np.ptp(series))

    @staticmethod
    def _magnitude_features(features: dict[str, FeatureValue], name: str, values: np.ndarray) -> None:
        magnitude = np.linalg.norm(values, axis=1)
        features[f"{name}_magnitude_mean"] = float(np.mean(magnitude))
        features[f"{name}_magnitude_std"] = float(np.std(magnitude, ddof=0))
        features[f"{name}_magnitude_rms"] = float(np.sqrt(np.mean(np.square(magnitude))))
        features[f"{name}_magnitude_min"] = float(np.min(magnitude))
        features[f"{name}_magnitude_max"] = float(np.max(magnitude))
        features[f"{name}_magnitude_range"] = float(np.ptp(magnitude))


class KnowledgeBaseFeatureExtractor:
    """Read configured, active scalar feature facts for one entity/window."""

    def __init__(self, specs: Iterable[FeatureFactSpec] | None = None) -> None:
        self.specs = tuple(
            specs
            if specs is not None
            else (
                FeatureFactSpec(
                    predicate="sensor_feature",
                    entity_argument_index=0,
                    name_argument_index=1,
                    value_argument_index=2,
                ),
            )
        )
        if not self.specs:
            raise ValueError("At least one FeatureFactSpec is required")

    def extract(self, knowledge_base: KnowledgeBase, entity_id: Hashable) -> FeatureVector:
        """Collect feature values and source fact IDs, rejecting disagreements."""

        values: dict[str, FeatureValue] = {}
        source_ids: list[str] = []
        for fact in knowledge_base.facts:
            for spec in self.specs:
                if fact.predicate != spec.predicate or fact.status not in spec.accepted_statuses:
                    continue
                if fact.polarity is not Polarity.POSITIVE:
                    continue
                required_index = max(
                    spec.entity_argument_index,
                    spec.value_argument_index,
                    spec.name_argument_index if spec.name_argument_index is not None else 0,
                )
                if len(fact.arguments) <= required_index:
                    continue
                if fact.arguments[spec.entity_argument_index] != entity_id:
                    continue
                feature_name = (
                    spec.feature_name
                    if spec.feature_name is not None
                    else fact.arguments[spec.name_argument_index]  # type: ignore[index]
                )
                if not isinstance(feature_name, str) or not feature_name.strip():
                    raise ValueError(f"Feature fact {fact.fact_id!r} has an invalid feature name")
                value = fact.arguments[spec.value_argument_index]
                checked = validate_feature_mapping({feature_name: value})[feature_name]
                if feature_name in values and values[feature_name] != checked:
                    raise ValueError(
                        f"Conflicting Knowledge Base values for feature {feature_name!r} "
                        f"on entity {entity_id!r}"
                    )
                values[feature_name] = checked
                if fact.fact_id and fact.fact_id not in source_ids:
                    source_ids.append(fact.fact_id)
        if not values:
            raise KeyError(f"No accepted feature facts found for entity {entity_id!r}")
        return FeatureVector(entity_id=entity_id, values=values, fact_ids=tuple(source_ids))


def write_sensor_features_to_knowledge_base(
    knowledge_base: KnowledgeBase,
    entity_id: Hashable,
    features: Mapping[str, Any],
    *,
    source: str = "feature_engineer",
    status: FactStatus = FactStatus.PROPOSED,
    observed_at: str | None = None,
    interval: tuple[float, float] | None = None,
    provenance: Iterable[str] = (),
    feature_version: str = "inertial-v1",
) -> tuple[Fact, ...]:
    """Store engineered features as sensor_feature(window, name, value) facts."""

    if not source or not source.strip():
        raise ValueError("source must not be empty")
    checked = validate_feature_mapping(features)
    timestamp = observed_at or datetime.now(timezone.utc).isoformat()
    support = tuple(provenance)
    return tuple(
        knowledge_base.assert_fact(
            "sensor_feature",
            (entity_id, name, value),
            confidence=1.0,
            status=status,
            source=source,
            observed_at=timestamp,
            interval=interval,
            provenance=support,
            metadata={"feature_version": feature_version},
        )
        for name, value in checked.items()
    )
