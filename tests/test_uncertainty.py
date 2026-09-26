"""Tests for exact categorical reasoning and uncertainty reports."""

from __future__ import annotations

import math

import pytest

from iaais.uncertainty import (
    ImpossibleEvidenceError,
    ProbabilityDistribution,
    UncertaintyModule,
    bayesian_update,
    expected_utility,
    propagate_categorical,
)


def test_bayesian_update_computes_normalized_posterior_and_evidence_probability():
    posterior, evidence_probability = bayesian_update(
        {"squat": 0.6, "lunge": 0.4},
        {"squat": 0.9, "lunge": 0.2},
    )

    assert evidence_probability == pytest.approx(0.62)
    assert posterior.probabilities["squat"] == pytest.approx(0.54 / 0.62)
    assert posterior.probabilities["lunge"] == pytest.approx(0.08 / 0.62)
    assert sum(posterior.probabilities.values()) == pytest.approx(1.0)


def test_impossible_evidence_is_rejected_instead_of_creating_nan_probabilities():
    with pytest.raises(ImpossibleEvidenceError):
        bayesian_update({"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 0.0})


def test_classifier_prediction_keeps_the_full_distribution_and_updates_report():
    uncertainty = UncertaintyModule()
    uncertainty.set_prior("activity", {"squat": 0.5, "lunge": 0.3, "rest": 0.2})

    report = uncertainty.observe_classifier_prediction(
        "activity",
        {"squat": 0.73, "lunge": 0.19, "rest": 0.08},
        metadata={"model_version": "demo-1"},
    )

    assert report.posterior.probabilities == {
        "squat": pytest.approx(0.73),
        "lunge": pytest.approx(0.19),
        "rest": pytest.approx(0.08),
    }
    assert report.most_likely == "squat"
    assert report.most_likely_probability == pytest.approx(0.73)
    assert report.evidence[-1].metadata["model_version"] == "demo-1"
    assert report.calibration_note


def test_belief_update_and_expected_utility_use_current_posterior():
    uncertainty = UncertaintyModule()
    uncertainty.set_prior("activity", {"squat": 0.6, "lunge": 0.4})
    report = uncertainty.update_categorical(
        "activity",
        "side_sensor_vote=squat",
        {"squat": 0.9, "lunge": 0.2},
        source="side_sensor",
    )

    assert report.posterior.probabilities["squat"] == pytest.approx(0.54 / 0.62)
    assert report.evidence_count == 1
    assert report.information_gain_bits > 0.0
    assert uncertainty.expected_utility("activity", {"squat": 10, "lunge": -2}) == pytest.approx(
        (0.54 / 0.62) * 10 + (0.08 / 0.62) * -2
    )


def test_probability_propagation_and_utility_validate_outcome_coverage():
    next_state = propagate_categorical(
        {"active": 0.7, "rest": 0.3},
        {
            "active": {"continue": 0.8, "stop": 0.2},
            "rest": {"continue": 0.1, "stop": 0.9},
        },
    )

    assert next_state.probabilities["continue"] == pytest.approx(0.59)
    assert next_state.probabilities["stop"] == pytest.approx(0.41)
    assert expected_utility(next_state, {"continue": 2, "stop": -1}) == pytest.approx(0.77)
    with pytest.raises(ValueError, match="missing outcomes"):
        expected_utility(next_state, {"continue": 2})


def test_calibration_metrics_are_computed_only_from_labeled_cases():
    uncertainty = UncertaintyModule()
    assert uncertainty.calibration_report("activity").sample_count == 0
    for actual in ("squat", "squat", "squat", "lunge"):
        report = uncertainty.record_calibration_case(
            "activity", {"squat": 0.8, "lunge": 0.2}, actual, bin_count=5
        )

    assert report.sample_count == 4
    assert report.expected_calibration_error == pytest.approx(0.05)
    assert report.brier_score is not None and math.isfinite(report.brier_score)
    assert report.log_loss is not None and math.isfinite(report.log_loss)
    assert report.bins[0].mean_confidence == pytest.approx(0.8)
    assert report.bins[0].accuracy == pytest.approx(0.75)


def test_distribution_rejects_non_normalized_or_mutable_probability_data():
    with pytest.raises(ValueError, match="sum to 1"):
        ProbabilityDistribution({"a": 0.7, "b": 0.4})

    values = {"a": 0.4, "b": 0.6}
    distribution = ProbabilityDistribution(values)
    values["a"] = 1.0
    assert distribution.probabilities["a"] == pytest.approx(0.4)
    with pytest.raises(TypeError):
        distribution.probabilities["a"] = 0.9
