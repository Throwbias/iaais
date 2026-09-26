"""Tests for particle filtering of noisy sensor values."""

from __future__ import annotations

import pytest

from iaais.uncertainty import ParticleFilter


def test_particle_filter_updates_posterior_weights_and_continuous_summary():
    particle_filter = ParticleFilter(
        (0.0, 1.0, 2.0, 3.0),
        resample_threshold=0.0,
        seed=5,
    )

    report = particle_filter.update(
        2.5,
        lambda state, observation: 0.8 if state >= 2.0 else 0.05,
        source="accelerometer",
    )

    assert report.mean == pytest.approx(2.3823529412)
    assert report.effective_sample_size < 4
    assert report.evidence_count == 1
    assert report.last_log_evidence is not None
    assert sum(report.weights) == pytest.approx(1.0)


def test_particle_filter_supports_vector_state_and_resamples_systematically():
    particle_filter = ParticleFilter(
        ((0.0, 0.0), (1.0, 2.0), (2.0, 4.0)),
        weights=(0.8, 0.1, 0.1),
        resample_threshold=0.0,
        seed=2,
    )
    report = particle_filter.report()
    assert report.mean == pytest.approx((0.3, 0.6))
    assert report.variance == pytest.approx((0.41, 1.64))

    particle_filter.systematic_resample()
    assert len(set(particle_filter.particles)) < 3
    assert particle_filter.weights == pytest.approx((1 / 3, 1 / 3, 1 / 3))


def test_zero_likelihood_preserves_existing_particle_weights():
    particle_filter = ParticleFilter((0, 1, 2), resample_threshold=0.0)
    before = particle_filter.weights

    with pytest.raises(ValueError, match="zero likelihood"):
        particle_filter.update("impossible", lambda state, _: 0.0)

    assert particle_filter.weights == before
