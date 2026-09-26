"""Empirical calibration metrics for categorical predictions."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Hashable

from .models import CalibrationBin, CalibrationReport, ProbabilityDistribution


class CalibrationTracker:
    """Accumulate labeled predictions and compute Brier, log-loss, and ECE."""

    def __init__(self, variable: str, *, bin_count: int = 10) -> None:
        if not variable or not variable.strip():
            raise ValueError("Variable name must not be empty")
        if not isinstance(bin_count, int) or bin_count < 1:
            raise ValueError("bin_count must be a positive integer")
        self.variable = variable
        self.bin_count = bin_count
        self._cases: list[tuple[ProbabilityDistribution, Hashable]] = []

    def record(self, probabilities: ProbabilityDistribution, actual_outcome: Hashable) -> None:
        """Add one forecast with its subsequently observed outcome."""

        if actual_outcome not in probabilities.probabilities:
            raise ValueError("Actual outcome is not in the prediction's outcome space")
        if self._cases and set(self._cases[0][0].probabilities) != set(probabilities.probabilities):
            raise ValueError("All calibration cases must use the same outcome space")
        self._cases.append((probabilities, actual_outcome))

    def report(self) -> CalibrationReport:
        """Return empirical calibration diagnostics for the recorded cases."""

        count = len(self._cases)
        if not count:
            return CalibrationReport(self.variable, 0, None, None, None, ())

        brier_total = 0.0
        log_loss_total = 0.0
        bin_cases: dict[int, list[tuple[float, bool]]] = defaultdict(list)
        for distribution, actual in self._cases:
            probabilities = distribution.probabilities
            brier_total += sum(
                (probability - (1.0 if outcome == actual else 0.0)) ** 2
                for outcome, probability in probabilities.items()
            )
            log_loss_total -= math.log(max(probabilities[actual], 1e-15))
            top = distribution.most_likely
            confidence = probabilities[top]
            bin_index = min(int(confidence * self.bin_count), self.bin_count - 1)
            bin_cases[bin_index].append((confidence, top == actual))

        bins: list[CalibrationBin] = []
        ece = 0.0
        for index in range(self.bin_count):
            values = bin_cases.get(index, [])
            if not values:
                continue
            mean_confidence = sum(confidence for confidence, _ in values) / len(values)
            accuracy = sum(correct for _, correct in values) / len(values)
            ece += (len(values) / count) * abs(mean_confidence - accuracy)
            bins.append(
                CalibrationBin(
                    lower_bound=index / self.bin_count,
                    upper_bound=(index + 1) / self.bin_count,
                    count=len(values),
                    mean_confidence=mean_confidence,
                    accuracy=accuracy,
                )
            )

        return CalibrationReport(
            variable=self.variable,
            sample_count=count,
            brier_score=brier_total / count,
            log_loss=log_loss_total / count,
            expected_calibration_error=ece,
            bins=tuple(bins),
        )
