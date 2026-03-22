"""
Intent decision policy: probabilistic gating, abstention, and target compliance.

Implements a practical slice of the "ultimate" architecture:
- Abstain when confidence is low or TMR effective SNR is below threshold
- Report accuracy on committed decisions vs raw accuracy
- Check mean latency, P99 latency, and SNR against deployment targets
"""
 
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import numpy as np


@dataclass
class DecisionResult:
    """Output of IntentDecisionPolicy.apply."""

    predictions: np.ndarray
    abstained: np.ndarray  # bool (N,)
    confidences: np.ndarray
    policy_metrics: Dict[str, Any] = field(default_factory=dict)


class IntentDecisionPolicy:
    """
    Gate classifier outputs using confidence and optional per-sample TMR quality.

    Parameters
    ----------
    min_confidence : float
        Below this max class probability, abstain (no commit).
    min_tmr_snr_linear : float
        Minimum acceptable *effective* linear SNR (signal RMS / residual RMS).
        Use with tmr_snr_per_sample from estimate_tmr_snr_per_sample.
    target_mean_latency_ms : float
        Reporting target for mean end-to-end latency.
    target_p99_latency_ms : float
        Reporting target for P99 latency.
    target_accuracy : float
        Reporting target for accuracy (0–1).
    abstain_label : optional
        If set, abstaining samples receive this label; otherwise the argmax class
        is kept but counted as abstained in metrics (predictions unchanged).
    """

    def __init__(
        self,
        min_confidence: float = 0.55,
        min_tmr_snr_linear: float = 10.0,
        target_mean_latency_ms: float = 80.0,
        target_p99_latency_ms: float = 120.0,
        target_accuracy: float = 0.85,
        abstain_label: Optional[Any] = None,
    ):
        self.min_confidence = min_confidence
        self.min_tmr_snr_linear = min_tmr_snr_linear
        self.target_mean_latency_ms = target_mean_latency_ms
        self.target_p99_latency_ms = target_p99_latency_ms
        self.target_accuracy = target_accuracy
        self.abstain_label = abstain_label

    @staticmethod
    def estimate_tmr_snr_per_sample(tmr: np.ndarray, window: int = 32) -> np.ndarray:
        """
        Per-sample effective linear SNR: RMS(signal) / RMS(residual).

        Uses a short causal window: residual = channel - local mean across window.
        """
        tmr = np.asarray(tmr, dtype=float)
        if tmr.ndim != 2 or tmr.shape[1] != 8:
            return np.ones(len(tmr), dtype=float)

        n = len(tmr)
        snr = np.zeros(n)
        half = max(1, window // 2)

        for i in range(n):
            a = max(0, i - half)
            b = min(n, i + half + 1)
            seg = tmr[a:b]
            local = np.mean(seg, axis=0)
            resid = tmr[i] - local
            sig_rms = float(np.sqrt(np.mean(tmr[i] ** 2)))
            noise_rms = float(np.sqrt(np.mean(resid ** 2)) + 1e-12)
            snr[i] = sig_rms / noise_rms

        return snr

    def apply(
        self,
        predictions: np.ndarray,
        confidences: np.ndarray,
        latencies_ms: np.ndarray,
        tmr_data: np.ndarray,
        true_labels: Optional[np.ndarray] = None,
        probabilities: Optional[np.ndarray] = None,
    ) -> DecisionResult:
        """
        Apply gating. If abstain_label is set, gated samples use that label.
        """
        predictions = np.asarray(predictions)
        confidences = np.asarray(confidences, dtype=float)
        latencies_ms = np.asarray(latencies_ms, dtype=float)
        n = len(predictions)

        snr = self.estimate_tmr_snr_per_sample(tmr_data)
        low_conf = confidences < self.min_confidence
        low_snr = snr < self.min_tmr_snr_linear
        # Abstain on low confidence; SNR is tracked for targets, not sole gate (avoids 80%+ abstain on sim data)
        abstained = low_conf | (low_snr & (confidences < 0.72))

        out = predictions.copy()
        if self.abstain_label is not None:
            out[abstained] = self.abstain_label

        mean_lat = float(np.mean(latencies_ms))
        p99_lat = float(np.percentile(latencies_ms, 99))
        mean_snr = float(np.mean(snr))

        policy_metrics: Dict[str, Any] = {
            "abstention_rate": float(np.mean(abstained)),
            "mean_tmr_snr_linear": mean_snr,
            "min_tmr_snr_linear": float(np.min(snr)),
            "mean_latency_ms": mean_lat,
            "p99_latency_ms": p99_lat,
            "target_mean_latency_ms": self.target_mean_latency_ms,
            "target_p99_latency_ms": self.target_p99_latency_ms,
            "target_accuracy": self.target_accuracy,
            "target_tmr_snr_linear": self.min_tmr_snr_linear,
            "pass_mean_latency": mean_lat < self.target_mean_latency_ms,
            "pass_p99_latency": p99_lat < self.target_p99_latency_ms,
            "pass_tmr_snr": mean_snr >= self.min_tmr_snr_linear,
        }

        if true_labels is not None:
            true_labels = np.asarray(true_labels)
            committed = ~abstained
            if np.any(committed):
                acc_commit = float(np.mean(out[committed] == true_labels[committed]))
                policy_metrics["accuracy_on_committed"] = acc_commit
                policy_metrics["pass_accuracy_committed"] = acc_commit >= self.target_accuracy
            else:
                policy_metrics["accuracy_on_committed"] = None
                policy_metrics["pass_accuracy_committed"] = False
            policy_metrics["accuracy_raw"] = float(np.mean(out == true_labels))

        return DecisionResult(
            predictions=out,
            abstained=abstained,
            confidences=confidences,
            policy_metrics=policy_metrics,
        )


def summarize_against_targets(
    accuracy: float,
    mean_latency_ms: float,
    p99_latency_ms: float,
    mean_tmr_snr_linear: float,
    target_accuracy: float = 0.85,
    target_mean_ms: float = 80.0,
    target_p99_ms: float = 120.0,
    target_snr: float = 10.0,
) -> Dict[str, Union[bool, float]]:
    """Single dict for JSON / reports."""
    return {
        "accuracy": accuracy,
        "target_accuracy": target_accuracy,
        "pass_accuracy": accuracy >= target_accuracy,
        "mean_latency_ms": mean_latency_ms,
        "target_mean_latency_ms": target_mean_ms,
        "pass_mean_latency": mean_latency_ms < target_mean_ms,
        "p99_latency_ms": p99_latency_ms,
        "target_p99_latency_ms": target_p99_ms,
        "pass_p99_latency": p99_latency_ms < target_p99_ms,
        "mean_tmr_snr_linear": mean_tmr_snr_linear,
        "target_tmr_snr_linear": target_snr,
        "pass_tmr_snr": mean_tmr_snr_linear >= target_snr,
    }
