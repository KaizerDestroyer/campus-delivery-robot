"""Topic 2 -- Probabilistic reasoning: a Bayes filter for "is the next cell blocked?".

Hidden state  H = "the cell ahead is blocked by a temporary obstacle".
Evidence      e = the obstacle sensor raises an ALERT (or stays silent).

Measurement update (Bayes' rule):

    P(H | e) = P(e | H) * P(H) / P(e),
    P(e)     = P(e | H) * P(H) + P(e | not H) * (1 - P(H))

Time update (obstacles appear / disappear while the robot waits),
a two-state Markov chain:

    P(H at t+1) = P(H at t) * (1 - p_clear) + (1 - P(H at t)) * p_block
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


def _check_prob(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


@dataclass(frozen=True)
class SensorModel:
    """Noisy obstacle sensor."""

    sensitivity: float = 0.90  # P(alert | blocked)  -- true-positive rate
    false_alarm: float = 0.20  # P(alert | clear)    -- false-positive rate

    def __post_init__(self) -> None:
        _check_prob("sensitivity", self.sensitivity)
        _check_prob("false_alarm", self.false_alarm)

    def likelihood(self, alert: bool, blocked: bool) -> float:
        """P(observation | state)."""
        p_alert = self.sensitivity if blocked else self.false_alarm
        return p_alert if alert else 1.0 - p_alert


@dataclass(frozen=True)
class ObstacleDynamics:
    """How a temporary obstacle changes from one tick to the next."""

    p_block: float  # P(clear  -> blocked) per tick
    p_clear: float  # P(blocked -> clear)  per tick

    def __post_init__(self) -> None:
        _check_prob("p_block", self.p_block)
        _check_prob("p_clear", self.p_clear)

    @classmethod
    def from_risk(cls, risk: float, mixing: float = 0.5) -> "ObstacleDynamics":
        """Chain whose long-run P(blocked) equals `risk`.

        `mixing` controls how fast obstacles change (1 = independent every tick).
        """
        _check_prob("risk", risk)
        _check_prob("mixing", mixing)
        return cls(p_block=risk * mixing, p_clear=(1.0 - risk) * mixing)

    @property
    def stationary(self) -> float:
        total = self.p_block + self.p_clear
        return self.p_block / total if total else 0.0


def bayes_update(prior: float, alert: bool, sensor: SensorModel) -> float:
    """Posterior P(blocked | observation) from a prior and one sensor reading."""
    _check_prob("prior", prior)
    p_e_blocked = sensor.likelihood(alert, blocked=True)
    p_e_clear = sensor.likelihood(alert, blocked=False)
    evidence = p_e_blocked * prior + p_e_clear * (1.0 - prior)  # P(e), law of total prob.
    if evidence == 0.0:
        raise ValueError("this observation is impossible under the sensor model")
    return p_e_blocked * prior / evidence


def predict(belief: float, dynamics: ObstacleDynamics) -> float:
    """Time update: belief one tick later, before the next sensor reading."""
    _check_prob("belief", belief)
    return belief * (1.0 - dynamics.p_clear) + (1.0 - belief) * dynamics.p_block


def filter_readings(prior: float, readings: Iterable[bool], sensor: SensorModel,
                    dynamics: ObstacleDynamics | None = None) -> list[float]:
    """Run the Bayes filter over a sequence of readings; returns beliefs after each.

    With `dynamics=None` the obstacle is assumed static (pure repeated Bayes).
    """
    beliefs, belief = [], prior
    for i, alert in enumerate(readings):
        if dynamics is not None and i > 0:
            belief = predict(belief, dynamics)
        belief = bayes_update(belief, alert, sensor)
        beliefs.append(belief)
    return beliefs
