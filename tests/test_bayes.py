"""Tests for topic 2: the Bayes filter."""

import random

import pytest

from robot.bayes import (ObstacleDynamics, SensorModel, bayes_update, filter_readings,
                         predict)

SENSOR = SensorModel(sensitivity=0.9, false_alarm=0.2)


def test_hand_calculation_from_the_report():
    # 0.9*0.3 / (0.9*0.3 + 0.2*0.7) = 0.27 / 0.41
    assert bayes_update(0.3, True, SENSOR) == pytest.approx(0.27 / 0.41)
    assert bayes_update(0.3, True, SENSOR) == pytest.approx(0.659, abs=1e-3)


def test_no_alert_lowers_the_belief():
    # 0.1*0.3 / (0.1*0.3 + 0.8*0.7) = 0.03 / 0.59
    assert bayes_update(0.3, False, SENSOR) == pytest.approx(0.03 / 0.59)


def test_base_rate_matters():
    """Same alert, rare obstacles -> the alert is most likely a false alarm."""
    assert bayes_update(0.02, True, SENSOR) < 0.1 < bayes_update(0.3, True, SENSOR)


def test_two_alerts_in_a_row():
    beliefs = filter_readings(0.3, [True, True], SENSOR)
    assert beliefs[0] == pytest.approx(0.659, abs=1e-3)
    assert beliefs[1] == pytest.approx(0.897, abs=1e-3)


def test_certain_priors_do_not_move():
    assert bayes_update(0.0, True, SENSOR) == 0.0
    assert bayes_update(1.0, False, SENSOR) == 1.0


def test_useless_sensor_gives_no_information():
    coin = SensorModel(sensitivity=0.5, false_alarm=0.5)
    assert bayes_update(0.3, True, coin) == pytest.approx(0.3)


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_invalid_probabilities_are_rejected(bad):
    with pytest.raises(ValueError):
        bayes_update(bad, True, SENSOR)
    with pytest.raises(ValueError):
        SensorModel(sensitivity=bad)


def test_impossible_observation_is_rejected():
    perfect = SensorModel(sensitivity=1.0, false_alarm=0.0)
    with pytest.raises(ValueError):
        bayes_update(1.0, False, perfect)   # blocked for sure, but no alert?


def test_time_update_converges_to_stationary_risk():
    dyn = ObstacleDynamics.from_risk(0.3, mixing=0.5)
    assert dyn.stationary == pytest.approx(0.3)
    belief = 1.0
    for _ in range(100):
        belief = predict(belief, dyn)
    assert belief == pytest.approx(0.3, abs=1e-6)
    assert predict(0.3, dyn) == pytest.approx(0.3)   # stationary point


def test_posterior_matches_monte_carlo_frequency():
    """Simulate many worlds: P(blocked | alert) measured empirically ~= Bayes."""
    rng = random.Random(42)
    alerts = blocked_and_alert = 0
    for _ in range(200_000):
        blocked = rng.random() < 0.3
        alert = rng.random() < (0.9 if blocked else 0.2)
        if alert:
            alerts += 1
            blocked_and_alert += blocked
    assert blocked_and_alert / alerts == pytest.approx(bayes_update(0.3, True, SENSOR),
                                                       abs=0.01)
