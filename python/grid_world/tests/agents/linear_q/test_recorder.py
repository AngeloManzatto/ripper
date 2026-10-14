"""
Created on Tue Oct 13 21:30:25 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

import numpy as np

from agents.linear_q_agent.diagnostics import Recorder

Q4 = [0.0, 0.0, 0.0, 0.0]


def raises(error, fn, *args):
    try:
        fn(*args)
    except error:
        return True
    return False


def test_bad_max_rows_and_empty_summary_raise():
    assert raises(ValueError, Recorder, 0)
    assert raises(ValueError, Recorder().summary)


def test_reset_clears_everything():
    rec = Recorder()
    rec.log_decision([1.0, 0.0], Q4, True)
    rec.log_update([1.0, 0.0], 0.5, 0.1, 1.0)
    assert len(rec.decisions) == 1 and len(rec.updates) == 1

    rec.reset()
    assert len(rec.decisions) == 0 and len(rec.updates) == 0


def test_logging_copies_its_inputs():
    rec = Recorder()
    x = np.array([1.0, 2.0])
    q = np.array([0.1, 0.5, 0.3, 0.2])
    rec.log_decision(x, q, False)

    x[:] = 0.0
    q[:] = 0.0

    assert np.allclose(rec.decisions[0]["x"], [1.0, 2.0])
    assert np.isclose(rec.summary()["margin_median"], 0.2)


def test_margins_use_only_evaluation_decisions():
    rec = Recorder()
    x = [1.0, 0.0]
    rec.log_decision(x, [0.1, 0.5, 0.3, 0.2], False)     # margin 0.2
    rec.log_decision(x, [1.0, 1.0, 0.0, 0.0], False)     # margin 0.0
    rec.log_decision(x, [0.0, 10.0, 0.0, 0.0], True)     # training: ignored

    s = rec.summary(margin_threshold=0.01)

    assert s["n_decisions"] == 3
    assert np.isclose(s["margin_median"], 0.1)
    assert np.isclose(s["margin_share_below"], 0.5)


def test_update_sizes_split_quiet_and_rewarded():
    rec = Recorder()
    x = [1.0, 0.0]
    rec.log_decision(x, [0.0, 0.1, 0.0, 0.0], False)     # margin 0.1

    for dq in (0.0001, -0.0003):
        rec.log_update(x, 0.0, dq, 0.0)                  # quiet steps
    for dq in (0.1, 0.2, -0.3):
        rec.log_update(x, 1.0, dq, 1.0)                  # rewarded steps

    s = rec.summary()

    assert s["n_updates"] == 5
    assert np.isclose(s["dq_quiet_median"], 0.0002)
    assert np.isclose(s["dq_quiet_p95"], np.percentile([0.0001, 0.0003], 95))
    assert np.isclose(s["dq_reward_median"], 0.2)
    assert np.isclose(s["dq_reward_p95"], np.percentile([0.1, 0.2, 0.3], 95))
    assert np.isclose(s["snr"], 0.1 / 0.2)


def test_missing_parts_are_none_not_errors():
    x = [1.0, 0.0]

    rec = Recorder()                                     # quiet updates only
    rec.log_decision(x, [0.0, 0.1, 0.0, 0.0], False)
    rec.log_update(x, 0.0, 0.0001, 0.0)
    s = rec.summary()
    assert s["dq_reward_median"] is None and s["snr"] is None
    assert s["dq_quiet_median"] is not None

    rec = Recorder()                                     # training decisions only
    rec.log_decision(x, Q4, True)
    rec.log_update(x, 1.0, 0.1, 1.0)
    s = rec.summary()
    assert s["margin_median"] is None and s["margin_share_below"] is None
    assert s["snr"] is None
    assert s["dq_reward_median"] is not None


def test_feature_statistics_use_all_decisions():
    rec = Recorder()
    rec.log_decision([1.0, 0.0], Q4, True)
    rec.log_decision([0.0, 0.1], Q4, False)

    s = rec.summary()

    assert np.allclose(s["feature_second_moments"], [0.5, 0.005])
    assert np.isclose(s["condition_number"], 100.0)


def test_max_rows_keeps_only_the_most_recent():
    rec = Recorder(max_rows=3)
    for i in range(5):
        rec.log_decision([float(i), 1.0], Q4, True)
        rec.log_update([float(i), 1.0], 0.0, 0.0, 0.0)

    assert len(rec.decisions) == 3 and len(rec.updates) == 3
    moments = rec.summary()["feature_second_moments"]
    assert np.isclose(moments[0], (4 + 9 + 16) / 3)      # rows 2, 3, 4 survived


def run():
    test_bad_max_rows_and_empty_summary_raise()
    test_reset_clears_everything()
    test_logging_copies_its_inputs()
    test_margins_use_only_evaluation_decisions()
    test_update_sizes_split_quiet_and_rewarded()
    test_missing_parts_are_none_not_errors()
    test_feature_statistics_use_all_decisions()
    test_max_rows_keeps_only_the_most_recent()
    print("recorder: all checks passed")


if __name__ == "__main__":
    run()