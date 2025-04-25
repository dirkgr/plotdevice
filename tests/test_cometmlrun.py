from plotdevice import CometmlRun


def test_cometmlrun_full_fidelity():
    run = CometmlRun(
        "ai2",
        "olmo-2-0325-32b",
        "^olmo-2-0325-32b$")

    gnorm = run.get_time_series("optim/total grad norm")
    assert len(gnorm.xs) > 16000
