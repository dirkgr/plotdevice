from plotdevice import WandbRun

def test_wandbrun_multiple_values_per_step():
    run = WandbRun("ai2-llm", "olmo3", "OLMo29", name="OLMo 2.9")

    ts = run.get_time_series("train/CE loss")
    assert len(ts.xs) == len(set(ts.xs))
