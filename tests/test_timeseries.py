import numpy as np
import pytest

from plotdevice import TimeSeries, _moving_average

@pytest.fixture
def sample_timeseries() -> TimeSeries:
    """Provides a simple TimeSeries instance for testing."""
    xs = np.arange(10, dtype=float)
    ys = np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1], dtype=float)
    return TimeSeries(xs=xs, ys=ys, name="test_series", run_name="test_run")

def test_timeseries_init(sample_timeseries):
    """Tests basic TimeSeries initialization."""
    assert sample_timeseries.name == "test_series"
    assert sample_timeseries.run_name == "test_run"
    assert len(sample_timeseries.xs) == 10
    assert len(sample_timeseries.ys) == 10
    np.testing.assert_array_equal(sample_timeseries.xs, np.arange(10, dtype=float))

def test_timeseries_init_mismatched_shapes():
    """Tests that initialization fails with mismatched shapes."""
    with pytest.raises(AssertionError):
        TimeSeries(xs=np.arange(5), ys=np.arange(6), name="mismatch")

def test_timeseries_with_name(sample_timeseries):
    """Tests the with_name method."""
    new_name = "new_test_name"
    ts_new_name = sample_timeseries.with_name(new_name)
    assert ts_new_name.name == new_name
    assert ts_new_name.run_name == sample_timeseries.run_name
    np.testing.assert_array_equal(ts_new_name.xs, sample_timeseries.xs)
    np.testing.assert_array_equal(ts_new_name.ys, sample_timeseries.ys)

def test_moving_average_simple():
    """Tests the _moving_average helper function."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    expected = np.array([2.0, 3.0, 4.0]) # Moving average with n=3
    result = _moving_average(data, n=3)
    np.testing.assert_allclose(result, expected)

def test_smooth_with_moving_average(sample_timeseries):
    """Tests the smooth_with_moving_average method."""
    # Smoothing with width 3
    # Expected length = len(original) - width + 1 after cumsum trick
    # Interpolation keeps original length, then moving avg reduces it
    # Final length adjustment happens in TimeSeries method
    # Let's test basic properties like length reduction
    smoothed_ts = sample_timeseries.smooth_with_moving_average(width=4)

    assert smoothed_ts.name == sample_timeseries.name # Default name
    assert smoothed_ts.run_name == sample_timeseries.run_name

    # Length calculation is a bit complex due to interpolation and windowing
    # Expected length after interp+moving_average: 10 - 4 + 1 = 7
    # The TimeSeries method adjusts x axis, check resulting length
    # Original: 10 points. interp to 10. moving_avg(4) -> 7 points. Slice x axis [width//2:] -> [2:]. Slice x axis [:len(ynew)] -> [:7]. xnew[2:7] has length 5? No, xnew[width//2:] = xnew[2:], length 8. xnew[:len(ynew)] = xnew[:7], length 7. Final length 7.
    assert len(smoothed_ts.xs) == 7
    assert len(smoothed_ts.ys) == 7
    # Check if x values are within original range after adjustment
    assert smoothed_ts.xs[0] >= sample_timeseries.xs[0]
    assert smoothed_ts.xs[-1] <= sample_timeseries.xs[-1]
    # Could add more specific value checks if needed

def test_timeseries_average():
    """Tests static average method."""
    ts1 = TimeSeries(xs=np.array([0, 2, 4]), ys=np.array([1, 3, 5]), name="ts1", run_name="runA")
    ts2 = TimeSeries(xs=np.array([1, 3]), ys=np.array([10, 30]), name="ts2", run_name="runA")

    avg_ts = TimeSeries.average([ts1, ts2])

    # Expected xs are the union of unique x-values: [0, 1, 2, 3, 4]
    expected_xs = np.array([0, 1, 2, 3, 4])
    np.testing.assert_array_equal(avg_ts.xs, expected_xs)

    # Expected ys are averages of interpolated values at each x in expected_xs
    # At x=0: ts1=1, ts2=interp(0, [1,3], [10,30])=10 (using default interp bounds) -> avg(1, 10) = 5.5? No, numpy interp default is edge values. interp(0, [1,3], [10,30]) is 10. Avg(1, 10)=5.5
    # At x=1: ts1=interp(1, [0,2,4], [1,3,5])=2, ts2=10 -> avg(2, 10) = 6.0
    # At x=2: ts1=3, ts2=interp(2, [1,3], [10,30])=20 -> avg(3, 20) = 11.5
    # At x=3: ts1=interp(3, [0,2,4], [1,3,5])=4, ts2=30 -> avg(4, 30) = 17.0
    # At x=4: ts1=5, ts2=interp(4, [1,3], [10,30])=30 -> avg(5, 30) = 17.5
    expected_ys = np.array([5.5, 6.0, 11.5, 17.0, 17.5])
    np.testing.assert_allclose(avg_ts.ys, expected_ys)

    # Check name generation and run_name propagation
    assert avg_ts.name == "avg(ts1,ts2)"
    assert avg_ts.run_name == "runA" # Since both runs are the same

def test_timeseries_average_different_runs():
    """Tests static average method with different run names."""
    ts1 = TimeSeries(xs=np.array([0, 2]), ys=np.array([1, 3]), name="ts1", run_name="runA")
    ts2 = TimeSeries(xs=np.array([1, 3]), ys=np.array([10, 30]), name="ts2", run_name="runB")

    avg_ts = TimeSeries.average([ts1, ts2])
    assert avg_ts.run_name is None # Should be None if runs differ
