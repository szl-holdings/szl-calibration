"""Runtime version must agree with installed distribution metadata."""
from importlib.metadata import version

import szl_calibration


def test_runtime_version_matches_distribution_metadata():
    assert szl_calibration.__version__ == version("szl-calibration")
