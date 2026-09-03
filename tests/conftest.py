import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from make_fixtures import main as _make_fixtures


def pytest_sessionstart(session):
    _make_fixtures()
