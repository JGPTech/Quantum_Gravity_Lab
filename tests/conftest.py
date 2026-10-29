"""pytest configuration: an opt-in `slow` marker for tests that integrate EinsteinPy geodesics."""
import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="also run slow tests (EinsteinPy geodesic integration, ~1-2 min)")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: slow test, enabled with --runslow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="slow test: pass --runslow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
