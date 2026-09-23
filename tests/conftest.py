import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--run-remote-tests", action="store_true", default=False, help="run tests on remotely stored data (these can be slow)"
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "remote: mark test as remote data")

def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-remote-tests"):
        # --run-remote-tests given in cli: do not skip remote tests
        return

    skip_remote = pytest.mark.skip(reason="need --run-remote-tests option to run")

    for item in items:
        if "remote" in item.keywords:
            item.add_marker(skip_remote)

