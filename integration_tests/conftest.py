import logging

import pytest


@pytest.fixture(autouse=True, scope="session")
def _configure_logging():
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.INFO,
    )