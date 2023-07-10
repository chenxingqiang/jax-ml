import pytest

import xlearn


@pytest.fixture
def print_changed_only_false():
    xlearn.set_config(print_changed_only=False)
    yield
    xlearn.set_config(print_changed_only=True)  # reset to default
