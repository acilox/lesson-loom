import pytest

from lesson_loom.context_packs.loader import load_pack
from lesson_loom.core.config import default_config
from lesson_loom.providers import build_provider


@pytest.fixture
def provider():
    return build_provider("mock")


@pytest.fixture
def science_pack():
    return load_pack("science")


@pytest.fixture
def history_pack():
    return load_pack("history")


@pytest.fixture
def base_config():
    return default_config()
