import pytest

from trainit_data.store import Store
from trainit_data.vocab import Vocabulary


@pytest.fixture
def vocab():
    return Vocabulary.default()


@pytest.fixture
def store(tmp_path):
    return Store.open(tmp_path / 'TrainItData', create=True)
