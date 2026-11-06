"""Store location and initialisation (acceptance 9)."""
import json
from pathlib import Path

import pytest

from trainit_data.errors import STORE_INSIDE_REPOSITORY, STORE_INVALID, STORE_NOT_INITIALIZED, DataError
from trainit_data.store import ENV_VAR, Store, enclosing_repository, resolve_store_root


def test_default_root_is_trainitdata_in_home(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))
    assert resolve_store_root({}) == (tmp_path / 'TrainItData').resolve()
    assert resolve_store_root({ENV_VAR: '   '}) == (tmp_path / 'TrainItData').resolve()


def test_environment_override(tmp_path, monkeypatch):
    target = tmp_path / 'elsewhere' / 'data'
    assert resolve_store_root({ENV_VAR: str(target)}) == target.resolve()
    monkeypatch.setenv(ENV_VAR, str(target))
    store = Store.open(create=True)
    assert store.root == target.resolve() and (target / 'store.json').is_file()


def test_the_store_and_the_software_have_distinct_default_names():
    assert resolve_store_root({}).name == 'TrainItData' != 'trainit_data'


@pytest.mark.parametrize('git_is_file', [False, True])         # a work tree, or a worktree / submodule
def test_a_store_inside_a_git_work_tree_is_refused(tmp_path, git_is_file):
    repo = tmp_path / 'some_repo'
    repo.mkdir()
    if git_is_file:
        (repo / '.git').write_text('gitdir: /elsewhere/.git/worktrees/x\n')
    else:
        (repo / '.git').mkdir()
    for root in (repo, repo / 'data', repo / 'deep' / 'er' / 'TrainItData'):
        with pytest.raises(DataError) as err:
            Store.open(root, create=True)
        assert err.value.code == STORE_INSIDE_REPOSITORY
        assert err.value.context['repository'] == str(repo.resolve())
    assert not (repo / 'data').exists()                      # refused before anything is created


def test_this_source_repository_itself_is_refused():
    here = Path(__file__).resolve().parent
    assert enclosing_repository(here) is not None
    with pytest.raises(DataError) as err:
        Store.open(here / 'TrainItData', create=True)
    assert err.value.code == STORE_INSIDE_REPOSITORY
    assert not (here / 'TrainItData').exists()


def test_a_plain_directory_beside_a_repository_is_fine(tmp_path):
    (tmp_path / 'repo' / '.git').mkdir(parents=True)
    store = Store.open(tmp_path / 'TrainItData', create=True)  # a sibling, not inside
    assert store.store_id.startswith('store-')


def test_a_symlink_into_a_repository_is_refused(tmp_path):
    (tmp_path / 'repo' / '.git').mkdir(parents=True)
    (tmp_path / 'repo' / 'data').mkdir()
    (tmp_path / 'link').symlink_to(tmp_path / 'repo' / 'data')
    with pytest.raises(DataError) as err:
        Store.open(tmp_path / 'link', create=True)
    assert err.value.code == STORE_INSIDE_REPOSITORY


def test_init_and_reopen(tmp_path):
    root = tmp_path / 'TrainItData'
    with pytest.raises(DataError) as err:
        Store.open(root)                                     # never created implicitly
    assert err.value.code == STORE_NOT_INITIALIZED and not root.exists()
    created = Store.open(root, create=True)
    header = json.loads((root / 'store.json').read_text())
    assert header['schema'] == 'trainit-store/1.0' and 'trainit-capture/1.0' in header['schemas']
    assert sorted(p.name for p in root.iterdir()) == ['calibrations', 'sessions', 'store.json', 'vocab']
    assert Store.open(root).store_id == created.store_id == Store.open(root, create=True).store_id


def test_foreign_content_is_never_adopted(tmp_path):
    root = tmp_path / 'photos'
    root.mkdir()
    (root / 'holiday.txt').write_text('x')
    with pytest.raises(DataError) as err:
        Store.open(root, create=True)
    assert err.value.code == STORE_INVALID
    assert sorted(p.name for p in root.iterdir()) == ['holiday.txt']
    (tmp_path / 'afile').write_text('x')
    with pytest.raises(DataError) as err:
        Store.open(tmp_path / 'afile', create=True)
    assert err.value.code == STORE_INVALID


def test_a_damaged_store_header_is_refused(tmp_path):
    root = tmp_path / 'TrainItData'
    Store.open(root, create=True)
    (root / 'store.json').write_text('{"schema": "trainit-store/1.0", "store_id": "store-00000000", "extra": 1}')
    with pytest.raises(DataError) as err:
        Store.open(root)
    assert err.value.code == STORE_INVALID


def test_a_store_under_home_is_legitimate(tmp_path, monkeypatch):
    """Being under $HOME is never a reason to refuse (only a git work tree is); a repository NEXT to the
    store in $HOME changes nothing."""
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.delenv(ENV_VAR, raising=False)
    (tmp_path / 'some_workspace' / 'src' / '.git').mkdir(parents=True)
    store = Store.open(create=True)
    assert store.root == (tmp_path / 'TrainItData').resolve()
