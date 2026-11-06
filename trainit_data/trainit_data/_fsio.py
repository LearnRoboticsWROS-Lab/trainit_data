"""Crash-safe file primitives (INTERNAL).

The ordering every durable write follows (D0_DESIGN.md §12): bytes -> ``<name>.tmp`` (created exclusively)
-> fsync -> rename onto the final name -> fsync of the directory. A final name therefore exists only with
complete, durable bytes. Nothing here deletes data; a failed write leaves its ``.tmp`` for the caller to
quarantine.

``write_fully`` is the single place bytes reach the OS: fault-injection tests replace it to simulate ENOSPC.
"""
import os

try:                                    # POSIX advisory locks; without them a live writer cannot be detected
    import fcntl
except ImportError:                     # pragma: no cover - non-POSIX platforms
    fcntl = None

TMP_SUFFIX = '.tmp'


def write_fully(fd: int, data: bytes):
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        view = view[written:]


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def tmp_path(path) -> str:
    return os.fspath(path) + TMP_SUFFIX


def _write_tmp(tmp: str, data: bytes):
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        write_fully(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)


def write_new_file(path, data: bytes):
    """Durably create ``path``, which must not exist. On failure the ``.tmp`` is left in place."""
    path = os.fspath(path)
    if os.path.lexists(path):
        raise FileExistsError(path)
    _write_tmp(tmp_path(path), data)
    os.rename(tmp_path(path), path)
    fsync_dir(os.path.dirname(path))


def replace_file(path, data: bytes):
    """Durably replace ``path`` (a header) atomically: readers see the old or the new bytes, never a mix."""
    path = os.fspath(path)
    tmp = tmp_path(path)
    if os.path.lexists(tmp):
        raise FileExistsError(tmp)      # a previous failed write; recover quarantines it
    _write_tmp(tmp, data)
    os.replace(tmp, path)
    fsync_dir(os.path.dirname(path))


class AppendLog:
    """An append-only file (the session index): each ``append`` is written in one call and fsynced."""

    def __init__(self, path):
        self.path = os.fspath(path)
        self._fd = os.open(self.path, os.O_WRONLY | os.O_APPEND)

    def append(self, data: bytes):
        write_fully(self._fd, data)
        os.fsync(self._fd)

    def close(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


class SessionLock:
    """The ``.active`` marker of an ACTIVE session, held under an exclusive advisory lock by its writer."""

    def __init__(self, path, *, create: bool):
        self.path = os.fspath(path)
        flags = os.O_WRONLY | (os.O_CREAT | os.O_EXCL if create else 0)
        self._fd = os.open(self.path, flags, 0o644)
        if fcntl is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(self._fd)
                self._fd = None
                raise
        # The marker may have been unlinked (a writer sealing) between open() and flock(): a lock on a removed
        # inode protects nothing. Treat it as "held" so the caller re-reads the session's state.
        held = os.fstat(self._fd)
        try:
            current = os.stat(self.path)
        except FileNotFoundError:
            current = None
        if held.st_nlink == 0 or current is None or (current.st_dev, current.st_ino) != (held.st_dev, held.st_ino):
            os.close(self._fd)
            self._fd = None
            raise BlockingIOError(f'{self.path} was replaced while being locked')

    def release(self, *, remove: bool):
        if self._fd is None:
            return
        if remove:
            os.unlink(self.path)        # the marker itself, never data
            fsync_dir(os.path.dirname(self.path))
        os.close(self._fd)
        self._fd = None


def quarantine(session_dir, rel: str) -> str:
    """MOVE ``<session>/<rel>`` to ``<session>/orphans/<rel>`` (never overwriting); returns the new relative path.

    Used for bytes no committed record references: a failed write's ``.tmp``, a payload whose index line
    was never written, a torn index tail. Nothing is deleted; a person decides later.
    """
    session_dir = os.fspath(session_dir)
    target_rel = 'orphans/' + rel
    target = os.path.join(session_dir, *target_rel.split('/'))
    suffix = 0
    while os.path.lexists(target):
        suffix += 1
        target = os.path.join(session_dir, *f'{target_rel}.{suffix}'.split('/'))
    source = os.path.join(session_dir, *rel.split('/'))
    makedirs_durable(os.path.dirname(target))
    os.rename(source, target)
    fsync_dir(os.path.dirname(target))
    fsync_dir(os.path.dirname(source))
    return os.path.relpath(target, session_dir).replace(os.sep, '/')


def makedirs_durable(path):
    """``mkdir -p`` whose every new directory entry is fsynced into its parent."""
    path = os.fspath(path)
    if os.path.isdir(path):
        return
    makedirs_durable(os.path.dirname(path))
    try:
        os.mkdir(path)
    except FileExistsError:
        return
    fsync_dir(os.path.dirname(path))
