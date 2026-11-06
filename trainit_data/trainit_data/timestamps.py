"""Timestamps with explicit clock domains (PUBLIC).

A :class:`Timestamp` is ``{sec, nanosec, clock_domain, origin}`` (+ ``sim_source`` for ``SIM_TIME``),
D0_SCHEMA_PROPOSAL.md §2. Two timestamps are comparable only in the same clock domain (and, for
``SIM_TIME``, the same simulator). Nothing in the core converts between domains: a conversion would need
an explicit clock mapping, which D0 does not define. ``Timestamp`` deliberately has no ordering operators;
:func:`compare_timestamps` and :func:`skew_ns` refuse cross-domain pairs with ``INCOMPATIBLE_CLOCKS``.

Known limit (D0.1): ``DEVICE_MONOTONIC`` and ``HOST_MONOTONIC`` carry no device / host identity, so two
devices' monotonic clocks share one comparability key. D0 compares only stamps of one session clock
(ROS_TIME / SIM_TIME); a device or host identity is added (in ``extensions``, then a minor) by the first
slice that compares such stamps.
"""
from dataclasses import dataclass

from . import _strict as s
from .errors import INCOMPATIBLE_CLOCKS, INVALID_VALUE, DataError

# CLOSED: comparability semantics are implemented here; a new domain needs code (a mapping), not a row.
CLOCK_DOMAINS = frozenset({'ROS_TIME', 'SIM_TIME', 'DEVICE_MONOTONIC', 'HOST_MONOTONIC', 'HOST_WALL_UTC'})
_NS = 1_000_000_000


@dataclass(frozen=True)
class Timestamp:
    sec: int
    nanosec: int
    clock_domain: str
    origin: str
    sim_source: str = None

    @property
    def clock(self) -> tuple:
        """The comparability key: two stamps are comparable iff their keys are equal."""
        return (self.clock_domain, self.sim_source)

    def to_ns(self) -> int:
        return self.sec * _NS + self.nanosec

    @classmethod
    def from_dict(cls, value, vocab, where='timestamp') -> 'Timestamp':
        f = s.Fields(value, where)
        stamp = cls(sec=f.req('sec', s.integer(0)),
                    nanosec=f.req('nanosec', s.integer(0)),
                    clock_domain=f.req('clock_domain', s.one_of(CLOCK_DOMAINS)),
                    origin=f.req('origin', vocab.checker('stamp_origins')),
                    sim_source=f.opt('sim_source', vocab.checker('sim_sources')))
        f.done()
        stamp._validate(vocab, where)
        return stamp

    def _validate(self, vocab, where='timestamp') -> 'Timestamp':
        s.integer(0)(self.sec, f'{where}.sec')
        if s.integer(0)(self.nanosec, f'{where}.nanosec') >= _NS:
            s.fail(INVALID_VALUE, 'nanosec must be < 1e9', f'{where}.nanosec')
        s.one_of(CLOCK_DOMAINS)(self.clock_domain, f'{where}.clock_domain')
        vocab.require('stamp_origins', self.origin, f'{where}.origin')
        if self.clock_domain == 'SIM_TIME':
            if self.sim_source is None:
                s.fail(INVALID_VALUE, 'a SIM_TIME stamp names its sim_source', where)
            vocab.require('sim_sources', self.sim_source, f'{where}.sim_source')
        elif self.sim_source is not None:
            s.fail(INVALID_VALUE, 'sim_source is only meaningful for SIM_TIME', where)
        return self

    def to_dict(self) -> dict:
        out = {'sec': self.sec, 'nanosec': self.nanosec, 'clock_domain': self.clock_domain, 'origin': self.origin}
        s.put(out, 'sim_source', self.sim_source)
        return out


def _require_same_clock(a: Timestamp, b: Timestamp):
    if a.clock != b.clock:
        raise DataError(INCOMPATIBLE_CLOCKS,
                        f'cannot compare a {a.clock_domain} stamp with a {b.clock_domain} stamp '
                        '(no clock mapping is defined)',
                        context={'a': a.to_dict(), 'b': b.to_dict()})


def compare_timestamps(a: Timestamp, b: Timestamp) -> int:
    """-1 / 0 / 1 for a < b / a == b / a > b, in one clock domain; else ``INCOMPATIBLE_CLOCKS``."""
    _require_same_clock(a, b)
    return (a.to_ns() > b.to_ns()) - (a.to_ns() < b.to_ns())


def skew_ns(a: Timestamp, b: Timestamp) -> int:
    """|a - b| in nanoseconds, in one clock domain; else ``INCOMPATIBLE_CLOCKS``."""
    _require_same_clock(a, b)
    return abs(a.to_ns() - b.to_ns())


__all__ = ['CLOCK_DOMAINS', 'Timestamp', 'compare_timestamps', 'skew_ns']
