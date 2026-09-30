"""A probabilistically balanced ordered map based on the skip list structure.

A skip list stores keys in sorted order and supports O(log n) average-time
lookup, insertion, deletion, and range iteration. The expected complexity is
achieved not by rebalancing (as an AVL or red-black tree does) but by giving
each node a randomly chosen "tower height": roughly half of the nodes appear at
level 1, a quarter at level 2, an eighth at level 3, and so on. Because the
heights come from coin flips, the long-run behavior is good with overwhelming
probability, but any single operation can be O(n) in the worst case.

We expose a dict-like ordered map: keys must be comparable, values are
arbitrary. The iteration order is ascending by key. Duplicate keys are not
allowed; inserting with an existing key replaces its value, matching the
semantics of a plain dict.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Iterator, Optional


class _Node:
    """A single skip-list node.

    A node stores one key/value pair and a list of forward pointers, one per
    level at which the node participates. ``forward[i]`` is the next node at
    level ``i`` (or None at the tail). Indexing levels from 0 keeps the
    pointer list and the level arithmetic in lockstep, which is easier to
    reason about than the 1-based numbering some textbooks use.
    """

    __slots__ = ("key", "value", "forward")

    def __init__(self, key: Any, value: Any, height: int) -> None:
        self.key = key
        self.value = value
        # A fresh node contributes to levels 0..height-1. Each entry points
        # to the next node at that level (or None at the tail).
        self.forward: list[Optional[_Node]] = [None] * height

    @property
    def height(self) -> int:
        return len(self.forward)


class SkipList:
    """An ordered map backed by a skip list.

    Keys must be totally ordered by ``<``. Values may be any object. The map
    keeps keys in ascending order at all times and supports dict-style access
    (``__getitem__``, ``__setitem__``, ``__delitem__``, ``__contains__``),
    iteration over keys in order, ``len()``, and range iteration via
    :meth:`range`.

    The expected time per operation is O(log n). Worst-case time is O(n) and
    occurs with vanishing probability under an honest random source. Memory is
    O(n) in expectation.
    """

    __slots__ = ("_head", "_max_level", "_size", "_p", "_rng")

    def __init__(
        self,
        p: float = 0.5,
        rng: Optional[Callable[[], float]] = None,
    ) -> None:
        """Construct an empty skip list.

        Args:
            p: Probability that a node's tower grows by one more level. The
                classic skip-list analysis uses 0.5. Values in (0, 1) other
                than 0.5 trade taller towers (and more pointer chasing) for
                wider spacing between index levels; 0.5 is almost always the
                right choice, so it is the default.
            rng: A zero-argument callable returning a float in [0, 1). Used
                to draw node heights. Injecting it makes the structure
                deterministic for tests; in production you leave it as
                ``None`` and get a ``random.random`` source seeded from OS
                entropy.

        Raises:
            ValueError: if ``p`` is not in the open interval (0, 1).
        """
        if not 0.0 < p < 1.0:
            raise ValueError("p must be in the open interval (0, 1)")
        self._p = p
        self._rng = rng if rng is not None else random.random
        # The head node participates at every level we ever create; its key
        # is never compared because nothing is smaller than it.
        self._head = _Node(None, None, 1)
        self._max_level = 1
        self._size = 0

    # ------------------------------------------------------------------
    # Height selection
    # ------------------------------------------------------------------

    def _random_height(self) -> int:
        """Draw a fresh node height via repeated Bernoulli trials.

        Returns 1 with probability ``1 - p``, 2 with probability ``p(1-p)``,
        and so on. We cap at 64 levels so a pathologically lucky streak cannot
        produce a tower taller than the list is long by more than a constant
        factor in any realistic input size.
        """
        height = 1
        cap = 64
        while height < cap and self._rng() < self._p:
            height += 1
        return height

    # ------------------------------------------------------------------
    # Core traversal
    # ------------------------------------------------------------------

    def _find_update_path(self, key: Any) -> list[_Node]:
        """Return the nodes whose forward pointers may need updating to insert
        or remove ``key``.

        For each level ``lvl`` (0-indexed), ``updates[lvl]`` is the rightmost
        node at that level whose key is strictly less than ``key``. When we
        later splice a node in or out, the pointers at exactly these nodes are
        the ones that change.
        """
        updates: list[_Node] = [self._head] * self._max_level
        node = self._head
        for lvl in range(self._max_level - 1, -1, -1):
            nxt = node.forward[lvl]
            while nxt is not None and nxt.key < key:
                node = nxt
                nxt = node.forward[lvl]
            updates[lvl] = node
        return updates

    # ------------------------------------------------------------------
    # Public API: insertion / lookup / deletion
    # ------------------------------------------------------------------

    def __setitem__(self, key: Any, value: Any) -> None:
        """Insert ``key`` -> ``value``, replacing the value if ``key`` exists.

        Replacement keeps the existing node's tower height, which preserves
        the random structure that earlier insertions produced. Creating a
        fresh node on every update would perturb the height distribution in a
        way the textbook analysis does not cover.
        """
        updates = self._find_update_path(key)
        predecessor = updates[0]
        existing = predecessor.forward[0]
        if existing is not None and not (key < existing.key) and not (existing.key < key):
            existing.value = value
            return

        height = self._random_height()
        if height > self._max_level:
            self._head.forward.extend([None] * (height - self._max_level))
            updates.extend([self._head] * (height - self._max_level))
            self._max_level = height

        new_node = _Node(key, value, height)
        for lvl in range(height):
            new_node.forward[lvl] = updates[lvl].forward[lvl]
            updates[lvl].forward[lvl] = new_node
        self._size += 1

    def __getitem__(self, key: Any) -> Any:
        """Return the value for ``key`` or raise ``KeyError``."""
        node = self._head
        for lvl in range(self._max_level - 1, -1, -1):
            nxt = node.forward[lvl]
            while nxt is not None and nxt.key < key:
                node = nxt
                nxt = node.forward[lvl]
        candidate = node.forward[0]
        if candidate is not None and not (key < candidate.key) and not (candidate.key < key):
            return candidate.value
        raise KeyError(key)

    def __delitem__(self, key: Any) -> None:
        """Remove ``key`` or raise ``KeyError`` if absent.

        After deletion we drop any trailing levels of the head that no longer
        point anywhere, so the search height tracks the data rather than the
        historical maximum. This is purely a space optimization; correctness
        does not depend on it.
        """
        updates = self._find_update_path(key)
        target = updates[0].forward[0]
        if target is None or (key < target.key) or (target.key < key):
            raise KeyError(key)
        for lvl in range(target.height):
            updates[lvl].forward[lvl] = target.forward[lvl]
        self._size -= 1
        while self._max_level > 1 and self._head.forward[self._max_level - 1] is None:
            self._max_level -= 1
        self._head.forward = self._head.forward[: self._max_level]

    def __contains__(self, key: object) -> bool:
        """Return whether ``key`` is present."""
        node = self._head
        for lvl in range(self._max_level - 1, -1, -1):
            nxt = node.forward[lvl]
            while nxt is not None and nxt.key < key:
                node = nxt
                nxt = node.forward[lvl]
        candidate = node.forward[0]
        if candidate is None:
            return False
        return not (key < candidate.key) and not (candidate.key < key)

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[Any]:
        """Yield keys in ascending order."""
        node = self._head.forward[0]
        while node is not None:
            yield node.key
            node = node.forward[0]

    def items(self) -> Iterator[tuple[Any, Any]]:
        """Yield ``(key, value)`` pairs in ascending key order."""
        node = self._head.forward[0]
        while node is not None:
            yield (node.key, node.value)
            node = node.forward[0]

    def keys(self) -> Iterator[Any]:
        """Yield keys in ascending order (alias for ``__iter__``)."""
        return iter(self)

    def values(self) -> Iterator[Any]:
        """Yield values in ascending key order."""
        node = self._head.forward[0]
        while node is not None:
            yield node.value
            node = node.forward[0]

    def get(self, key: Any, default: Any = None) -> Any:
        """Return the value for ``key``, or ``default`` if absent."""
        try:
            return self[key]
        except KeyError:
            return default

    def range(
        self,
        start: Any = None,
        stop: Any = None,
        inclusive_start: bool = True,
        inclusive_stop: bool = False,
    ) -> Iterator[tuple[Any, Any]]:
        """Yield ``(key, value)`` pairs whose key lies in ``[start, stop)``.

        ``None`` for either bound means "no bound on that side". The
        inclusivity flags let callers express any of the four half-open /
        closed combinations without inventing sentinel values, which is the
        kind of ambiguity that quietly corrupts range queries.
        """
        if start is None:
            node: Optional[_Node] = self._head.forward[0]
        else:
            node = self._head
            for lvl in range(self._max_level - 1, -1, -1):
                nxt = node.forward[lvl]
                while nxt is not None and nxt.key < start:
                    node = nxt
                    nxt = node.forward[lvl]
            node = node.forward[0]
            if node is not None and not inclusive_start and not (start < node.key) and not (node.key < start):
                node = node.forward[0]

        while node is not None:
            if stop is not None:
                if inclusive_stop:
                    if stop < node.key:
                        break
                else:
                    if not (node.key < stop):
                        break
            yield (node.key, node.value)
            node = node.forward[0]

    def min_item(self) -> tuple[Any, Any]:
        """Return the ``(key, value)`` with the smallest key, or raise
        ``KeyError`` if the map is empty."""
        node = self._head.forward[0]
        if node is None:
            raise KeyError("min_item() called on an empty SkipList")
        return (node.key, node.value)

    def max_item(self) -> tuple[Any, Any]:
        """Return the ``(key, value)`` with the largest key, or raise
        ``KeyError`` if the map is empty.

        Finding the maximum is the one operation where a skip list is slower
        than a balanced tree: there is no back-pointer, so we walk down the
        highest level until the next pointer is ``None``. That is still
        O(log n) in expectation.
        """
        node = self._head
        for lvl in range(self._max_level - 1, -1, -1):
            while node.forward[lvl] is not None:
                node = node.forward[lvl]
        if node is self._head:
            raise KeyError("max_item() called on an empty SkipList")
        return (node.key, node.value)

    def __repr__(self) -> str:
        parts = ", ".join(f"{k!r}: {v!r}" for k, v in self.items())
        return f"SkipList({{ {parts} }})" if parts else "SkipList({})"
