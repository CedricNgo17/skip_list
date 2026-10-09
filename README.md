# skip_list

A probabilistically balanced ordered map. Keys are kept in ascending order and support O(log n) average-time lookup, insertion, deletion, and range iteration. Standard library only; no dependencies.

## Usage

```python
from skip_list import SkipList

sl = SkipList()
sl[3] = "three"
sl[1] = "one"
sl[2] = "two"

assert sl[2] == "two"          # lookup
assert list(sl) == [1, 2, 3]    # keys in ascending order

del sl[2]
assert list(sl) == [1, 3]

for key, value in sl.range(1, 3):  # half-open [1, 3)
    ...
```

## Why this exists

A skip list gives you ordered-map operations at the same expected cost as a balanced tree, but without rebalancing machinery. The structure is a tower of sorted linked lists: most nodes live only at the bottom level, a quarter also appear one level up, an eighth two levels up, and so on. Search walks the top levels to skip long runs, then drops down. Heights come from coin flips, so any single operation can be O(n) in the worst case — but with vanishing probability under an honest random source.

The trade-off versus a red-black tree: simpler code, no rebalancing, and trivial concurrent variants, in exchange for a probabilistic bound instead of a deterministic one and O(n) worst case.

## Edge cases you will hit

- **Duplicate keys are not stored separately.** Setting an existing key replaces its value, matching plain `dict` semantics. The existing node's tower height is preserved so the random structure is not perturbed by updates.
- **Keys must be totally ordered by `<`.** Mixing incomparable key types (e.g. `int` and `str` in the same map) raises `TypeError` at comparison time, same as Python's built-in sorted containers.
- **`range()` defaults to half-open `[start, stop)`.** Pass `inclusive_start` / `inclusive_stop` to express the other three combinations. `None` for either bound means "no bound on that side".
- **`min_item()` / `max_item()` raise `KeyError` on an empty map.** They do not return a sentinel; callers must handle the empty case.
- **`max_item()` is O(log n) in expectation but walks the full spine of the list**, because skip lists carry forward pointers only. If you need O(1) maxima, maintain your own pointer to the tail.

## Exports

- `SkipList` — the ordered map class.

## Running the tests

```sh
PYTHONPATH=src python -m unittest discover -s tests
```

## Performance

The window keeps a bounded buffer, so `push` is constant time and memory does not
grow with the length of the stream. `peak` and `trough` are linear in the window
size, which is the trade that keeps `push` cheap.

