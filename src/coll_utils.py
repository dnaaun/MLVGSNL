from __future__ import annotations
from functools import cached_property

from typing import (
    Dict,
    Sequence,
    Union,
    Mapping,
    MutableMapping,
    overload,
    Generic,
    TypeVar,
    Iterator,
    Iterable,
    Tuple,
    AbstractSet,
    ItemsView,
    KeysView,
    ValuesView,
)

_K = TypeVar("_K")
_V = TypeVar("_V")


class BiDict(MutableMapping[_K, _V]):
    @overload
    def __init__(self, __tuples: Iterable[Tuple[_K, _V]]) -> None:
        ...

    @overload
    def __init__(self, __dict: Dict[_K, _V], rev_dict: Dict[_V, _K] = None) -> None:
        ...

    def __init__(
        self,
        __item: Union[Iterable[Tuple[_K, _V]], Dict[_K, _V]],
        rev_dict: Dict[_V, _K] = None,
    ) -> None:
        if isinstance(__item, dict):
            self._dict = __item
        else:
            self._dict = {k: v for k, v in __item}

        self._rev_dict: "Dict[_V, _K]"  # help pyright
        if rev_dict is None:
            self._rev_dict = {v: k for k, v in self._dict.items()}
        else:
            self._rev_dict = rev_dict

    @cached_property
    def rev(self) -> BiDict[_V, _K]:
        return BiDict(self._rev_dict, self._dict)

    def __eq__(self, o: object) -> bool:
        return isinstance(o, BiDict) and self._dict == o._dict

    def __len__(self) -> int:
        return len(self._dict)

    def __getitem__(self, k: _K) -> _V:
        return self._dict[k]

    def __delitem__(self, k: _K) -> None:
        v = self._dict[k]
        del self._dict[k]
        del self._rev_dict[v]

    def __setitem__(self, k: _K, v: _V) -> None:
        self._dict[k] = v
        self._rev_dict[v] = k

    def items(self) -> ItemsView[_K, _V]:
        return self._dict.items()

    def keys(self) -> KeysView[_K]:
        return self._dict.keys()

    def values(self) -> ValuesView[_V]:
        return self._dict.values()

    def __iter__(self) -> Iterator[_K]:
        return iter(self._dict.keys())

    def __repr__(self) -> str:
        return f"BiDict({list(self._dict.items())})"


_Tco = TypeVar("_Tco", covariant=True)


class Ordering(Sequence[_Tco]):
    def __init__(self, __item: Iterable[_Tco]) -> None:
        self._bidict = BiDict(enumerate(__item))

    @cached_property
    def indices(self) -> Mapping[_Tco, int]:
        return self._bidict.rev

    @overload
    def __getitem__(self, __item: int) -> _Tco:
        ...

    @overload
    def __getitem__(self, __item: slice) -> Ordering[_Tco]:
        ...

    def __getitem__(self, __item: Union[int, slice]) -> Union[_Tco, Ordering[_Tco]]:
        if isinstance(__item, int):
            try:
                return self._bidict[__item]
            except KeyError:  # Be a true collection
                raise IndexError
        else:
            return Ordering(
                self[i] for i in range(__item.start, __item.stop, __item.step)
            )

    def __len__(self) -> int:
        return len(self._bidict)

    def __contains__(self, __item: object) -> bool:
        return __item in self._bidict.rev

    def __eq__(self, o: object) -> bool:
        return isinstance(o, Ordering) and self._bidict == o._bidict

    def __repr__(self) -> str:
        return f"Ordering({[self[i] for i in range(len(self))]})"


if __name__ == "__main__":
    BiDict({})
    Ordering([])
