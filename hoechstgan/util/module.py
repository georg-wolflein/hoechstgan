
import ast
import typing
from torch import nn


class SerializedModuleDict(nn.ModuleDict):

    K = typing.TypeVar("K")

    @classmethod
    def serialize_key(cls, key: K) -> str:
        s = repr(key)
        if key != cls.deserialize_key(s):
            raise ValueError(
                f"Invalid serialization: {key} vs {cls.deserialize_key(s)}")
        return s

    @classmethod
    def deserialize_key(cls, serialized: str) -> K:
        return ast.literal_eval(serialized)

    def __getitem__(self, key: K) -> nn.Module:
        return super().__getitem__(self.serialize_key(key))

    def __setitem__(self, key: K, module: nn.Module) -> None:
        super().__setitem__(self.serialize_key(key), module)

    def __delitem__(self, key: K) -> None:
        super().__delitem__(self.serialize_key(key))

    def __contains__(self, key: K) -> bool:
        return super().__contains__(self.serialize_key(key))

    def keys(self) -> typing.Iterable[K]:
        return set(map(self.deserialize_key, super().keys()))

    def items(self) -> typing.Iterable[typing.Tuple[K, nn.Module]]:
        return [(self.deserialize_key(k), v)
                for (k, v) in super().items()]


class PairedSerializedModuleDict(SerializedModuleDict):

    K = typing.TypeVar("K")
    _SEPARATOR = "=>"

    @classmethod
    def serialize_key(cls, key: K) -> str:
        s = cls._SEPARATOR.join(map(repr, key))
        if key != cls.deserialize_key(s):
            raise ValueError(
                f"Invalid serialization: {key} vs {cls.deserialize_key(s)}")
        return s

    @classmethod
    def deserialize_key(cls, serialized: str) -> K:
        return tuple(map(ast.literal_eval, serialized.split(cls._SEPARATOR)))
