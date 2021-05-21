import json
from pathlib import Path
from typing import Type, TypeVar


# Makes a dataclass loadable from a json file.


# https://stackoverflow.com/a/44644576
# Create a generic variable that can be "Parent", or any subclass.
LoadableJsonType = TypeVar("LoadableJsonType", bound="LoadableJson")


class LoadableJson:
    @classmethod
    def from_json_bytes(cls: Type[LoadableJsonType], json_bytes: bytes) -> LoadableJsonType:
        json_data = json.loads(json_bytes)
        instance = cls(**json_data)  # type: ignore

        return instance

    @classmethod
    def from_json_path(cls: Type[LoadableJsonType], json_path: Path) -> LoadableJsonType:
        json_data = json_path.read_bytes()

        return cls.from_json_bytes(json_data)
