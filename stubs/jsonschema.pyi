from typing import Any, Iterator, Mapping


class Draft4Validator:
    def __init__(self, schema: Mapping) -> None: ...

    def iter_errors(self, instance: Mapping) -> Iterator[Any]: ...
