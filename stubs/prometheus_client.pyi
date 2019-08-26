from typing import Any, Callable, ContextManager, Iterable, TypeVar, Union

T = TypeVar('T')

def start_http_server(port: int, addr: str = '') -> None: ...

class Counter:
    def __init__(
        self, name: str, documentation: str, labelnames: Iterable[str] = ()
    ) -> None: ...

    def labels(self, *labelsvalues, **labelkwargs) -> Counter: ...

    def inc(self, amount: int = 1) -> None: ...


class Gauge:
    def __init__(
            self, name: str, documentation: str, labelnames: Iterable[str] = ()
    ) -> None: ...

    def set_function(self, f: Callable[[], float]) -> None: ...

class Summary:
    def __init__(
        self, name: str, documentation: str, labelnames: Iterable[str] = ()
    ) -> None: ...

    def time(self) -> Any: ...
