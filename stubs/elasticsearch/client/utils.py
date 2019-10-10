from elasticsearch import Elasticsearch
from elasticsearch.transport import Transport

from typing import Callable, List, Optional, TypeVar

_T = TypeVar('_T')

class NamespacedClient:
    def __init__(self, client: Elasticsearch) -> None:
        ...

    @property
    def client(self) -> Elasticsearch:
        ...

    @property
    def transport(self) -> Transport:
        ...

class AddonClient(NamespacedClient):
    ...

def query_params(*es_query_params: str) -> Callable[[_T], _T]:
    ...

def _make_path(*parts: Optional[str]) -> str:
    ...

