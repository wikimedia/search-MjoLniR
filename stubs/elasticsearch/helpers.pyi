from elasticsearch import Elasticsearch
from typing import Callable, Dict, Generator, Tuple


def streaming_bulk(
    client: Elasticsearch, actions,
    chunk_size: int=..., max_chunk_bytes: int=...,
    queue_size: int=..., expand_action_callback=Callable,
    **kwargs
) ->Generator[Tuple[bool, Dict], None, None]: ...
