from elasticsearch import Elasticsearch
from typing import Callable, Dict, Generator, Tuple


def parallel_bulk(
    client: Elasticsearch, actions, thread_count: int=...,
    chunk_size: int=..., max_chunk_bytes: int=...,
    queue_size: int=..., expand_action_callback=Callable,
    **kwargs
) ->Generator[Tuple[bool, Dict], None, None]: ...
