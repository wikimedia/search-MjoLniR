from typing import Any, Mapping, Optional

class Transport:
    def perform_request(
        self, method: str, url: str,
        params: Optional[Mapping[str, Any]] = None,
        body: Optional[Mapping[str, Any]] = None
    ) -> Mapping:
        ...
