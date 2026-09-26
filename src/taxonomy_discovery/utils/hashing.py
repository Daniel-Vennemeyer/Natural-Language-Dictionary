from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_hash(obj: Any, length: int = 8) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]