"""Strict bounded schema-one decoding for inert animated property data."""

import json
import math

MAX_JSON_CHARS = 1000000
MAX_NODES = 100000
MAX_DEPTH = 24
MAX_KEYFRAMES = 4096


def validate_payload(payload):
    """Reject non-JSON data, excessive depth, nonfinite numbers and large trees."""
    pending = [(payload, 0)]
    nodes = 0
    while pending:
        value, depth = pending.pop()
        nodes += 1
        if nodes > MAX_NODES or depth > MAX_DEPTH:
            raise ValueError("property payload exceeds size or depth budget")
        if type(value) is dict:
            if len(value) + len(pending) + nodes > MAX_NODES:
                raise ValueError("property payload exceeds node budget")
            if any(type(key) is not str for key in value):
                raise TypeError("JSON object keys must be strings")
            pending.extend((child, depth + 1) for child in value.values())
        elif type(value) is list:
            if len(value) + len(pending) + nodes > MAX_NODES:
                raise ValueError("property payload exceeds node budget")
            pending.extend((child, depth + 1) for child in value)
        elif type(value) in (int, float):
            if abs(value) > 1e100 or not math.isfinite(value):
                raise ValueError("JSON numeric values must be finite and bounded")
        elif type(value) is str:
            if len(value) > 8192:
                raise ValueError("JSON string exceeds budget")
        elif value is not None and type(value) is not bool:
            raise TypeError("payload must contain only inert JSON values")


def require_fields(value, fields, label):
    """Require an exact JSON object shape without ignoring unknown fields."""
    if type(value) is not dict or set(value) != set(fields):
        raise ValueError(f"invalid {label} fields")


def _pairs_to_dict(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def loads(text):
    """Parse bounded JSON and reject duplicate keys and nonfinite constants."""
    if type(text) is not str:
        raise TypeError("JSON input must be str")
    if len(text) > MAX_JSON_CHARS:
        raise ValueError("JSON input exceeds size budget")
    try:
        result = json.loads(
            text, object_pairs_hook=_pairs_to_dict, parse_constant=_reject_constant
        )
    except (RecursionError, OverflowError) as error:
        raise ValueError("JSON input exceeds parsing budget") from error
    validate_payload(result)
    return result


def dumps(payload):
    """Return deterministic JSON without NaN or Infinity extensions."""
    validate_payload(payload)
    result = json.dumps(payload, sort_keys=True, allow_nan=False)
    if len(result) > MAX_JSON_CHARS:
        raise ValueError("JSON output exceeds size budget")
    return result
