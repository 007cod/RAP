import json
import re


_ESCAPED_TRAILING_WHITESPACE = re.compile(r"^(?:\\[nrt]\s*)+$")
# Some providers append a closing Markdown fence even without an opening
# fence, or emit only two backticks. Accept only that isolated suffix after
# the JSON decoder has successfully consumed a complete object.
_CLOSING_MARKDOWN_FENCE = re.compile(r"`{2,3}")


def _is_response_suffix(text: str) -> bool:
    return (
        not text
        or _ESCAPED_TRAILING_WHITESPACE.fullmatch(text) is not None
        or _CLOSING_MARKDOWN_FENCE.fullmatch(text) is not None
    )


def parse_json_object_response(raw_response: str, label: str) -> dict:
    if not isinstance(raw_response, str):
        raise TypeError(f"{label} raw response must be a string, got {type(raw_response).__name__}")
    text = raw_response.strip()
    start = text.find("{")
    if start < 0:
        raise ValueError(f"{label} response does not contain a JSON object: {text[:300]!r}")
    parsed, end = json.JSONDecoder().raw_decode(text[start:])
    end = start + end
    remaining = text[end:].strip()
    if _is_response_suffix(remaining):
        remaining = ""
    else:
        remaining = _consume_duplicate_json_objects(remaining, parsed, label)
    if remaining:
        raise ValueError(
            f"{label} response contains extra non-JSON data after the first object at char {end}: "
            f"{remaining[:300]!r}"
        )
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} response must be a JSON object, got {type(parsed).__name__}")
    return parsed


def _consume_duplicate_json_objects(text: str, expected: dict, label: str) -> str:
    decoder = json.JSONDecoder()
    remaining = text.strip()
    while remaining:
        if _is_response_suffix(remaining):
            return ""
        if not remaining.startswith("{"):
            return remaining
        duplicate, end = decoder.raw_decode(remaining)
        if duplicate != expected:
            raise ValueError(f"{label} response contains a second JSON object that differs from the first object")
        remaining = remaining[end:].strip()
    return ""
