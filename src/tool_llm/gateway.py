from pathlib import Path

from src.config import AgentConfig
from src.tool_llm.timing import TimingRecorder, timed_section
from src.utils.json_io import write_json
from src.llmmanager import LLMManager


PARSED_LLM_MAX_ATTEMPTS = 5


def _attempt_path(response_path: Path, attempt: int) -> Path:
    return response_path.with_name(f"{response_path.stem}_attempt_{attempt:02d}{response_path.suffix}")


def _parse_errors_path(response_path: Path) -> Path:
    return response_path.with_name(f"{response_path.stem}_parse_errors.json")


def call_and_parse_llm_response(
    agent_cfg: AgentConfig,
    messages: list[dict],
    raw_response_path: Path,
    parse_response,
    label: str,
    timings: TimingRecorder | None = None,
    timing_module: str = "llm",
    timing_metadata: dict | None = None,
    llm_request_semaphore=None,
) -> tuple[str, dict]:
    errors = []
    last_error = None
    retry_messages = list(messages)
    route = "reflection" if timing_module == "episode_reflection" else "forecast"
    manager = LLMManager(agent_cfg, route=route, max_concurrent=1)
    for attempt in range(1, PARSED_LLM_MAX_ATTEMPTS + 1):
        with timed_section(
            timings,
            f"{timing_module}.request",
            attempt=attempt,
            **(timing_metadata or {}),
        ):
            raw_response = manager.complete_chat(
                retry_messages,
                request_slots=llm_request_semaphore,
                request_kind=route,
                timings=timings,
                timing_module=timing_module,
                timing_metadata={**(timing_metadata or {}), 'parse_attempt': attempt},
            )
        attempt_path = _attempt_path(raw_response_path, attempt)
        attempt_path.write_text(raw_response, encoding="utf-8")
        try:
            with timed_section(
                timings,
                f"{timing_module}.parse",
                attempt=attempt,
                **(timing_metadata or {}),
            ):
                parsed = parse_response(raw_response)
        except Exception as exc:
            last_error = exc
            errors.append(
                {
                    "attempt": attempt,
                    "raw_response_path": str(attempt_path),
                    "error": str(exc),
                }
            )
            if attempt < PARSED_LLM_MAX_ATTEMPTS:
                compact_retry_hint = ""
                # Reflection responses occasionally arrive as truncated JSON
                # (the decoder reports the position near the end of the
                # document).  Explicitly request a short response on the
                # next attempt; this reduces the chance that the provider
                # repeats the long, malformed object.
                if label.lower().startswith("episode reflection") and isinstance(exc, ValueError) and "char " in str(exc):
                    compact_retry_hint = (
                        " Keep every string to one short sentence and keep the complete JSON under 1200 characters; "
                        "do not repeat the input payload."
                    )
                print(
                    f"{label} parse failed, retrying: "
                    f"attempt={attempt}/{PARSED_LLM_MAX_ATTEMPTS}, raw_response_path={attempt_path}, error={exc}"
                )
                retry_messages = [
                    *messages,
                    {
                        "role": "user",
                        "content": (
                            "The previous response failed validation and was discarded. "
                            f"Validation error: {exc}. Regenerate the full response as exactly one JSON object "
                            "using the required schema. Do not return commentary, markdown, or a duplicate JSON object."
                            + compact_retry_hint
                        ),
                    },
                ]
            continue
        raw_response_path.write_text(raw_response, encoding="utf-8")
        if errors:
            write_json(_parse_errors_path(raw_response_path), errors)
        return raw_response, parsed
    write_json(_parse_errors_path(raw_response_path), errors)
    raise ValueError(
        f"Failed to parse {label} response after {PARSED_LLM_MAX_ATTEMPTS} attempts: {raw_response_path}"
    ) from last_error
