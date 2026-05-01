"""Build per-user, per-phase universal chat datasets from synthesis logs.

This module keeps source files unchanged and writes normalized JSON outputs that
can replace legacy chat exports after parser compatibility updates.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
SYNTHESIS_DIR = REPO_ROOT / "synthesis"
COPILOT_DIR = SYNTHESIS_DIR / "copilot_data"
KIRO_DIR = SYNTHESIS_DIR / "kiro_data"
UNIVERSAL_DIR = SYNTHESIS_DIR / "universal"

PHASES = ["Phase 1", "Phase 2", "Phase 3"]
COPILOT_EDIT_TOOL_NAMES = {
    "apply_patch",
    "replace_string_in_file",
    "multi_replace_string_in_file",
    "create_file",
}
NEGATIVE_KEYWORDS = [
    "error",
    "fix",
    "no",
    "wrong",
    "fail",
    "broken",
    "bug",
    "issue",
    "doesn't work",
    "didn't work",
    "restore",
]
TOOL_FOLLOWUP_PREFIX = "[Tool Result Follow-up]"
REDACTED_SOURCE_PLACEHOLDER = "[Redacted by source log]"
COPILOT_PATCH_LINE_PATTERN = re.compile(r"Generating patch \((\d+) lines?\)", re.IGNORECASE)


@dataclass
class ChatTurn:
    timestamp: str | None
    session_id: str
    phase: str
    user_text: str
    assistant_text: str
    model: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    actual_prompt_tokens: int | None
    actual_completion_tokens: int | None
    estimated_prompt_tokens: int | None
    estimated_completion_tokens: int | None
    token_count_method: str | None
    code_lines_suggested: int | None
    assistant_code_lines: int | None
    tool_code_lines: int | None
    code_line_breakdown: dict[str, int]
    latency_ms: int | None
    ttft_ms: int | None
    referenced_files: list[str]
    languages: list[str]
    edited_file_events: int | None
    checkpoints_restored: int | None
    tool_calls: list[dict[str, Any]]
    metering_usage: float | None
    metering_unit: str | None
    context_usage_pct: float | None
    source_platform: str
    source_file: str
    source_ref: dict[str, Any]
    copilot_applied_edit_lines: int | None = None
    copilot_applied_edit_calls: int | None = None
    copilot_text_edit_groups: int | None = None
    copilot_distinct_edited_files: int | None = None
    copilot_edit_tool_counts: dict[str, int] = field(default_factory=dict)
    copilot_edited_files: list[str] = field(default_factory=list)


def _safe_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(value)
    except Exception:
        return 0


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _extract_code_metadata(text: str) -> tuple[int, list[str]]:
    if not text or "```" not in text:
        return 0, []

    code_lines = 0
    langs: list[str] = []
    in_block = False
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            if not in_block:
                lang = line.strip().replace("```", "").strip()
                if lang:
                    langs.append(lang)
            in_block = not in_block
        elif in_block:
            code_lines += 1

    return code_lines, langs


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    compact = text.strip()
    if not compact:
        return 0
    # Rough fallback for sources that do not expose exact token counts.
    return int(math.ceil(len(compact) / 4.0))


def _line_count(text: str) -> int:
    if not text:
        return 0
    lines = text.splitlines()
    return len(lines) if lines else 1


def _extract_uri_path(value: Any) -> str | None:
    if isinstance(value, dict):
        for key in ("fsPath", "path", "external"):
            inner = value.get(key)
            if isinstance(inner, str) and inner.strip():
                return inner
    if isinstance(value, str) and value.strip():
        return value
    return None


def _diff_added_lines(old_text: str, new_text: str) -> int:
    old_lines = (old_text or "").splitlines()
    new_lines = (new_text or "").splitlines()
    matcher = SequenceMatcher(a=old_lines, b=new_lines)
    added = 0
    for tag, _, _, j1, j2 in matcher.get_opcodes():
        if tag in {"insert", "replace"}:
            added += max(0, j2 - j1)
    return added


def _apply_patch_added_lines(patch_text: str) -> tuple[int, list[str]]:
    added = 0
    touched_paths: list[str] = []

    for line in (patch_text or "").splitlines():
        if line.startswith("*** Update File: "):
            touched_paths.append(line.replace("*** Update File: ", "", 1).strip())
            continue
        if line.startswith("*** Add File: "):
            touched_paths.append(line.replace("*** Add File: ", "", 1).strip())
            continue
        if line.startswith("*** Move to: "):
            touched_paths.append(line.replace("*** Move to: ", "", 1).strip())
            continue
        if line.startswith("+++ ") or line.startswith("--- ") or line.startswith("@@"):
            continue
        if line.startswith("+"):
            added += 1

    return added, _dedupe_preserve_order(touched_paths)


def _text_edit_group_lines(edit_groups: Any) -> int:
    total = 0

    if not isinstance(edit_groups, list):
        return total

    for edit_group in edit_groups:
        group_items: list[Any]
        if isinstance(edit_group, list):
            group_items = edit_group
        elif isinstance(edit_group, dict):
            group_items = [edit_group]
        else:
            continue

        for edit in group_items:
            if not isinstance(edit, dict):
                continue
            text = edit.get("text")
            if isinstance(text, str) and text != "":
                total += _line_count(text)

    return total


def _copilot_tool_call_summary(
    name: str,
    args: dict[str, Any],
    line_count: int,
    touched_files: list[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if touched_files:
        summary["paths"] = touched_files
    if line_count > 0:
        summary["applied_lines"] = line_count
    if name == "multi_replace_string_in_file":
        replacements = args.get("replacements")
        if isinstance(replacements, list):
            summary["replacement_count"] = len(replacements)
    return summary


def _extract_copilot_tool_calls_and_metrics(
    response_parts: list[dict[str, Any]],
    result_meta: dict[str, Any],
    edited_file_events: list[dict[str, Any]],
) -> dict[str, Any]:
    tool_calls: list[dict[str, Any]] = []
    edit_tool_counts: dict[str, int] = defaultdict(int)
    edited_files: list[str] = []
    applied_edit_lines = 0
    applied_edit_calls = 0
    text_edit_groups = 0
    text_edit_lines_fallback = 0
    patch_event_lines = 0

    rounds = result_meta.get("toolCallRounds", []) if isinstance(result_meta, dict) else []
    if not isinstance(rounds, list):
        rounds = []

    for round_index, round_obj in enumerate(rounds):
        if not isinstance(round_obj, dict):
            continue
        raw_calls = round_obj.get("toolCalls", [])
        if not isinstance(raw_calls, list):
            continue

        for call_index, raw_call in enumerate(raw_calls):
            if not isinstance(raw_call, dict):
                continue

            name = str(raw_call.get("name") or "unknown")
            args = _coerce_tool_args(raw_call.get("arguments"))
            line_count = 0
            touched_files = _dedupe_preserve_order(_extract_paths_from_args(args))

            if name in COPILOT_EDIT_TOOL_NAMES:
                edit_tool_counts[name] += 1
                applied_edit_calls += 1

                if name == "apply_patch":
                    patch_input = str(args.get("input") or "")
                    line_count, patch_files = _apply_patch_added_lines(patch_input)
                    if patch_files:
                        touched_files = _dedupe_preserve_order(touched_files + patch_files)
                elif name == "replace_string_in_file":
                    line_count = _diff_added_lines(
                        str(args.get("oldString") or ""),
                        str(args.get("newString") or ""),
                    )
                elif name == "multi_replace_string_in_file":
                    replacements = args.get("replacements", [])
                    if isinstance(replacements, list):
                        for replacement in replacements:
                            if not isinstance(replacement, dict):
                                continue
                            line_count += _diff_added_lines(
                                str(replacement.get("oldString") or ""),
                                str(replacement.get("newString") or ""),
                            )
                elif name == "create_file":
                    content = (
                        args.get("content")
                        or args.get("text")
                        or args.get("fileContent")
                        or args.get("contents")
                        or ""
                    )
                    line_count = _line_count(str(content))

                applied_edit_lines += line_count
                edited_files.extend(touched_files)

            tool_calls.append(
                {
                    "id": raw_call.get("id"),
                    "name": name,
                    "args": _copilot_tool_call_summary(name, args, line_count, _basename_refs(touched_files)),
                    "index": len(tool_calls),
                    "round_index": round_index,
                    "call_index": call_index,
                }
            )

    for part in response_parts:
        if not isinstance(part, dict):
            continue

        if part.get("toolId") == "copilot_applyPatch":
            invocation = part.get("invocationMessage")
            invocation_text = invocation.get("value") if isinstance(invocation, dict) else ""
            match = COPILOT_PATCH_LINE_PATTERN.search(str(invocation_text or ""))
            if match:
                try:
                    patch_event_lines += int(match.group(1))
                except Exception:
                    pass

        if part.get("kind") == "textEditGroup":
            text_edit_groups += 1
            text_edit_lines_fallback += _text_edit_group_lines(part.get("edits"))
            uri_path = _extract_uri_path(part.get("uri"))
            if uri_path:
                edited_files.append(uri_path)

        if part.get("kind") == "codeblockUri" and part.get("isEdit"):
            uri_path = _extract_uri_path(part.get("uri"))
            if uri_path:
                edited_files.append(uri_path)

    for event in edited_file_events:
        if not isinstance(event, dict):
            continue
        uri_path = _extract_uri_path(event.get("uri"))
        if uri_path:
            edited_files.append(uri_path)

    edited_files = _dedupe_preserve_order(edited_files)

    if applied_edit_lines <= 0 and text_edit_lines_fallback > 0:
        applied_edit_lines = text_edit_lines_fallback
    if applied_edit_calls <= 0 and text_edit_groups > 0:
        applied_edit_calls = text_edit_groups
    if applied_edit_lines <= 0 and patch_event_lines > 0:
        applied_edit_lines = patch_event_lines
    if applied_edit_calls <= 0 and edited_file_events:
        applied_edit_calls = len(edited_file_events)

    return {
        "tool_calls": tool_calls,
        "applied_edit_lines": applied_edit_lines,
        "applied_edit_calls": applied_edit_calls,
        "text_edit_groups": text_edit_groups,
        "edited_files": edited_files,
        "distinct_edited_files": len(edited_files),
        "edit_tool_counts": dict(sorted(edit_tool_counts.items())),
        "text_edit_lines_fallback": text_edit_lines_fallback,
        "patch_event_lines": patch_event_lines,
    }


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _basename_refs(values: list[str]) -> list[str]:
    refs: list[str] = []
    for value in values:
        try:
            refs.append(os.path.basename(value))
        except Exception:
            continue
    return _dedupe_preserve_order(refs)


def _extract_paths_from_args(value: Any) -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, inner in value.items():
            lowered = str(key).lower()
            if lowered in {"path", "filepath"} and isinstance(inner, str):
                paths.append(inner)
                continue
            if lowered == "paths" and isinstance(inner, list):
                paths.extend(str(item) for item in inner if isinstance(item, str))
                continue
            if lowered in {"contextfiles", "files"} and isinstance(inner, list):
                for item in inner:
                    if isinstance(item, dict):
                        item_path = item.get("path")
                        if isinstance(item_path, str):
                            paths.append(item_path)
                continue
            paths.extend(_extract_paths_from_args(inner))
    elif isinstance(value, list):
        for item in value:
            paths.extend(_extract_paths_from_args(item))
    return paths


def _coerce_tool_args(raw_args: Any) -> dict[str, Any]:
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {"raw": raw_args}
    return {}


def _extract_tool_calls(
    response_obj: dict[str, Any],
    response_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = []

    kwargs = (
        ((response_obj.get("fullMessage") or {}).get("message") or {}).get("kwargs")
        if isinstance(response_obj, dict)
        else {}
    )
    if not isinstance(kwargs, dict):
        kwargs = {}

    for idx, tool_call in enumerate(kwargs.get("tool_calls", []) or []):
        if not isinstance(tool_call, dict):
            continue
        name = tool_call.get("name")
        if not name:
            continue
        tool_calls.append(
            {
                "id": tool_call.get("id"),
                "name": str(name),
                "args": _coerce_tool_args(tool_call.get("args")),
                "index": idx,
            }
        )

    if tool_calls:
        return tool_calls

    tool_acc: dict[str, str] = defaultdict(str)
    tool_meta: dict[str, dict[str, Any]] = {}
    for ev in response_events:
        if not isinstance(ev, dict):
            continue
        tue = ev.get("toolUseEvent")
        if not isinstance(tue, dict):
            continue
        tool_use_id = tue.get("toolUseId")
        if not tool_use_id:
            continue
        entry = tool_meta.setdefault(
            str(tool_use_id),
            {"id": tool_use_id, "name": tue.get("name"), "args": {}, "index": len(tool_meta)},
        )
        if tue.get("name"):
            entry["name"] = tue.get("name")
        chunk = tue.get("input")
        if isinstance(chunk, str):
            tool_acc[str(tool_use_id)] += chunk

    for tool_use_id, entry in tool_meta.items():
        args = _coerce_tool_args(tool_acc.get(tool_use_id, ""))
        tool_calls.append(
            {
                "id": entry.get("id"),
                "name": str(entry.get("name") or "unknown"),
                "args": args,
                "index": entry.get("index"),
            }
        )

    return tool_calls


def _tool_code_metrics(tool_calls: list[dict[str, Any]]) -> tuple[int, dict[str, int], list[str]]:
    breakdown = {
        "fs_write_lines": 0,
        "fs_append_lines": 0,
        "str_replace_lines": 0,
    }
    referenced_paths: list[str] = []

    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        name = str(tool_call.get("name") or "")
        args = tool_call.get("args")
        if not isinstance(args, dict):
            args = {}

        referenced_paths.extend(_extract_paths_from_args(args))

        if name == "fsWrite":
            payload = args.get("text") or args.get("content") or ""
            if isinstance(payload, str):
                breakdown["fs_write_lines"] += _line_count(payload)
        elif name == "fsAppend":
            payload = args.get("text") or args.get("content") or ""
            if isinstance(payload, str):
                breakdown["fs_append_lines"] += _line_count(payload)
        elif name == "strReplace":
            new_str = args.get("newStr")
            if not isinstance(new_str, str):
                new_str = args.get("new_str") if isinstance(args.get("new_str"), str) else ""
            old_str = args.get("oldStr")
            if not isinstance(old_str, str):
                old_str = args.get("old_str") if isinstance(args.get("old_str"), str) else ""
            if new_str and new_str != old_str:
                breakdown["str_replace_lines"] += _line_count(new_str)

    total = sum(breakdown.values())
    return total, breakdown, referenced_paths


def _phase_from_name(name: str) -> str:
    lower = name.lower()
    if "phase 3" in lower or "phase3" in lower or "p3" in lower:
        return "Phase 3"
    if "phase 2" in lower or "phase2" in lower or "p2" in lower:
        return "Phase 2"
    return "Phase 1"


def _discover_users() -> set[str]:
    users: set[str] = set()

    if COPILOT_DIR.exists():
        for child in COPILOT_DIR.iterdir():
            if child.is_dir():
                # New layout: copilot_data/<phase>/<user>/*.json
                user_children = [grandchild for grandchild in child.iterdir() if grandchild.is_dir()]
                if user_children:
                    for user_dir in user_children:
                        users.add(user_dir.name.lower())
                else:
                    # Backward-compatible fallback: copilot_data/<user>/*.json
                    users.add(child.name.lower())

    if KIRO_DIR.exists():
        for child in KIRO_DIR.iterdir():
            if child.is_dir():
                # New layout: kiro_data/<user>/<run>/...
                users.add(child.name.lower())

    return users


def _parse_copilot_user(user: str) -> dict[str, list[ChatTurn]]:
    by_phase: dict[str, list[ChatTurn]] = {phase: [] for phase in PHASES}

    candidate_dirs: list[tuple[str, Path]] = []
    if COPILOT_DIR.exists():
        for phase_dir in sorted([p for p in COPILOT_DIR.iterdir() if p.is_dir()]):
            user_dir = phase_dir / user
            if not user_dir.exists():
                user_dir = phase_dir / user.capitalize()
            if user_dir.exists() and user_dir.is_dir():
                candidate_dirs.append((_phase_from_name(phase_dir.name), user_dir))

    # Backward-compatible fallback: copilot_data/<user>/*.json
    fallback_user_dir = COPILOT_DIR / user.capitalize()
    if not fallback_user_dir.exists():
        fallback_user_dir = COPILOT_DIR / user
    if fallback_user_dir.exists() and fallback_user_dir.is_dir():
        candidate_dirs.append(("", fallback_user_dir))

    if not candidate_dirs:
        return by_phase

    for phase_hint, user_dir in candidate_dirs:
        for file_path in sorted(user_dir.glob("*.json")):
            phase = phase_hint or _phase_from_name(file_path.name)
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            requests = data.get("requests", [])
            if not isinstance(requests, list):
                continue

            for idx, req in enumerate(requests):
                if not isinstance(req, dict):
                    continue

                ts_ms = req.get("timestamp")
                if not ts_ms:
                    continue
                try:
                    ts = datetime.fromtimestamp(float(ts_ms) / 1000.0).isoformat()
                except Exception:
                    ts = None

                message_obj = req.get("message", {})
                if not isinstance(message_obj, dict):
                    message_obj = {}
                user_text = str(message_obj.get("text") or "")

                response_parts = req.get("response", [])
                if not isinstance(response_parts, list):
                    response_parts = []

                assistant_text_parts: list[str] = []
                for part in response_parts:
                    if not isinstance(part, dict):
                        continue
                    if part.get("kind") == "thinking":
                        continue
                    val = part.get("value")
                    if isinstance(val, str):
                        assistant_text_parts.append(val)
                assistant_text = "".join(assistant_text_parts)

                result = req.get("result", {})
                if not isinstance(result, dict):
                    result = {}
                usage = result.get("usage", {}) if isinstance(result.get("usage", {}), dict) else {}
                timings = result.get("timings", {}) if isinstance(result.get("timings", {}), dict) else {}
                result_meta = result.get("metadata", {}) if isinstance(result.get("metadata", {}), dict) else {}

                model = result.get("details") or req.get("modelId") or "Unknown"
                assistant_code_lines, languages = _extract_code_metadata(assistant_text)

                variable_data = req.get("variableData", {})
                variables = variable_data.get("variables", []) if isinstance(variable_data, dict) else []
                referenced_files: list[str] = []
                if isinstance(variables, list):
                    for var in variables:
                        if not isinstance(var, dict):
                            continue
                        val = var.get("value", {})
                        if not isinstance(val, dict):
                            continue
                        p = val.get("fsPath") or val.get("path")
                        if p:
                            referenced_files.append(os.path.basename(str(p)))

                edited_file_events = req.get("editedFileEvents", [])
                if not isinstance(edited_file_events, list):
                    edited_file_events = []

                copilot_edit_metrics = _extract_copilot_tool_calls_and_metrics(
                    response_parts=response_parts,
                    result_meta=result_meta,
                    edited_file_events=edited_file_events,
                )
                tool_code_lines = _safe_int(copilot_edit_metrics.get("applied_edit_lines"))
                total_ai_lines = tool_code_lines if tool_code_lines > 0 else assistant_code_lines
                referenced_files = _dedupe_preserve_order(
                    referenced_files + _basename_refs(list(copilot_edit_metrics.get("edited_files", [])))
                )

                checkpoints_restored = 1 if any(
                    isinstance(p, dict) and p.get("kind") == "undoStop" for p in response_parts
                ) else 0

                by_phase[phase].append(
                    ChatTurn(
                        timestamp=ts,
                        session_id=file_path.name,
                        phase=phase,
                        user_text=user_text,
                        assistant_text=assistant_text,
                        model=str(model),
                        prompt_tokens=_safe_int(usage.get("promptTokens")) if usage else None,
                        completion_tokens=_safe_int(usage.get("completionTokens")) if usage else None,
                        actual_prompt_tokens=_safe_int(usage.get("promptTokens")) if usage else None,
                        actual_completion_tokens=_safe_int(usage.get("completionTokens")) if usage else None,
                        estimated_prompt_tokens=None,
                        estimated_completion_tokens=None,
                        token_count_method="actual",
                        code_lines_suggested=total_ai_lines,
                        assistant_code_lines=assistant_code_lines,
                        tool_code_lines=tool_code_lines,
                        code_line_breakdown={
                            "assistant_code_lines": assistant_code_lines,
                            "tool_code_lines": tool_code_lines,
                            "copilot_applied_edit_lines": tool_code_lines,
                            "copilot_text_edit_lines_fallback": _safe_int(
                                copilot_edit_metrics.get("text_edit_lines_fallback")
                            ),
                            "copilot_patch_event_lines": _safe_int(
                                copilot_edit_metrics.get("patch_event_lines")
                            ),
                        },
                        latency_ms=_safe_int(timings.get("totalElapsed")) if timings else None,
                        ttft_ms=_safe_int(timings.get("firstProgress")) if timings else None,
                        referenced_files=referenced_files,
                        languages=languages,
                        edited_file_events=len(edited_file_events),
                        checkpoints_restored=checkpoints_restored,
                        tool_calls=list(copilot_edit_metrics.get("tool_calls", [])),
                        metering_usage=None,
                        metering_unit=None,
                        context_usage_pct=None,
                        source_platform="copilot",
                        source_file=str(file_path.relative_to(REPO_ROOT)),
                        source_ref={"request_index": idx},
                        copilot_applied_edit_lines=tool_code_lines,
                        copilot_applied_edit_calls=_safe_int(copilot_edit_metrics.get("applied_edit_calls")),
                        copilot_text_edit_groups=_safe_int(copilot_edit_metrics.get("text_edit_groups")),
                        copilot_distinct_edited_files=_safe_int(
                            copilot_edit_metrics.get("distinct_edited_files")
                        ),
                        copilot_edit_tool_counts=dict(copilot_edit_metrics.get("edit_tool_counts", {})),
                        copilot_edited_files=list(copilot_edit_metrics.get("edited_files", [])),
                    )
                )

    return by_phase


def _extract_json_lines(path: Path) -> list[dict[str, Any]]:
    objs: list[dict[str, Any]] = []
    if not path.exists():
        return objs
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        json_start = line.find("{")
        if json_start < 0:
            continue
        try:
            parsed = json.loads(line[json_start:])
            if isinstance(parsed, dict):
                if json_start > 0:
                    parsed["_source_prefix"] = line[:json_start].strip()
                objs.append(parsed)
        except Exception:
            continue
    return objs


def _parse_prefixed_line_timestamp(prefix: str | None) -> str | None:
    if not prefix:
        return None
    match = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})", prefix.strip())
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S.%f").isoformat()
    except Exception:
        return None


def _parse_kiro_timestamps(kiro_logs_path: Path) -> list[str]:
    timestamps: list[str] = []
    if not kiro_logs_path.exists():
        return timestamps

    pattern = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})")
    for line in kiro_logs_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "Triggered new agent" not in line:
            continue
        m = pattern.match(line)
        if not m:
            continue
        try:
            dt = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f")
            timestamps.append(dt.isoformat())
        except Exception:
            continue
    return timestamps


def _count_fswrite_lines(text: str) -> int:
    if not text:
        return 0
    count = 0
    pattern = re.compile(
        r"<atml:invoke name=\"fsWrite\">.*?<atml:parameter name=\"contents\">(.*?)</atml:parameter>",
        re.DOTALL,
    )
    for m in pattern.finditer(text):
        contents = m.group(1)
        count += len(contents.splitlines())
    return count


def _normalize_kiro_text(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    if raw == "***SensitiveInformation***":
        return "[Redacted by source log]"
    return raw


def _kiro_text_is_redacted(text: str) -> bool:
    normalized = _normalize_kiro_text(text)
    return normalized == "[Redacted by source log]"


def _extract_q_client_tool_calls(tool_uses: Any) -> list[dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = []
    if not isinstance(tool_uses, list):
        return tool_calls
    for idx, tool_use in enumerate(tool_uses):
        if not isinstance(tool_use, dict):
            continue
        tool_calls.append(
            {
                "id": tool_use.get("toolUseId"),
                "name": str(tool_use.get("name") or "unknown"),
                "args": _coerce_tool_args(tool_use.get("input")),
                "index": idx,
            }
        )
    return tool_calls


def _extract_kiro_user_message_text(message: dict[str, Any]) -> str:
    text = _normalize_kiro_text(message.get("content") or "")

    docs = message.get("documents")
    if not text and isinstance(docs, list) and docs:
        names = []
        for doc in docs:
            if isinstance(doc, dict) and doc.get("name"):
                names.append(str(doc["name"]))
        if names:
            text = "[Document Context]\n" + "\n".join(names)

    ctx = message.get("userInputMessageContext", {})
    tool_results = ctx.get("toolResults", []) if isinstance(ctx, dict) else []
    if not text and tool_results:
        lines = ["[Tool Result Follow-up]"]
        for tr in tool_results:
            if not isinstance(tr, dict):
                continue
            status = tr.get("status")
            if status:
                lines.append(f"status={status}")
            for content in tr.get("content", []) or []:
                if isinstance(content, dict) and content.get("text"):
                    lines.append(_normalize_kiro_text(content["text"]))
        text = "\n".join(lines)

    return text


def _parse_q_client_history_turns(history: Any) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    if not isinstance(history, list):
        return turns

    pending_user: dict[str, Any] | None = None
    for item in history:
        if not isinstance(item, dict):
            continue

        uim = item.get("userInputMessage")
        if isinstance(uim, dict):
            text = _extract_kiro_user_message_text(uim)
            if text:
                pending_user = {
                    "text": text,
                    "model": uim.get("modelId"),
                }
            continue

        arm = item.get("assistantResponseMessage")
        if not isinstance(arm, dict) or pending_user is None:
            continue

        turns.append(
            {
                "user_text": pending_user.get("text") or "",
                "model": pending_user.get("model"),
                "assistant_text": _normalize_kiro_text(arm.get("content") or ""),
                "tool_calls": _extract_q_client_tool_calls(arm.get("toolUses")),
            }
        )
        pending_user = None

    return turns


def _parse_q_client_user(
    user: str,
    run_dir: Path,
    q_client_path: Path,
    phase: str,
) -> list[ChatTurn]:
    turns: list[ChatTurn] = []
    seen_turns: set[tuple[str, int]] = set()
    records = _extract_json_lines(q_client_path)

    run_base_dt: datetime | None = None
    try:
        run_base_dt = datetime.strptime(run_dir.name, "%Y%m%dT%H%M%S")
    except Exception:
        run_base_dt = None

    fallback_index = 0
    for rec_idx, rec in enumerate(records):
        if rec.get("commandName") != "GenerateAssistantResponseCommand":
            continue

        input_obj = rec.get("input")
        if not isinstance(input_obj, dict):
            continue
        state = input_obj.get("conversationState")
        if not isinstance(state, dict):
            continue

        conversation_id = str(state.get("conversationId") or "")
        if not conversation_id:
            continue

        history_turns = _parse_q_client_history_turns(state.get("history"))
        source_ts = _parse_prefixed_line_timestamp(rec.get("_source_prefix"))

        for turn_index, parsed_turn in enumerate(history_turns):
            turn_key = (conversation_id, turn_index)
            if turn_key in seen_turns:
                continue
            seen_turns.add(turn_key)

            user_text = str(parsed_turn.get("user_text") or "")
            assistant_text = str(parsed_turn.get("assistant_text") or "")
            tool_calls = parsed_turn.get("tool_calls") if isinstance(parsed_turn.get("tool_calls"), list) else []

            any_redaction = _kiro_text_is_redacted(user_text) or _kiro_text_is_redacted(assistant_text)
            estimated_prompt_tokens = None if any_redaction else _estimate_tokens(user_text)
            estimated_completion_tokens = None if any_redaction else _estimate_tokens(assistant_text)
            token_method = "redacted_source_unavailable" if any_redaction else "estimated_char_ratio_4"

            timestamp = source_ts
            if not timestamp and run_base_dt is not None:
                timestamp = (run_base_dt + timedelta(seconds=fallback_index)).isoformat()
            fallback_index += 1

            turns.append(
                ChatTurn(
                    timestamp=timestamp,
                    session_id=run_dir.name,
                    phase=phase,
                    user_text=user_text,
                    assistant_text=assistant_text,
                    model=str(parsed_turn.get("model")) if parsed_turn.get("model") is not None else None,
                    prompt_tokens=estimated_prompt_tokens,
                    completion_tokens=estimated_completion_tokens,
                    actual_prompt_tokens=None,
                    actual_completion_tokens=None,
                    estimated_prompt_tokens=estimated_prompt_tokens,
                    estimated_completion_tokens=estimated_completion_tokens,
                    token_count_method=token_method,
                    code_lines_suggested=0,
                    assistant_code_lines=0,
                    tool_code_lines=0,
                    code_line_breakdown={"assistant_code_lines": 0, "tool_code_lines": 0},
                    latency_ms=None,
                    ttft_ms=None,
                    referenced_files=[],
                    languages=[],
                    edited_file_events=None,
                    checkpoints_restored=None,
                    tool_calls=tool_calls,
                    metering_usage=None,
                    metering_unit=None,
                    context_usage_pct=None,
                    source_platform="kiro",
                    source_file=str(q_client_path.relative_to(REPO_ROOT)),
                    source_ref={
                        "record_index": rec_idx,
                        "conversation_id": conversation_id,
                        "history_turn_index": turn_index,
                        "source_log_type": "q-client-history",
                    },
                )
            )

    return turns


def _parse_kiro_user(user: str) -> dict[str, list[ChatTurn]]:
    by_phase: dict[str, list[ChatTurn]] = {phase: [] for phase in PHASES}

    user_dirs: list[Path] = []
    direct_user_dir = KIRO_DIR / user
    if direct_user_dir.exists() and direct_user_dir.is_dir():
        user_dirs.append(direct_user_dir)
    legacy_dirs = [p for p in KIRO_DIR.glob(f"{user}-phase*-kiro-logs") if p.is_dir()]
    user_dirs.extend(legacy_dirs)

    for user_dir in user_dirs:
        phase = "Phase 3"

        for run_dir in sorted([p for p in user_dir.iterdir() if p.is_dir()]):
            # Folder names vary by user dump, so search recursively inside each run folder.
            q_chat_candidates = list(run_dir.rglob("5-Q Chat API.log"))
            if not q_chat_candidates:
                q_chat_candidates = list(run_dir.rglob("q-client.log"))
            if not q_chat_candidates:
                continue
            q_chat_path = q_chat_candidates[0]

            if q_chat_path.name == "q-client.log":
                by_phase[phase].extend(_parse_q_client_user(user, run_dir, q_chat_path, phase))
                continue

            kiro_log_candidates = list(run_dir.rglob("Kiro Logs.log"))
            kiro_log_path = kiro_log_candidates[0] if kiro_log_candidates else None
            trigger_ts = _parse_kiro_timestamps(kiro_log_path) if kiro_log_path else []

            run_base_dt: datetime | None = None
            try:
                run_base_dt = datetime.strptime(run_dir.name, "%Y%m%dT%H%M%S")
            except Exception:
                run_base_dt = None

            records = _extract_json_lines(q_chat_path)
            pending_prompts: list[dict[str, Any]] = []
            response_turn_index = 0

            for rec_idx, rec in enumerate(records):
                if "request" in rec:
                    request_obj = rec.get("request", {})
                    if not isinstance(request_obj, dict):
                        continue
                    state = request_obj.get("conversationState", {})
                    if not isinstance(state, dict):
                        continue
                    current = state.get("currentMessage", {})
                    if not isinstance(current, dict):
                        continue
                    uim = current.get("userInputMessage", {})
                    if not isinstance(uim, dict):
                        continue
                    
                    text = str(uim.get("content") or "")
                    
                    ctx = uim.get("userInputMessageContext", {})
                    tool_results = ctx.get("toolResults", []) if isinstance(ctx, dict) else []
                    
                    if not text.strip() and tool_results:
                        lines = ["[Tool Result Follow-up]"]
                        for tr in tool_results:
                            if isinstance(tr, dict):
                                content_list = tr.get("content", [])
                                for c in content_list:
                                    if isinstance(c, dict) and "text" in c:
                                        lines.append(c["text"])
                        text = "\n".join(lines)

                    if not text.strip():
                        continue
                    pending_prompts.append(
                        {
                            "text": text,
                            "model": uim.get("modelId"),
                            "conversation_id": state.get("conversationId"),
                            "request_record_index": rec_idx,
                        }
                    )

                if "response" in rec:
                    response_obj = rec.get("response", {})
                    if not isinstance(response_obj, dict):
                        continue

                    prompt = pending_prompts.pop(0) if pending_prompts else {
                        "text": "",
                        "model": None,
                        "conversation_id": None,
                        "request_record_index": None,
                    }

                    full_response = str(response_obj.get("fullResponse") or "")
                    if not full_response:
                        full_message = response_obj.get("fullMessage", {})
                        if isinstance(full_message, dict):
                            full_response = str(full_message.get("text") or "")

                    response_events = response_obj.get("events", [])
                    model = prompt.get("model")

                    context_usage_pct: float | None = None
                    metering_usage: float | None = None
                    metering_unit: str | None = None

                    if isinstance(response_events, list):
                        for ev in response_events:
                            if not isinstance(ev, dict):
                                continue

                            ae = ev.get("assistantResponseEvent")
                            if isinstance(ae, dict) and ae.get("modelId"):
                                model = ae.get("modelId")

                            cue = ev.get("contextUsageEvent")
                            if isinstance(cue, dict) and cue.get("contextUsagePercentage") is not None:
                                context_usage_pct = _safe_float(cue.get("contextUsagePercentage"))

                            me = ev.get("meteringEvent")
                            if isinstance(me, dict):
                                metering_usage = _safe_float(me.get("usage"))
                                if me.get("unit"):
                                    metering_unit = str(me.get("unit"))

                    tool_calls = _extract_tool_calls(response_obj, response_events if isinstance(response_events, list) else [])
                    assistant_code_lines, languages = _extract_code_metadata(full_response)
                    legacy_fswrite_lines = _count_fswrite_lines(full_response)
                    tool_code_lines, tool_breakdown, tool_paths = _tool_code_metrics(tool_calls)
                    tool_code_lines += legacy_fswrite_lines
                    tool_breakdown["legacy_fswrite_lines"] = legacy_fswrite_lines

                    total_ai_lines = tool_code_lines if tool_code_lines > 0 else assistant_code_lines
                    referenced_files = _basename_refs(tool_paths)
                    languages = _dedupe_preserve_order(languages)

                    actual_prompt_tokens = None
                    actual_completion_tokens = None
                    estimated_prompt_tokens = _estimate_tokens(str(prompt.get("text") or ""))
                    tool_payload_text = "\n".join(
                        json.dumps(tc.get("args") or {}, ensure_ascii=False)
                        for tc in tool_calls
                        if isinstance(tc, dict)
                    )
                    estimated_completion_tokens = _estimate_tokens(
                        "\n".join(part for part in [full_response, tool_payload_text] if part)
                    )

                    if response_turn_index < len(trigger_ts):
                        ts = trigger_ts[response_turn_index]
                    elif run_base_dt is not None:
                        ts = (run_base_dt + timedelta(seconds=response_turn_index)).isoformat()
                    else:
                        ts = None
                    response_turn_index += 1

                    by_phase[phase].append(
                        ChatTurn(
                            timestamp=ts,
                            session_id=run_dir.name,
                            phase=phase,
                            user_text=str(prompt.get("text") or ""),
                            assistant_text=full_response,
                            model=str(model) if model is not None else None,
                            prompt_tokens=estimated_prompt_tokens,
                            completion_tokens=estimated_completion_tokens,
                            actual_prompt_tokens=actual_prompt_tokens,
                            actual_completion_tokens=actual_completion_tokens,
                            estimated_prompt_tokens=estimated_prompt_tokens,
                            estimated_completion_tokens=estimated_completion_tokens,
                            token_count_method="estimated_char_ratio_4",
                            code_lines_suggested=total_ai_lines,
                            assistant_code_lines=assistant_code_lines,
                            tool_code_lines=tool_code_lines,
                            code_line_breakdown={
                                "assistant_code_lines": assistant_code_lines,
                                "tool_code_lines": tool_code_lines,
                                **tool_breakdown,
                            },
                            latency_ms=None,
                            ttft_ms=None,
                            referenced_files=referenced_files,
                            languages=languages,
                            edited_file_events=None,
                            checkpoints_restored=None,
                            tool_calls=tool_calls,
                            metering_usage=metering_usage,
                            metering_unit=metering_unit,
                            context_usage_pct=context_usage_pct,
                            source_platform="kiro",
                            source_file=str(q_chat_path.relative_to(REPO_ROOT)),
                            source_ref={
                                "record_index": rec_idx,
                                "request_record_index": prompt.get("request_record_index"),
                                "conversation_id": prompt.get("conversation_id"),
                            },
                        )
                    )

    return by_phase


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def _is_tool_followup_prompt(text: str | None) -> bool:
    return bool((text or "").strip().startswith(TOOL_FOLLOWUP_PREFIX))


def _is_redacted_text(text: str | None) -> bool:
    return (text or "").strip() == REDACTED_SOURCE_PLACEHOLDER


def _has_visible_text(text: str | None) -> bool:
    stripped = (text or "").strip()
    return bool(stripped) and stripped != REDACTED_SOURCE_PLACEHOLDER


def _is_natural_prompt_turn(turn: ChatTurn) -> bool:
    text = (turn.user_text or "").strip()
    return bool(text) and not _is_tool_followup_prompt(text)


def _looks_like_retry_prompt(text: str | None, short_word_count: int = 25) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered or lowered == REDACTED_SOURCE_PLACEHOLDER.lower():
        return False
    word_count = len(lowered.split())
    return word_count <= short_word_count and any(keyword in lowered for keyword in NEGATIVE_KEYWORDS)


def _segment_has_measured_activity(segment: list[ChatTurn]) -> bool:
    for turn in segment:
        if _safe_int(turn.code_lines_suggested) > 0:
            return True
        if _safe_int(turn.assistant_code_lines) > 0 or _safe_int(turn.tool_code_lines) > 0:
            return True
        if isinstance(turn.tool_calls, list) and len(turn.tool_calls) > 0:
            return True
        if _has_visible_text(turn.assistant_text):
            return True
        if turn.metering_usage is not None and turn.metering_usage > 0:
            return True
        if _safe_int(turn.actual_prompt_tokens) > 0 or _safe_int(turn.actual_completion_tokens) > 0:
            return True
        if _safe_int(turn.estimated_prompt_tokens) > 0 or _safe_int(turn.estimated_completion_tokens) > 0:
            return True
    return False


def _prompt_segments(turns: list[ChatTurn]) -> list[list[ChatTurn]]:
    if not turns:
        return []

    sorted_turns = sorted(
        turns,
        key=lambda t: (_parse_ts(t.timestamp) or datetime.min),
    )
    natural_indices = [idx for idx, turn in enumerate(sorted_turns) if _is_natural_prompt_turn(turn)]
    segments: list[list[ChatTurn]] = []

    for pos, start_idx in enumerate(natural_indices):
        end_idx = natural_indices[pos + 1] if pos + 1 < len(natural_indices) else len(sorted_turns)
        segment = sorted_turns[start_idx:end_idx]
        if segment:
            segments.append(segment)

    return segments


def _compute_man_days(turns: list[ChatTurn]) -> int:
    days = set()
    for t in turns:
        dt = _parse_ts(t.timestamp)
        if dt:
            days.add(dt.date().isoformat())
    return len(days)


def _compute_man_hours(turns: list[ChatTurn], inactivity_minutes: int = 30) -> float:
    dts = sorted(dt for dt in (_parse_ts(t.timestamp) for t in turns) if dt is not None)
    if len(dts) < 2:
        return 0.0

    threshold = timedelta(minutes=inactivity_minutes)
    total = timedelta(0)
    session_start = dts[0]
    last = dts[0]

    for curr in dts[1:]:
        if curr - last > threshold:
            total += (last - session_start)
            session_start = curr
        last = curr

    total += (last - session_start)
    return round(total.total_seconds() / 3600.0, 3)


def _success_stats(turns: list[ChatTurn]) -> tuple[int, int, float, int]:
    total_code_responses = 0
    flagged_reverts = 0
    ai_lines_needing_revision = 0

    sorted_turns = sorted(
        turns,
        key=lambda t: (_parse_ts(t.timestamp) or datetime.min),
    )

    for i in range(len(sorted_turns) - 1):
        curr = sorted_turns[i]
        nxt = sorted_turns[i + 1]

        curr_code = _safe_int(curr.code_lines_suggested)
        if curr_code <= 0:
            continue

        total_code_responses += 1
        next_msg = (nxt.user_text or "").lower()
        word_count = len(next_msg.split())

        if word_count < 20 and any(k in next_msg for k in NEGATIVE_KEYWORDS):
            flagged_reverts += 1
            ai_lines_needing_revision += curr_code

    success_ratio = (
        (total_code_responses - flagged_reverts) / total_code_responses
        if total_code_responses > 0
        else 0.0
    )
    return total_code_responses, flagged_reverts, round(success_ratio, 4), ai_lines_needing_revision


def _prompt_activity_stats(turns: list[ChatTurn]) -> dict[str, Any]:
    if not turns:
        return {
            "total_prompts": 0,
            "total_visible_prompts": 0,
            "total_redacted_prompts": 0,
            "total_tool_followup_turns": 0,
            "retry_prompt_count": 0,
            "completed_features": 0,
            "prompt_success_rate": None,
            "prompt_to_feature_ratio": None,
            "prompt_visibility_rate": None,
            "prompt_metric_method": "natural_user_prompts_excludes_tool_followups",
            "feature_metric_method": "visible_prompt_segments_with_measured_activity",
        }

    sorted_turns = sorted(
        turns,
        key=lambda t: (_parse_ts(t.timestamp) or datetime.min),
    )
    tool_followup_turns = sum(1 for turn in sorted_turns if _is_tool_followup_prompt(turn.user_text))
    segments = _prompt_segments(sorted_turns)

    total_prompts = len(segments)
    total_visible_prompts = 0
    total_redacted_prompts = 0
    retry_prompt_count = 0
    completed_features = 0

    for segment in segments:
        prompt_turn = segment[0]
        prompt_text = prompt_turn.user_text or ""

        if _is_redacted_text(prompt_text):
            total_redacted_prompts += 1
        else:
            total_visible_prompts += 1

        is_retry = _looks_like_retry_prompt(prompt_text)
        if is_retry:
            retry_prompt_count += 1

        if _has_visible_text(prompt_text) and not is_retry and _segment_has_measured_activity(segment):
            completed_features += 1

    prompt_success_rate = (
        (total_visible_prompts - retry_prompt_count) / total_visible_prompts
        if total_visible_prompts > 0
        else None
    )
    prompt_to_feature_ratio = (
        total_visible_prompts / completed_features
        if completed_features > 0
        else None
    )
    prompt_visibility_rate = (
        total_visible_prompts / total_prompts
        if total_prompts > 0
        else None
    )

    return {
        "total_prompts": total_prompts,
        "total_visible_prompts": total_visible_prompts,
        "total_redacted_prompts": total_redacted_prompts,
        "total_tool_followup_turns": tool_followup_turns,
        "retry_prompt_count": retry_prompt_count,
        "completed_features": completed_features,
        "prompt_success_rate": round(prompt_success_rate, 4) if prompt_success_rate is not None else None,
        "prompt_to_feature_ratio": round(prompt_to_feature_ratio, 4) if prompt_to_feature_ratio is not None else None,
        "prompt_visibility_rate": round(prompt_visibility_rate, 4) if prompt_visibility_rate is not None else None,
        "prompt_metric_method": "natural_user_prompts_excludes_tool_followups",
        "feature_metric_method": "visible_prompt_segments_with_measured_activity",
    }


def _turn_to_dict(t: ChatTurn, user_id: str) -> dict[str, Any]:
    return {
        "timestamp": t.timestamp,
        "session_id": t.session_id,
        "phase": t.phase,
        "suspected_user": user_id,
        "user_text": t.user_text,
        "assistant_text": t.assistant_text,
        "model": t.model,
        "prompt_tokens": t.prompt_tokens,
        "completion_tokens": t.completion_tokens,
        "actual_prompt_tokens": t.actual_prompt_tokens,
        "actual_completion_tokens": t.actual_completion_tokens,
        "estimated_prompt_tokens": t.estimated_prompt_tokens,
        "estimated_completion_tokens": t.estimated_completion_tokens,
        "token_count_method": t.token_count_method,
        "code_lines_suggested": t.code_lines_suggested,
        "assistant_code_lines": t.assistant_code_lines,
        "tool_code_lines": t.tool_code_lines,
        "code_line_breakdown": t.code_line_breakdown,
        "latency_ms": t.latency_ms,
        "ttft_ms": t.ttft_ms,
        "referenced_files": t.referenced_files,
        "languages": t.languages,
        "edited_file_events": t.edited_file_events,
        "checkpoints_restored": t.checkpoints_restored,
        "tool_calls": t.tool_calls,
        "metering_usage": t.metering_usage,
        "metering_unit": t.metering_unit,
        "context_usage_pct": t.context_usage_pct,
        "source_platform": t.source_platform,
        "source_file": t.source_file,
        "source_ref": t.source_ref,
        "copilot_applied_edit_lines": t.copilot_applied_edit_lines,
        "copilot_applied_edit_calls": t.copilot_applied_edit_calls,
        "copilot_text_edit_groups": t.copilot_text_edit_groups,
        "copilot_distinct_edited_files": t.copilot_distinct_edited_files,
        "copilot_edit_tool_counts": t.copilot_edit_tool_counts,
        "copilot_edited_files": t.copilot_edited_files,
    }


def _build_user_phase_file(user: str, phase: str, turns: list[ChatTurn]) -> dict[str, Any]:
    total_code, flagged_reverts, _, ai_lines_needing_revision = _success_stats(turns)
    prompt_stats = _prompt_activity_stats(turns)
    token_method_counts = defaultdict(int)
    tool_name_counts = defaultdict(int)
    copilot_edit_tool_counts = defaultdict(int)
    copilot_edited_files: set[str] = set()

    for turn in turns:
        token_method = turn.token_count_method or "unknown"
        token_method_counts[token_method] += 1
        for tool_call in turn.tool_calls:
            if not isinstance(tool_call, dict):
                continue
            name = str(tool_call.get("name") or "unknown")
            tool_name_counts[name] += 1
        for tool_name, count in (turn.copilot_edit_tool_counts or {}).items():
            try:
                copilot_edit_tool_counts[str(tool_name)] += int(count)
            except Exception:
                continue
        for edited_file in turn.copilot_edited_files or []:
            if edited_file:
                copilot_edited_files.add(str(edited_file))

    data = {
        "schema_version": "1.1.0",
        "format": "universal_chat_sync",
        "user_id": user,
        "phase": phase,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_preservation": {
            "is_destructive": False,
            "notes": "Original source files remain unchanged; this file is normalized and source-linked.",
        },
        "chat_turns": [_turn_to_dict(t, user) for t in turns],
        "metrics": {
            "man_days": _compute_man_days(turns),
            "man_hours": _compute_man_hours(turns),
            "prompt_success_ratio": prompt_stats["prompt_success_rate"],
            "prompt_success_rate": prompt_stats["prompt_success_rate"],
            "prompt_to_feature_ratio": prompt_stats["prompt_to_feature_ratio"],
            "total_prompts": prompt_stats["total_prompts"],
            "total_visible_prompts": prompt_stats["total_visible_prompts"],
            "total_redacted_prompts": prompt_stats["total_redacted_prompts"],
            "total_tool_followup_turns": prompt_stats["total_tool_followup_turns"],
            "retry_prompt_count": prompt_stats["retry_prompt_count"],
            "prompt_visibility_rate": prompt_stats["prompt_visibility_rate"],
            "completed_features": prompt_stats["completed_features"],
            "prompt_metric_method": prompt_stats["prompt_metric_method"],
            "feature_metric_method": prompt_stats["feature_metric_method"],
            "total_lines_written_by_selected_model_agent": sum(_safe_int(t.code_lines_suggested) for t in turns),
            "total_lines_written_by_humans": None,
            "number_of_ai_lines_needing_human_revision": ai_lines_needing_revision,
            "total_copilot_applied_edit_lines": sum(_safe_int(t.copilot_applied_edit_lines) for t in turns),
            "total_copilot_applied_edit_calls": sum(_safe_int(t.copilot_applied_edit_calls) for t in turns),
            "total_copilot_text_edit_groups": sum(_safe_int(t.copilot_text_edit_groups) for t in turns),
            "total_copilot_edited_file_events": sum(
                _safe_int(t.edited_file_events) for t in turns if t.source_platform == "copilot"
            ),
            "total_copilot_distinct_edited_files": len(copilot_edited_files),
            "total_prompt_tokens": sum(_safe_int(t.prompt_tokens) for t in turns),
            "total_completion_tokens": sum(_safe_int(t.completion_tokens) for t in turns),
            "total_actual_prompt_tokens": sum(_safe_int(t.actual_prompt_tokens) for t in turns),
            "total_actual_completion_tokens": sum(_safe_int(t.actual_completion_tokens) for t in turns),
            "total_estimated_prompt_tokens": sum(_safe_int(t.estimated_prompt_tokens) for t in turns),
            "total_estimated_completion_tokens": sum(_safe_int(t.estimated_completion_tokens) for t in turns),
            "total_assistant_code_lines": sum(_safe_int(t.assistant_code_lines) for t in turns),
            "total_assistant_fallback_code_lines": sum(
                _safe_int(t.assistant_code_lines) for t in turns if _safe_int(t.tool_code_lines) <= 0
            ),
            "total_tool_code_lines": sum(_safe_int(t.tool_code_lines) for t in turns),
            "total_metering_usage": round(
                sum(t.metering_usage or 0.0 for t in turns),
                6,
            ),
            "metering_unit": next((t.metering_unit for t in turns if t.metering_unit), None),
            "average_context_usage_percentage": round(
                (
                    sum(t.context_usage_pct or 0.0 for t in turns if t.context_usage_pct is not None)
                    / max(1, len([t for t in turns if t.context_usage_pct is not None]))
                ),
                4,
            )
            if any(t.context_usage_pct is not None for t in turns)
            else None,
            "token_count_methods": dict(sorted(token_method_counts.items())),
            "tool_call_counts": dict(sorted(tool_name_counts.items())),
            "copilot_edit_tool_counts": dict(sorted(copilot_edit_tool_counts.items())),
            "total_code_responses": total_code,
            "flagged_reverts": flagged_reverts,
            "measured_ai_output_definition": (
                "Structured applied editor/tool edit lines when preserved by the source logs; "
                "otherwise fenced assistant code lines."
            ),
            "heuristic": {
                "negative_keywords": NEGATIVE_KEYWORDS,
                "next_prompt_short_word_count_lt": 20,
            },
        },
        "nullability_contract": {
            "rule": "Fields missing in a source are preserved as null in normalized turns.",
            "common_null_fields_for_kiro": [
                "actual_prompt_tokens",
                "actual_completion_tokens",
                "latency_ms",
                "ttft_ms",
                "edited_file_events",
                "checkpoints_restored",
            ],
        },
    }

    return data


def build_universal_files() -> list[Path]:
    users = _discover_users()
    written: list[Path] = []

    for user in sorted(users):
        combined: dict[str, list[ChatTurn]] = {phase: [] for phase in PHASES}

        copilot_phase_turns = _parse_copilot_user(user)
        for phase in PHASES:
            combined[phase].extend(copilot_phase_turns.get(phase, []))

        kiro_phase_turns = _parse_kiro_user(user)
        for phase in PHASES:
            combined[phase].extend(kiro_phase_turns.get(phase, []))

        user_dir = UNIVERSAL_DIR / user
        user_dir.mkdir(parents=True, exist_ok=True)

        for phase in PHASES:
            turns = combined.get(phase, [])
            payload = _build_user_phase_file(user, phase, turns)
            out_path = user_dir / f"{phase.lower().replace(' ', '_')}_universal.json"
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            written.append(out_path)

    return written


def main() -> None:
    written = build_universal_files()
    print(f"Wrote {len(written)} universal files")
    for path in written:
        print(path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
