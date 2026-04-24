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
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
SYNTHESIS_DIR = REPO_ROOT / "synthesis"
COPILOT_DIR = SYNTHESIS_DIR / "copilot_data"
KIRO_DIR = SYNTHESIS_DIR / "kiro_data"
UNIVERSAL_DIR = SYNTHESIS_DIR / "universal"

PHASES = ["Phase 1", "Phase 2", "Phase 3"]
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
    code_lines_suggested: int | None
    latency_ms: int | None
    ttft_ms: int | None
    referenced_files: list[str]
    languages: list[str]
    edited_file_events: int | None
    checkpoints_restored: int | None
    source_platform: str
    source_file: str
    source_ref: dict[str, Any]


def _safe_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(value)
    except Exception:
        return 0


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
                users.add(child.name.lower())

    if KIRO_DIR.exists():
        for child in KIRO_DIR.iterdir():
            if child.is_dir():
                # Example: jason-phase3-kiro-logs
                m = re.match(r"([a-zA-Z0-9_-]+)-phase\d+", child.name)
                if m:
                    users.add(m.group(1).lower())

    return users


def _parse_copilot_user(user: str) -> dict[str, list[ChatTurn]]:
    by_phase: dict[str, list[ChatTurn]] = {phase: [] for phase in PHASES}
    user_dir = COPILOT_DIR / user.capitalize()
    if not user_dir.exists():
        # try lowercase fallback
        user_dir = COPILOT_DIR / user
    if not user_dir.exists():
        return by_phase

    for file_path in sorted(user_dir.glob("*.json")):
        phase = _phase_from_name(file_path.name)
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

            model = result.get("details") or req.get("modelId") or "Unknown"
            code_lines, languages = _extract_code_metadata(assistant_text)

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
                    code_lines_suggested=code_lines,
                    latency_ms=_safe_int(timings.get("totalElapsed")) if timings else None,
                    ttft_ms=_safe_int(timings.get("firstProgress")) if timings else None,
                    referenced_files=referenced_files,
                    languages=languages,
                    edited_file_events=len(edited_file_events),
                    checkpoints_restored=checkpoints_restored,
                    source_platform="copilot",
                    source_file=str(file_path.relative_to(REPO_ROOT)),
                    source_ref={"request_index": idx},
                )
            )

    return by_phase


def _extract_json_lines(path: Path) -> list[dict[str, Any]]:
    objs: list[dict[str, Any]] = []
    if not path.exists():
        return objs
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                objs.append(parsed)
        except Exception:
            continue
    return objs


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


def _parse_kiro_user(user: str) -> dict[str, list[ChatTurn]]:
    by_phase: dict[str, list[ChatTurn]] = {phase: [] for phase in PHASES}

    user_dirs = [
        p for p in KIRO_DIR.glob(f"{user}-phase*-kiro-logs") if p.is_dir()
    ]
    for user_dir in user_dirs:
        phase = _phase_from_name(user_dir.name)

        for run_dir in sorted([p for p in user_dir.iterdir() if p.is_dir()]):
            # Folder names vary. Try glob first.
            q_chat_candidates = list((run_dir / "window1" / "exthost").glob("output_logging_*/5-Q Chat API.log"))
            if not q_chat_candidates:
                continue
            q_chat_path = q_chat_candidates[0]

            kiro_log_candidates = list((run_dir / "window1" / "exthost" / "kiro.kiroAgent").glob("Kiro Logs.log"))
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
                    
                    tool_acc = defaultdict(str)
                    
                    if isinstance(response_events, list):
                        for ev in response_events:
                            if not isinstance(ev, dict):
                                continue
                            
                            ae = ev.get("assistantResponseEvent")
                            if isinstance(ae, dict) and ae.get("modelId"):
                                model = ae.get("modelId")
                            
                            tue = ev.get("toolUseEvent")
                            if isinstance(tue, dict):
                                tid = tue.get("toolUseId")
                                chunk = tue.get("input")
                                if tid and isinstance(chunk, str):
                                    tool_acc[tid] += chunk

                    code_lines, languages = _extract_code_metadata(full_response)
                    fswrite_lines = _count_fswrite_lines(full_response)
                    
                    for tid, inputs_json in tool_acc.items():
                        full_response += f"\n\n[Tool Input {tid}]: {inputs_json}"
                        try:
                            tinput = json.loads(inputs_json)
                            if isinstance(tinput, dict):
                                if "content" in tinput and isinstance(tinput["content"], str):
                                    fswrite_lines += len(tinput["content"].splitlines())
                                elif "new_str" in tinput and isinstance(tinput["new_str"], str):
                                    fswrite_lines += len(tinput["new_str"].splitlines())
                        except Exception:
                            pass
                        
                        cl, langs = _extract_code_metadata(inputs_json)
                        code_lines += cl
                        languages.extend(langs)

                    total_ai_lines = code_lines + fswrite_lines
                    languages = list(set(languages))

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
                            prompt_tokens=None,
                            completion_tokens=None,
                            code_lines_suggested=total_ai_lines,
                            latency_ms=None,
                            ttft_ms=None,
                            referenced_files=[],
                            languages=languages,
                            edited_file_events=None,
                            checkpoints_restored=None,
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


def _feature_stats(turns: list[ChatTurn]) -> tuple[int, int, float]:
    if not turns:
        return 0, 0, 0.0

    sorted_turns = sorted(
        turns,
        key=lambda t: (_parse_ts(t.timestamp) or datetime.min),
    )

    prompts = len([t for t in sorted_turns if (t.user_text or "").strip()])
    batch_size = 3
    completed = 0

    for i in range(0, len(sorted_turns), batch_size):
        batch = sorted_turns[i : i + batch_size]
        code_sum = sum(_safe_int(t.code_lines_suggested) for t in batch)
        if not batch:
            continue
        last_prompt = (batch[-1].user_text or "").lower()
        is_retry = any(k in last_prompt for k in NEGATIVE_KEYWORDS)
        if code_sum > 0 and not is_retry:
            completed += 1

    ratio = (prompts / completed) if completed > 0 else 0.0
    return prompts, completed, round(ratio, 4)


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
        "code_lines_suggested": t.code_lines_suggested,
        "latency_ms": t.latency_ms,
        "ttft_ms": t.ttft_ms,
        "referenced_files": t.referenced_files,
        "languages": t.languages,
        "edited_file_events": t.edited_file_events,
        "checkpoints_restored": t.checkpoints_restored,
        "source_platform": t.source_platform,
        "source_file": t.source_file,
        "source_ref": t.source_ref,
    }


def _build_user_phase_file(user: str, phase: str, turns: list[ChatTurn]) -> dict[str, Any]:
    total_code, flagged_reverts, success_ratio, ai_lines_needing_revision = _success_stats(turns)
    prompts, completed_features, prompt_to_feature_ratio = _feature_stats(turns)

    data = {
        "schema_version": "1.0.0",
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
            "prompt_success_ratio": success_ratio,
            "prompt_success_rate": success_ratio,
            "prompt_to_feature_ratio": prompt_to_feature_ratio,
            "total_prompts": prompts,
            "completed_features": completed_features,
            "total_lines_written_by_selected_model_agent": sum(_safe_int(t.code_lines_suggested) for t in turns),
            "total_lines_written_by_humans": None,
            "number_of_ai_lines_needing_human_revision": ai_lines_needing_revision,
            "total_code_responses": total_code,
            "flagged_reverts": flagged_reverts,
            "heuristic": {
                "negative_keywords": NEGATIVE_KEYWORDS,
                "next_prompt_short_word_count_lt": 20,
            },
        },
        "nullability_contract": {
            "rule": "Fields missing in a source are preserved as null in normalized turns.",
            "common_null_fields_for_kiro": [
                "prompt_tokens",
                "completion_tokens",
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
