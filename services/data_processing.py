"""Data loading and normalization utilities for chat and git sources.

The helpers in this module keep parsing behavior deterministic and defensive
against malformed or partial JSON records.
"""

import json
import os
import re
from datetime import datetime
from typing import Iterable

import pandas as pd
import streamlit as st
from git import Repo


DATA_DIR = "data"


def discover_available_phases(data_dir: str = DATA_DIR) -> list[str]:
    """Discover phase directories under the data root.

    Args:
        data_dir: Root directory containing phase folders.

    Returns:
        List of first-level folder names inside data_dir.
    """
    if not os.path.exists(data_dir):
        return []
    return [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]


def gather_phase_files(selected_phases: Iterable[str], data_dir: str = DATA_DIR) -> list[str]:
    """Collect JSON files for selected phases.

    Args:
        selected_phases: Iterable of phase folder names to scan.
        data_dir: Root data directory that contains phase folders.

    Returns:
        Flat list of absolute/relative file paths to JSON files.
    """
    files_to_parse: list[str] = []

    for selected_phase in selected_phases:
        phase_dir = os.path.join(data_dir, selected_phase)
        if not os.path.isdir(phase_dir):
            continue

        for root, _, files in os.walk(phase_dir):
            for file_name in files:
                if file_name.endswith(".json"):
                    files_to_parse.append(os.path.join(root, file_name))

    return files_to_parse


def _infer_phase_name(source_path: str) -> str:
    """Infer phase label from source path.

    Args:
        source_path: Path for an uploaded file or local file.

    Returns:
        Inferred phase name when path contains data/<phase>/..., else "Uploaded".
    """
    if not source_path or not isinstance(source_path, str):
        return "Uploaded"

    normalized = os.path.normpath(source_path)
    parts = normalized.split(os.sep)
    if "data" in parts:
        data_idx = parts.index("data")
        if data_idx + 1 < len(parts):
            return parts[data_idx + 1]

    return "Uploaded"


def _extract_suspected_user_from_requests(requests: list[dict]) -> str:
    """Infer username from request variable path metadata.

    Args:
        requests: Raw request records from exported chat JSON.

    Returns:
        Inferred username from path patterns, or "Unknown User" when unresolved.
    """
    suspected_user = "Unknown User"

    for req in requests:
        if not isinstance(req, dict):
            continue
        if suspected_user != "Unknown User":
            break

        variable_data = req.get("variableData", {})
        if not isinstance(variable_data, dict):
            variable_data = {}

        variables = variable_data.get("variables", [])
        if not isinstance(variables, list):
            variables = []

        for var in variables:
            if not isinstance(var, dict):
                continue
            val = var.get("value", {})
            if not isinstance(val, dict):
                continue

            path = val.get("fsPath") or val.get("path")
            if not path:
                continue

            match = re.search(r"[/\\](?:Users|home)[/\\]([^/\\]+)", path, re.IGNORECASE)
            if match:
                suspected_user = match.group(1)
                break

    return suspected_user


def _extract_referenced_files(req: dict) -> list[str]:
    """Extract basename-only file references from one request.

    Args:
        req: Single request dictionary.

    Returns:
        List of referenced file base names found in variable metadata.
    """
    referenced_files: list[str] = []

    variable_data = req.get("variableData", {})
    if not isinstance(variable_data, dict):
        variable_data = {}

    variables = variable_data.get("variables", [])
    if not isinstance(variables, list):
        variables = []

    for var in variables:
        if not isinstance(var, dict):
            continue
        val = var.get("value", {})
        if not isinstance(val, dict):
            continue

        path = val.get("fsPath") or val.get("path")
        if not path:
            continue

        try:
            referenced_files.append(os.path.basename(path))
        except Exception:
            # If basename extraction fails, we simply skip that entry.
            pass

    return referenced_files


def _extract_code_metadata(assistant_msg: str) -> tuple[int, list[str]]:
    """Extract code-line count and code fence languages.

    Args:
        assistant_msg: Raw assistant response text.

    Returns:
        Tuple of (code_line_count, language_list).
    """
    code_lines = 0
    languages: list[str] = []

    if "```" not in assistant_msg:
        return code_lines, languages

    lines = assistant_msg.split("\n")
    in_block = False

    for line in lines:
        if line.strip().startswith("```"):
            if not in_block:
                lang = line.strip().replace("```", "").strip()
                if lang:
                    languages.append(lang)
            in_block = not in_block
        elif in_block:
            code_lines += 1

    return code_lines, languages


def _parse_universal_chat_turns(data: dict, file_name: str, source_path: str) -> list[dict]:
    """Parse universal synchronized chat format into legacy app row shape.

    Args:
        data: Loaded universal JSON payload.
        file_name: Source file name being parsed.
        source_path: Source path on disk.

    Returns:
        Rows in the same normalized shape produced for Copilot exports.
    """
    rows: list[dict] = []
    chat_turns = data.get("chat_turns", [])
    if not isinstance(chat_turns, list):
        return rows

    default_user = str(data.get("user_id") or "Unknown User")
    default_phase = str(data.get("phase") or _infer_phase_name(source_path))

    for turn in chat_turns:
        if not isinstance(turn, dict):
            continue

        timestamp_raw = turn.get("timestamp")
        if not timestamp_raw:
            continue

        try:
            dt = datetime.fromisoformat(str(timestamp_raw).replace("Z", "+00:00"))
        except Exception:
            continue

        languages = turn.get("languages", [])
        if not isinstance(languages, list):
            languages = []

        referenced_files = turn.get("referenced_files", [])
        if not isinstance(referenced_files, list):
            referenced_files = []

        rows.append(
            {
                "timestamp": dt,
                "user_text": str(turn.get("user_text") or ""),
                "assistant_text": str(turn.get("assistant_text") or ""),
                "model": str(turn.get("model") or "Unknown"),
                "completion_tokens": int(turn.get("completion_tokens") or 0),
                "prompt_tokens": int(turn.get("prompt_tokens") or 0),
                "actual_completion_tokens": int(turn.get("actual_completion_tokens") or 0),
                "actual_prompt_tokens": int(turn.get("actual_prompt_tokens") or 0),
                "estimated_completion_tokens": int(turn.get("estimated_completion_tokens") or 0),
                "estimated_prompt_tokens": int(turn.get("estimated_prompt_tokens") or 0),
                "token_count_method": str(turn.get("token_count_method") or ""),
                "code_lines_suggested": int(turn.get("code_lines_suggested") or 0),
                "assistant_code_lines": int(turn.get("assistant_code_lines") or 0),
                "tool_code_lines": int(turn.get("tool_code_lines") or 0),
                "file_name": str(turn.get("session_id") or file_name),
                "suspected_user": str(turn.get("suspected_user") or default_user),
                "latency_ms": int(turn.get("latency_ms") or 0),
                "ttft_ms": int(turn.get("ttft_ms") or 0),
                "referenced_files": referenced_files,
                "languages": languages,
                "phase": str(turn.get("phase") or default_phase),
                "source_path": source_path,
                "source_platform": str(turn.get("source_platform") or ""),
                "edited_file_events": int(turn.get("edited_file_events") or 0),
                "checkpoints_restored": int(turn.get("checkpoints_restored") or 0),
                "metering_usage": float(turn.get("metering_usage") or 0.0),
                "metering_unit": str(turn.get("metering_unit") or ""),
                "context_usage_pct": float(turn.get("context_usage_pct") or 0.0),
                "copilot_applied_edit_lines": int(turn.get("copilot_applied_edit_lines") or 0),
                "copilot_applied_edit_calls": int(turn.get("copilot_applied_edit_calls") or 0),
                "copilot_text_edit_groups": int(turn.get("copilot_text_edit_groups") or 0),
                "copilot_distinct_edited_files": int(turn.get("copilot_distinct_edited_files") or 0),
            }
        )

    return rows


@st.cache_data
def parse_chat_data(files: list) -> pd.DataFrame:
    """Parse uploaded or on-disk chat JSON files into a normalized dataframe.

    Args:
        files: Mixed list of file paths and Streamlit UploadedFile objects.

    Returns:
        Normalized dataframe where each row is one request/response pair.

    Notes:
        Output rows include model, token usage, timing, language, and editor
        event metadata. Invalid files are reported and skipped.
    """
    all_requests = []

    for uploaded_file in files:
        try:
            # Support both local file paths and Streamlit UploadedFile objects.
            if isinstance(uploaded_file, str):
                with open(uploaded_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                file_name = os.path.basename(uploaded_file)
                source_path = uploaded_file
            else:
                data = json.load(uploaded_file)
                file_name = uploaded_file.name
                source_path = uploaded_file.name

            phase_name = _infer_phase_name(source_path)

            # Universal synchronized format support.
            if isinstance(data, dict) and isinstance(data.get("chat_turns"), list):
                all_requests.extend(_parse_universal_chat_turns(data, file_name, source_path))
                continue

            requests = data.get("requests", [])
            if not isinstance(requests, list):
                requests = []

            suspected_user = _extract_suspected_user_from_requests(requests)

            for req in requests:
                if not isinstance(req, dict):
                    continue

                timestamp = req.get("timestamp")
                if not timestamp:
                    continue

                dt = datetime.fromtimestamp(timestamp / 1000.0)

                message_obj = req.get("message", {})
                if not isinstance(message_obj, dict):
                    message_obj = {}
                user_msg = message_obj.get("text", "")

                response_parts = req.get("response", [])
                if not isinstance(response_parts, list):
                    response_parts = []

                assistant_msg = ""
                model_name = "Unknown"
                metrics = {}

                # Concatenate all assistant content, skipping explicit thinking blocks.
                for part in response_parts:
                    if not isinstance(part, dict):
                        continue
                    val = part.get("value")
                    if val and isinstance(val, str):
                        if part.get("kind") == "thinking":
                            continue
                        assistant_msg += val

                result = req.get("result", {})
                if not isinstance(result, dict):
                    result = {}

                timings = result.get("timings", {})
                if not isinstance(timings, dict):
                    timings = {}

                if result:
                    details = result.get("details", "")
                    if details:
                        model_name = details

                    usage = result.get("usage", {})
                    if not isinstance(usage, dict):
                        usage = {}
                    metrics = usage

                referenced_files = _extract_referenced_files(req)
                code_lines, languages = _extract_code_metadata(assistant_msg)

                edited_file_events = req.get("editedFileEvents", [])
                if not isinstance(edited_file_events, list):
                    edited_file_events = []

                all_requests.append(
                    {
                        "timestamp": dt,
                        "user_text": user_msg,
                        "assistant_text": assistant_msg,
                        "model": model_name,
                        "completion_tokens": metrics.get("completionTokens", 0),
                        "prompt_tokens": metrics.get("promptTokens", 0),
                        "code_lines_suggested": code_lines,
                        "file_name": file_name,
                        "suspected_user": suspected_user,
                        "latency_ms": timings.get("totalElapsed", 0),
                        "ttft_ms": timings.get("firstProgress", 0),
                        "referenced_files": referenced_files,
                        "languages": languages,
                        "phase": phase_name,
                        "source_path": source_path,
                        "edited_file_events": len(edited_file_events),
                        "checkpoints_restored": 1
                        if any(isinstance(p, dict) and p.get("kind") == "undoStop" for p in response_parts)
                        else 0,
                    }
                )

        except Exception as e:
            # Parsing should continue even if one file fails.
            file_ref = uploaded_file if isinstance(uploaded_file, str) else uploaded_file.name
            st.error(f"Error parsing {file_ref}: {e}")

    return pd.DataFrame(all_requests)


def parse_git_history(path: str) -> pd.DataFrame:
    """Read git commit history for correlation views.

    Args:
        path: Path to the repository root.

    Returns:
        Dataframe of commit metadata (timestamp, author, message, and stats).
        Returns an empty dataframe when parsing fails.
    """
    commits_data = []
    try:
        repo = Repo(path)
        for commit in repo.iter_commits():
            commits_data.append(
                {
                    "timestamp": datetime.fromtimestamp(commit.committed_date),
                    "author": commit.author.name,
                    "message": commit.message,
                    "insertions": commit.stats.total["insertions"],
                    "deletions": commit.stats.total["deletions"],
                    "files": commit.stats.files,
                }
            )
    except Exception as e:
        st.error(f"Error reading git repo: {e}")

    return pd.DataFrame(commits_data)
