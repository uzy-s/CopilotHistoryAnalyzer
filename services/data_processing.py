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
    if not os.path.exists(data_dir):
        return []
    return [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]


def gather_phase_files(selected_phases: Iterable[str], data_dir: str = DATA_DIR) -> list[str]:
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


@st.cache_data
def parse_chat_data(files: list) -> pd.DataFrame:
    all_requests = []

    for uploaded_file in files:
        try:
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
            file_ref = uploaded_file if isinstance(uploaded_file, str) else uploaded_file.name
            st.error(f"Error parsing {file_ref}: {e}")

    return pd.DataFrame(all_requests)


def parse_git_history(path: str) -> pd.DataFrame:
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
