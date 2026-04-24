"""Streamlit entrypoint for the Copilot History Analyzer app.

This module wires together:
1. Sidebar configuration and filters.
2. Data loading/parsing orchestration.
3. Delegation to tab renderer functions.
"""

import os
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from services.auth import require_password
from services.data_processing import (
    DATA_DIR,
    discover_available_phases,
    gather_phase_files,
    parse_chat_data,
    parse_git_history,
)
from views.tabs import (
    render_chat_history_tab,
    render_comparative_tab,
    render_prompt_analysis_tab,
    render_statistics_tab,
    render_timeline_tab,
    render_universal_required_stats_tab,
)
from views.flow_visualization import render_flow_visualization_tab


st.set_page_config(page_title="Copilot History Analyzer", layout="wide")
# require_password()

st.title("Copilot History Analyzer")


def build_chat_session_sidebar(
    df_chat_all: pd.DataFrame,
    section_title: str = "Chat History View",
    radio_label: str = "Select Chat Session to View",
    radio_key: str | None = None,
) -> str | None:
    """Render chat session selection controls.

    Args:
        df_chat_all: Full chat dataframe containing at least file_name,
            suspected_user, and timestamp columns.

    Returns:
        The selected session file name, or None if no sessions are available.

    Notes:
        Display labels include session start time and inferred user for context,
        but the returned value is the raw file name used for filtering.
    """
    st.sidebar.divider()
    st.sidebar.subheader(section_title)

    session_map: dict[str, str] = {}
    unique_sessions = df_chat_all[["file_name", "suspected_user", "timestamp"]].drop_duplicates(
        subset=["file_name"], keep="first"
    )

    display_options = []
    for _, row in unique_sessions.iterrows():
        start_time_str = row["timestamp"].strftime("%Y-%m-%d %H:%M")
        label = f"{start_time_str} | {row['file_name']} ({row['suspected_user']})"
        display_options.append(label)
        session_map[label] = row["file_name"]

    selected_display_label = st.sidebar.radio(
        radio_label,
        options=display_options,
        index=0 if display_options else None,
        key=radio_key,
    )
    return session_map.get(selected_display_label)


def discover_universal_files(base_dir: str = "synthesis/universal") -> list[str]:
    """Discover generated universal JSON files.

    Args:
        base_dir: Universal output root.

    Returns:
        Sorted list of universal JSON file paths.
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return []

    files = [str(p) for p in base_path.rglob("*_universal.json")]
    return sorted(files)


def load_universal_metrics(file_paths: list[str]) -> pd.DataFrame:
    """Load per-file universal metrics into a dataframe.

    Args:
        file_paths: Universal JSON file paths.

    Returns:
        Dataframe with one row per file and required KPI columns.
    """
    rows: list[dict] = []

    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        metrics = data.get("metrics", {}) if isinstance(data, dict) else {}
        if not isinstance(metrics, dict):
            metrics = {}

        rows.append(
            {
                "file_path": file_path,
                "user_id": data.get("user_id"),
                "phase": data.get("phase"),
                "man_hours": metrics.get("man_hours"),
                "man_days": metrics.get("man_days"),
                "prompt_success_ratio": metrics.get("prompt_success_ratio"),
                "prompt_success_rate": metrics.get("prompt_success_rate"),
                "prompt_to_feature_ratio": metrics.get("prompt_to_feature_ratio"),
                "total_lines_written_by_humans": metrics.get("total_lines_written_by_humans"),
                "total_lines_written_by_selected_model_agent": metrics.get("total_lines_written_by_selected_model_agent"),
                "number_of_ai_lines_needing_human_revision": metrics.get("number_of_ai_lines_needing_human_revision"),
            }
        )

    return pd.DataFrame(rows)


def build_analysis_filters_sidebar(df_chat_all: pd.DataFrame) -> pd.DataFrame:
    """Render analysis filters and return filtered chat rows.

    Args:
        df_chat_all: Full chat dataframe containing session-level rows.

    Returns:
        A subset of df_chat_all restricted to sessions selected in the sidebar.
    """
    st.sidebar.divider()
    st.sidebar.subheader("Analysis Filters")

    all_sessions = df_chat_all["file_name"].unique().tolist()

    with st.sidebar.expander("Select Sessions for Analysis", expanded=True):
        selected_sessions_analysis = st.multiselect(
            "Filter statistics by session:",
            options=all_sessions,
            default=all_sessions,
        )

    return df_chat_all[df_chat_all["file_name"].isin(selected_sessions_analysis)]


def load_git_data(repo_path: str) -> pd.DataFrame:
    """Load git history for a repository path.

    Args:
        repo_path: User-provided path to a local git repository root.

    Returns:
        A timestamp-sorted dataframe of commits when the path is valid,
        otherwise an empty dataframe.
    """
    if repo_path and os.path.isdir(repo_path):
        df_git = parse_git_history(repo_path)
        if not df_git.empty:
            return df_git.sort_values("timestamp")
    return pd.DataFrame()


def main() -> None:
    """Run the Streamlit app lifecycle.

    This function coordinates:
    1. Sidebar inputs.
    2. Data loading/parsing.
    3. Session filtering.
    4. Tab rendering delegation.
    """
    st.sidebar.header("Configuration")

    available_phases = discover_available_phases(DATA_DIR)
    selected_phases = st.sidebar.multiselect(
        "Select Phases to Pool Data From",
        options=available_phases,
        default=available_phases,
    )

    uploaded_files = st.sidebar.file_uploader(
        "Or Upload chatTemplate.json manually",
        type="json",
        accept_multiple_files=True,
    )

    st.sidebar.divider()
    st.sidebar.subheader("Universal Files")
    universal_files = discover_universal_files()
    selected_universal_files = st.sidebar.multiselect(
        "Select universal files",
        options=universal_files,
        default=universal_files,
        help="Generated files in synthesis/universal/*/*_universal.json",
    )

    repo_path = st.sidebar.text_input(
        "Local Git Repository Path (Optional)",
        help="Path to the root of your local git repository to correlate chat with commits.",
    )

    # Merge uploaded files with discovered phase files into a single parse list.
    files_to_parse = list(uploaded_files) if uploaded_files else []
    files_to_parse.extend(gather_phase_files(selected_phases, DATA_DIR))

    if not files_to_parse and not selected_universal_files:
        st.info("Please select/upload files or choose one or more universal files to begin.")
        return

    df_chat_all = parse_chat_data(files_to_parse) if files_to_parse else pd.DataFrame()
    if not df_chat_all.empty:
        df_chat_all = df_chat_all.sort_values("timestamp")

    selected_chat_file = None
    df_chat_analysis = pd.DataFrame()
    if not df_chat_all.empty:
        selected_chat_file = build_chat_session_sidebar(
            df_chat_all,
            section_title="Chat History View",
            radio_label="Select Chat Session to View",
            radio_key="legacy_chat_session",
        )
        df_chat_analysis = build_analysis_filters_sidebar(df_chat_all)

    df_universal_all = parse_chat_data(selected_universal_files) if selected_universal_files else pd.DataFrame()
    if not df_universal_all.empty:
        df_universal_all = df_universal_all.sort_values("timestamp")

    selected_universal_chat_file = None
    if not df_universal_all.empty:
        selected_universal_chat_file = build_chat_session_sidebar(
            df_universal_all,
            section_title="Universal Chat History View",
            radio_label="Select Universal Session to View",
            radio_key="universal_chat_session",
        )

    df_universal_metrics = load_universal_metrics(selected_universal_files)
    df_git = load_git_data(repo_path)

    # Each tab delegates rendering to a focused function in views.tabs.
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
        [
            "Chat History",
            "Statistics",
            "Development Timeline",
            "Comparative Analysis",
            "Prompt Analysis",
            "Prompt-to-Feature Flow",
            "Universal Chat History",
            "Universal Required Stats",
        ]
    )

    with tab1:
        if df_chat_all.empty:
            st.info("No legacy chat data loaded from selected/uploaded files.")
        else:
            render_chat_history_tab(df_chat_all, selected_chat_file)

    with tab2:
        if df_chat_analysis.empty:
            st.info("No legacy analysis data loaded.")
        else:
            render_statistics_tab(df_chat_analysis, df_git)

    with tab3:
        if df_chat_analysis.empty:
            st.info("No legacy analysis data loaded.")
        else:
            render_timeline_tab(df_chat_analysis, df_git)

    with tab4:
        if df_chat_analysis.empty:
            st.info("No legacy analysis data loaded.")
        else:
            render_comparative_tab(df_chat_analysis)

    with tab5:
        if df_chat_analysis.empty:
            st.info("No legacy analysis data loaded.")
        else:
            render_prompt_analysis_tab(df_chat_analysis)

    with tab6:
        if df_chat_analysis.empty:
            st.info("No legacy analysis data loaded.")
        else:
            render_flow_visualization_tab(df_chat_analysis)

    with tab7:
        if df_universal_all.empty:
            st.info("No universal chat data loaded. Select files in the sidebar.")
        else:
            render_chat_history_tab(df_universal_all, selected_universal_chat_file)

    with tab8:
        render_universal_required_stats_tab(df_universal_metrics)


if __name__ == "__main__":
    main()
