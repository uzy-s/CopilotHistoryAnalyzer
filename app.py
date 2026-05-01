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
from services.ritm_data import discover_ritm_pdf, load_ritm_data
from services.tracking_metrics import discover_tracking_workbooks, load_tracking_metrics, merge_tracking_metrics
from views.tabs import (
    render_chat_history_tab,
    render_comparative_tab,
    render_statistics_tab,
    render_timeline_tab,
    render_universal_dashboard,  # Updated import
    render_unified_overview_tab,
    render_universal_key_insights_tab,
    render_universal_required_stats_tab,
)
from views.flow_visualization import render_flow_visualization_tab


st.set_page_config(page_title="Copilot History Analyzer", layout="wide")
# require_password()

st.title("Copilot History Analyzer")


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
                "total_copilot_applied_edit_lines": metrics.get("total_copilot_applied_edit_lines"),
                "total_copilot_applied_edit_calls": metrics.get("total_copilot_applied_edit_calls"),
                "total_copilot_text_edit_groups": metrics.get("total_copilot_text_edit_groups"),
                "total_copilot_edited_file_events": metrics.get("total_copilot_edited_file_events"),
                "total_copilot_distinct_edited_files": metrics.get("total_copilot_distinct_edited_files"),
                "total_visible_prompts": metrics.get("total_visible_prompts"),
                "total_redacted_prompts": metrics.get("total_redacted_prompts"),
                "total_tool_followup_turns": metrics.get("total_tool_followup_turns"),
                "retry_prompt_count": metrics.get("retry_prompt_count"),
                "prompt_visibility_rate": metrics.get("prompt_visibility_rate"),
                "prompt_metric_method": metrics.get("prompt_metric_method"),
                "feature_metric_method": metrics.get("feature_metric_method"),
                "total_prompt_tokens": metrics.get("total_prompt_tokens"),
                "total_completion_tokens": metrics.get("total_completion_tokens"),
                "total_actual_prompt_tokens": metrics.get("total_actual_prompt_tokens"),
                "total_actual_completion_tokens": metrics.get("total_actual_completion_tokens"),
                "total_estimated_prompt_tokens": metrics.get("total_estimated_prompt_tokens"),
                "total_estimated_completion_tokens": metrics.get("total_estimated_completion_tokens"),
                "total_assistant_code_lines": metrics.get("total_assistant_code_lines"),
                "total_assistant_fallback_code_lines": metrics.get("total_assistant_fallback_code_lines"),
                "total_tool_code_lines": metrics.get("total_tool_code_lines"),
                "total_metering_usage": metrics.get("total_metering_usage"),
                "metering_unit": metrics.get("metering_unit"),
                "average_context_usage_percentage": metrics.get("average_context_usage_percentage"),
                "total_prompts": metrics.get("total_prompts"),
                "completed_features": metrics.get("completed_features"),
                "total_code_responses": metrics.get("total_code_responses"),
                "flagged_reverts": metrics.get("flagged_reverts"),
                "measured_ai_output_definition": metrics.get("measured_ai_output_definition"),
                "token_count_methods": json.dumps(metrics.get("token_count_methods", {})),
                "tool_call_counts": json.dumps(metrics.get("tool_call_counts", {})),
                "copilot_edit_tool_counts": json.dumps(metrics.get("copilot_edit_tool_counts", {})),
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


def load_git_data(repo_path: str, phase_name: str | None = None) -> pd.DataFrame:
    """Load git history for a repository path.

    Args:
        repo_path: User-provided path to a local git repository root.
        phase_name: Optional phase name to map these commits to.

    Returns:
        A timestamp-sorted dataframe of commits when the path is valid,
        otherwise an empty dataframe.
    """
    if repo_path and os.path.isdir(repo_path):
        df_git = parse_git_history(repo_path)
        if not df_git.empty:
            if phase_name:
                df_git["phase"] = phase_name
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
    universal_files = discover_universal_files()
    selected_universal_files = universal_files
    
    # Pre-load universal metrics to discover all possible phases (like Phase 3)
    df_universal_metrics_base = load_universal_metrics(selected_universal_files)
    universal_phases = df_universal_metrics_base["phase"].dropna().unique().tolist() if not df_universal_metrics_base.empty else []
    all_discovered_phases = sorted(list(set(available_phases + universal_phases)))

    st.sidebar.divider()
    st.sidebar.subheader("Tracking Workbooks")
    discovered_tracking_workbooks = discover_tracking_workbooks()
    if discovered_tracking_workbooks:
        for phase, workbook_path in discovered_tracking_workbooks.items():
            st.sidebar.caption(f"{phase}: {workbook_path.name}")
    else:
        st.sidebar.caption("No local tracking workbooks were auto-detected.")

    st.sidebar.divider()
    st.sidebar.subheader("RITM PDF")
    discovered_ritm_pdf = discover_ritm_pdf()
    if discovered_ritm_pdf:
        st.sidebar.caption(discovered_ritm_pdf.name)
    else:
        st.sidebar.caption("No local RITM PDF was auto-detected.")

    st.sidebar.divider()
    st.sidebar.subheader("Local Git History")
    repo_path = st.sidebar.text_input(
        "Local Git Repository Path (Optional)",
        help="Path to the root of your local git repository to correlate chat with commits.",
    )
    repo_phase = st.sidebar.selectbox(
        "Git Repository Phase",
        options=[""] + all_discovered_phases,
        help="Select which phase this git repository belongs to.",
    ) if repo_path else ""

    # Merge uploaded files with discovered phase files into a single parse list.
    files_to_parse = list(uploaded_files) if uploaded_files else []
    files_to_parse.extend(gather_phase_files(selected_phases, DATA_DIR))

    if not files_to_parse and not selected_universal_files:
        st.info("Please select/upload files or ensure universal files exist to begin.")
        return

    df_chat_all = parse_chat_data(files_to_parse) if files_to_parse else pd.DataFrame()
    if not df_chat_all.empty:
        df_chat_all = df_chat_all.sort_values("timestamp")

    df_chat_analysis = pd.DataFrame()
    if not df_chat_all.empty:
        df_chat_analysis = build_analysis_filters_sidebar(df_chat_all)

    df_universal_all = parse_chat_data(selected_universal_files) if selected_universal_files else pd.DataFrame()
    if not df_universal_all.empty:
        df_universal_all = df_universal_all.sort_values("timestamp")

    df_universal_metrics = df_universal_metrics_base
    df_tracking_metrics = load_tracking_metrics(discovered_tracking_workbooks)
    df_universal_metrics = merge_tracking_metrics(df_universal_metrics, df_tracking_metrics)
    df_ritm_phase, df_ritm_notes = load_ritm_data(discovered_ritm_pdf)
    df_git = load_git_data(repo_path, phase_name=repo_phase)

    # Each tab delegates rendering to a focused function in views.tabs.
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Chat History",
            "Universal Chat History",
            "Statistics",
            "Executive Dashboard",
            "Comparative Analysis & Timelines",
        ]
    )

    with tab1:
        if df_chat_all.empty:
            st.info("No legacy chat data loaded from selected/uploaded files.")
        else:
            render_chat_history_tab(df_chat_all, key_prefix="legacy")

    with tab2:
        if df_universal_all.empty:
            st.info("No universal chat data loaded.")
        else:
            render_chat_history_tab(df_universal_all, key_prefix="universal")

    with tab3:
        if df_chat_analysis.empty:
            st.info("No legacy analysis data loaded.")
        else:
            render_statistics_tab(df_chat_analysis, df_git)

    with tab4:
        dash_sub1, dash_sub2, dash_sub3 = st.tabs([
            "Universal Dashboard", 
            "Unified Overview", 
            "Key Insights"
        ])
        
        with dash_sub1:
            render_universal_dashboard(
                df_universal_metrics,
                df_universal_all,
                df_git,
                df_ritm_phase,
                df_ritm_notes,
                df_chat_analysis,
            )
            
        with dash_sub2:
            render_unified_overview_tab(df_universal_metrics, df_universal_all, df_git, df_ritm_phase, df_ritm_notes)
            
        with dash_sub3:
            render_universal_key_insights_tab(df_universal_metrics, df_universal_all, df_ritm_phase, df_ritm_notes)

    with tab5:
        comp_sub1, comp_sub2, comp_sub3, comp_sub4, comp_sub5 = st.tabs([
            "Comparative Analysis",
            "Required Presentation Metrics",
            "Universal Timeline",
            "Legacy Development Timeline",
            "Prompt-to-Feature Flow"
        ])
        
        with comp_sub1:
            if df_chat_analysis.empty:
                st.info("No legacy analysis data loaded.")
            else:
                render_comparative_tab(df_chat_analysis)
                
        with comp_sub2:
            render_universal_required_stats_tab(df_universal_metrics, df_git, df_ritm_phase, df_ritm_notes)
            
        with comp_sub3:
            if df_universal_all.empty:
                st.info("No universal chat data loaded.")
            else:
                render_timeline_tab(df_universal_all, df_git)
                
        with comp_sub4:
            if df_chat_analysis.empty:
                st.info("No legacy analysis data loaded.")
            else:
                render_timeline_tab(df_chat_analysis, df_git)

        with comp_sub5:
            if df_chat_analysis.empty:
                st.info("No legacy analysis data loaded.")
            else:
                render_flow_visualization_tab(df_chat_analysis)


if __name__ == "__main__":
    main()
