import os

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
)


st.set_page_config(page_title="Copilot History Analyzer", layout="wide")
require_password()

st.title("Copilot History Analyzer")


def build_chat_session_sidebar(df_chat_all: pd.DataFrame) -> str | None:
    st.sidebar.divider()
    st.sidebar.subheader("Chat History View")

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
        "Select Chat Session to View",
        options=display_options,
        index=0 if display_options else None,
    )
    return session_map.get(selected_display_label)


def build_analysis_filters_sidebar(df_chat_all: pd.DataFrame) -> pd.DataFrame:
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
    if repo_path and os.path.isdir(repo_path):
        df_git = parse_git_history(repo_path)
        if not df_git.empty:
            return df_git.sort_values("timestamp")
    return pd.DataFrame()


def main() -> None:
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
    repo_path = st.sidebar.text_input(
        "Local Git Repository Path (Optional)",
        help="Path to the root of your local git repository to correlate chat with commits.",
    )

    files_to_parse = list(uploaded_files) if uploaded_files else []
    files_to_parse.extend(gather_phase_files(selected_phases, DATA_DIR))

    if not files_to_parse:
        st.info("Please select one or more phases or upload a 'chatTemplate.json' file to begin.")
        return

    df_chat_all = parse_chat_data(files_to_parse)
    if df_chat_all.empty:
        st.warning("No valid chat requests found in the selected files.")
        return

    df_chat_all = df_chat_all.sort_values("timestamp")
    selected_chat_file = build_chat_session_sidebar(df_chat_all)
    df_chat_analysis = build_analysis_filters_sidebar(df_chat_all)
    df_git = load_git_data(repo_path)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Chat History", "Statistics", "Development Timeline", "Comparative Analysis", "Prompt Analysis"]
    )

    with tab1:
        render_chat_history_tab(df_chat_all, selected_chat_file)

    with tab2:
        render_statistics_tab(df_chat_analysis, df_git)

    with tab3:
        render_timeline_tab(df_chat_analysis, df_git)

    with tab4:
        render_comparative_tab(df_chat_analysis)

    with tab5:
        render_prompt_analysis_tab(df_chat_analysis)


if __name__ == "__main__":
    main()
