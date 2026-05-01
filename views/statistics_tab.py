import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import json

from views.shared import *

def render_statistics_tab(df_chat_analysis: pd.DataFrame, df_git: pd.DataFrame) -> None:
    """Render the Statistics tab.

    Args:
        df_chat_analysis: Session-filtered chat dataframe for analysis.
        df_git: Optional git commit dataframe for human contribution overlays.

    Returns:
        None. Renders Streamlit UI elements directly.
    """
    st.subheader("Statistics: Who Created Content?")

    if df_chat_analysis.empty:
        st.warning("Please select at least one session in 'Analysis Filters' to view statistics.")
        return

    col1, col2 = st.columns(2)

    total_code, flagged_reverts = calculate_success_metrics(df_chat_analysis)
    success_rate = ((total_code - flagged_reverts) / total_code) * 100 if total_code > 0 else 100

    with col1:
        # AI-side contribution and quality metrics.
        st.write("### AI Contribution & Quality")
        total_code_lines = df_chat_analysis["code_lines_suggested"].sum()
        total_tokens = df_chat_analysis["completion_tokens"].sum()

        c1, c2 = st.columns(2)
        c1.metric("Total Code Lines", total_code_lines)
        c1.metric("Total Tokens", total_tokens)
        c2.metric("Success Rate (Est.)", f"{success_rate:.1f}%", help="Based on lack of negative follow-up prompts (e.g. 'fix', 'error')")
        c2.metric("Flagged Reverts", flagged_reverts, help="Number of responses followed immediately by a correction request.")

        if "model" in df_chat_analysis.columns:
            # Model usage share across selected sessions.
            model_counts = df_chat_analysis["model"].value_counts()
            fig_model = px.pie(model_counts, values=model_counts.values, names=model_counts.index, title="AI Models Used")
            st.plotly_chart(fig_model)

    with col2:
        # Git-side metrics are optional and only available with a valid repo path.
        if not df_git.empty:
            st.write("### Git Human Contribution")
            
            # If the git data is tagged to a specific phase, filter the chat analysis to match
            # so the volume comparison is apples-to-apples.
            git_phase = df_git["phase"].iloc[0] if "phase" in df_git.columns else ""
            if git_phase:
                filtered_chat_analysis = df_chat_analysis[df_chat_analysis["phase"] == git_phase]
                matched_code_lines = filtered_chat_analysis["code_lines_suggested"].sum() if not filtered_chat_analysis.empty else 0
                st.caption(f"Comparing against AI data for: **{git_phase}**")
            else:
                matched_code_lines = total_code_lines
            
            total_insertions = df_git["insertions"].sum()
            match_commits = len(df_git)
            st.metric("Total Git Insertions", total_insertions)
            st.metric("Total Commits", match_commits)

            author_counts = df_git["author"].value_counts()
            fig_author = px.pie(author_counts, values=author_counts.values, names=author_counts.index, title="Code Commits by Author")
            st.plotly_chart(fig_author)

            st.write("### Volume Comparison")
            # This is a volume comparison only; it does not imply code authorship.
            fig_comp = go.Figure(
                data=[
                    go.Bar(name="AI Suggested Lines", x=["Code Volume"], y=[matched_code_lines]),
                    go.Bar(name="Git Insertions", x=["Code Volume"], y=[total_insertions]),
                ]
            )
            fig_comp.update_layout(title="AI Suggestions vs Committed Code Volume")
            st.plotly_chart(fig_comp)
        else:
            st.info("Enter a valid Git Repository path to see Human/Git statistics.")

    st.divider()
    st.subheader("Deep Dive Analytics")

    col3, col4 = st.columns(2)

    with col3:
        st.write("### Response Latency by Model")
        if "latency_ms" in df_chat_analysis.columns:
            # Remove zero/failed latency events for cleaner distributions.
            df_latency = df_chat_analysis[df_chat_analysis["latency_ms"] > 0].copy()
            if not df_latency.empty:
                try:
                    # Convert ms to seconds for easier reading.
                    df_latency["latency_s"] = df_latency["latency_ms"] / 1000.0
                    df_latency["ttft_s"] = df_latency["ttft_ms"] / 1000.0

                    fig_latency = px.box(
                        df_latency,
                        x="model",
                        y="latency_s",
                        points="all",
                        title="Total Response Latency (seconds)",
                        hover_data={"latency_s": ":.2f", "model": False},
                    )
                    st.plotly_chart(fig_latency, width="stretch")

                    # TTFT is shown separately to highlight model thinking delay.
                    df_ttft = df_latency[df_latency["ttft_s"] > 0]
                    if not df_ttft.empty:
                        fig_ttft = px.box(
                            df_ttft,
                            x="model",
                            y="ttft_s",
                            points="all",
                            title="Thinking Time / TTFT (seconds)",
                            hover_data={"ttft_s": ":.2f", "model": False},
                        )
                        st.plotly_chart(fig_ttft, width="stretch")
                except Exception as e:
                    st.error(f"Error plotting latency: {e}")

    with col4:
        st.write("### Languages & Context")

        # Flatten language lists from all responses into one frequency table.
        all_langs = []
        for langs in df_chat_analysis["languages"]:
            if isinstance(langs, list):
                all_langs.extend(langs)

        if all_langs:
            lang_counts = pd.Series(all_langs).value_counts().reset_index()
            lang_counts.columns = ["Language", "Count"]
            fig_langs = px.bar(lang_counts, x="Language", y="Count", title="Top Programming Languages Generated")
            st.plotly_chart(fig_langs, width="stretch")
        else:
            st.info("No code blocks detected.")

        # Flatten file references to find most-used context files.
        all_files = []
        for files in df_chat_analysis.get("referenced_files", []):
            if isinstance(files, list):
                all_files.extend(files)

        if all_files:
            file_counts = pd.Series(all_files).value_counts().head(10).reset_index()
            file_counts.columns = ["File Name", "References"]
            fig_files = px.bar(file_counts, x="References", y="File Name", orientation="h", title="Top 10 Context Files")
            fig_files.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_files, width="stretch")
        else:
            st.info("No file context data found.")

    st.divider()
    st.subheader("Editor Events & Reliability")

    col5, col6 = st.columns(2)

    with col5:
        # Event counters from editor-side metadata attached to chat requests.
        total_checkpoints = df_chat_analysis.get("checkpoints_restored", pd.Series([0] * len(df_chat_analysis))).sum()
        editor_edits = df_chat_analysis.get("edited_file_events", pd.Series([0] * len(df_chat_analysis))).sum()

        st.metric(
            "Checkpoints Restored",
            total_checkpoints,
            help="Number of times an 'undoStop' event was recorded, likely indicating a revert or checkpoint restoration.",
        )
        st.metric("File Edit Events", editor_edits, help="Total number of file edit events triggered by the AI agent.")

    with col6:
        # Plot only rows that actually contain event activity.
        if total_checkpoints > 0 or editor_edits > 0:
            df_events = df_chat_analysis[
                (df_chat_analysis["checkpoints_restored"] > 0) | (df_chat_analysis["edited_file_events"] > 0)
            ]
            if not df_events.empty:
                fig_events = px.scatter(
                    df_events,
                    x="timestamp",
                    y="model",
                    size="edited_file_events",
                    color="checkpoints_restored",
                    title="Timeline of Edits and Restores",
                )
                st.plotly_chart(fig_events, width="stretch")
            else:
                st.info("No events to plot.")
        else:
            st.info("No checkpoint or edit events found in these sessions.")



