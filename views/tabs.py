import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from services.analytics import (
    analyze_prompt_style,
    avg_interactions_per_day,
    calculate_success_metrics,
    error_rate_percent,
    phase_duration_days,
    safe_mean,
    safe_rate,
    tokens_per_prompt,
    top_model_share_percent,
)


def render_chat_history_tab(df_chat_all: pd.DataFrame, selected_chat_file: str | None) -> None:
    st.subheader("Recreated Chat Session")

    if selected_chat_file:
        df_chat_view = df_chat_all[df_chat_all["file_name"] == selected_chat_file]

        user_info = df_chat_view["suspected_user"].iloc[0] if not df_chat_view.empty else "Unknown"
        st.caption(f"Viewing Session: **{selected_chat_file}** | User: **{user_info}**")

        if df_chat_view.empty:
            st.info("No messages in this session.")

        with st.container(height=600):
            for _, row in df_chat_view.iterrows():
                with st.chat_message("user"):
                    st.markdown(f"**{row['suspected_user']}**")
                    st.write(row["user_text"])
                    st.caption(f"{row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")

                assistant_header = f"{row['model']} | Code Lines: {row['code_lines_suggested']}"
                with st.chat_message("assistant"):
                    st.markdown(f"**{assistant_header}**")
                    st.markdown(row["assistant_text"])
                    st.caption(f"Tokens: {row['completion_tokens']}")

                st.divider()
    else:
        st.info("Select a chat session from the sidebar to view.")


def render_statistics_tab(df_chat_analysis: pd.DataFrame, df_git: pd.DataFrame) -> None:
    st.subheader("Statistics: Who Created Content?")

    if df_chat_analysis.empty:
        st.warning("Please select at least one session in 'Analysis Filters' to view statistics.")
        return

    col1, col2 = st.columns(2)

    total_code, flagged_reverts = calculate_success_metrics(df_chat_analysis)
    success_rate = ((total_code - flagged_reverts) / total_code) * 100 if total_code > 0 else 100

    with col1:
        st.write("### AI Contribution & Quality")
        total_code_lines = df_chat_analysis["code_lines_suggested"].sum()
        total_tokens = df_chat_analysis["completion_tokens"].sum()

        c1, c2 = st.columns(2)
        c1.metric("Total Code Lines", total_code_lines)
        c1.metric("Total Tokens", total_tokens)
        c2.metric("Success Rate (Est.)", f"{success_rate:.1f}%", help="Based on lack of negative follow-up prompts (e.g. 'fix', 'error')")
        c2.metric("Flagged Reverts", flagged_reverts, help="Number of responses followed immediately by a correction request.")

        if "model" in df_chat_analysis.columns:
            model_counts = df_chat_analysis["model"].value_counts()
            fig_model = px.pie(model_counts, values=model_counts.values, names=model_counts.index, title="AI Models Used")
            st.plotly_chart(fig_model)

    with col2:
        if not df_git.empty:
            st.write("### Git Human Contribution")
            total_insertions = df_git["insertions"].sum()
            match_commits = len(df_git)
            st.metric("Total Git Insertions", total_insertions)
            st.metric("Total Commits", match_commits)

            author_counts = df_git["author"].value_counts()
            fig_author = px.pie(author_counts, values=author_counts.values, names=author_counts.index, title="Code Commits by Author")
            st.plotly_chart(fig_author)

            st.write("### Volume Comparison")
            fig_comp = go.Figure(
                data=[
                    go.Bar(name="AI Suggested Lines", x=["Code Volume"], y=[total_code_lines]),
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
            df_latency = df_chat_analysis[df_chat_analysis["latency_ms"] > 0].copy()
            if not df_latency.empty:
                try:
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
        total_checkpoints = df_chat_analysis.get("checkpoints_restored", pd.Series([0] * len(df_chat_analysis))).sum()
        editor_edits = df_chat_analysis.get("edited_file_events", pd.Series([0] * len(df_chat_analysis))).sum()

        st.metric(
            "Checkpoints Restored",
            total_checkpoints,
            help="Number of times an 'undoStop' event was recorded, likely indicating a revert or checkpoint restoration.",
        )
        st.metric("File Edit Events", editor_edits, help="Total number of file edit events triggered by the AI agent.")

    with col6:
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


def render_timeline_tab(df_chat_analysis: pd.DataFrame, df_git: pd.DataFrame) -> None:
    st.subheader("Development History Timeline")

    timeline_data = []

    for _, row in df_chat_analysis.iterrows():
        timeline_data.append(
            {
                "timestamp": row["timestamp"],
                "type": "Chat Interaction",
                "count": 1,
                "code_volume": row["code_lines_suggested"],
                "session": row["file_name"],
                "details": f"Model: {row['model']} | Session: {row['file_name']}",
            }
        )

    if not df_git.empty:
        for _, row in df_git.iterrows():
            timeline_data.append(
                {
                    "timestamp": row["timestamp"],
                    "type": "Git Commit",
                    "count": 1,
                    "code_volume": row["insertions"],
                    "session": "Git Repo",
                    "details": f"Author: {row['author']} | Insertions: {row['insertions']}",
                }
            )

    df_timeline = pd.DataFrame(timeline_data)

    if df_timeline.empty:
        return

    fig_timeline = px.scatter(
        df_timeline,
        x="timestamp",
        y="type",
        size="code_volume",
        color="session",
        symbol="type",
        hover_data=["details"],
        title="Activity Timeline (Color=Session, Symbol=Type)",
    )
    st.plotly_chart(fig_timeline, width="stretch")

    st.write("### Code Velocity (AI Suggestions vs User Commits)")
    fig_velocity = px.line(
        df_timeline.sort_values("timestamp"),
        x="timestamp",
        y="code_volume",
        color="type",
        title="Code Volume Over Time",
    )
    st.plotly_chart(fig_velocity, width="stretch")

    st.write("### Daily Activity Volume")
    df_timeline["date"] = df_timeline["timestamp"].dt.date
    daily_counts = df_timeline.groupby(["date", "type"]).size().reset_index(name="count")

    fig_daily = px.bar(daily_counts, x="date", y="count", color="type", title="Daily Interactions vs Commits")
    st.plotly_chart(fig_daily, width="stretch")


def render_comparative_tab(df_chat_analysis: pd.DataFrame) -> None:
    st.subheader("Comparative Analysis")

    prompt_df = analyze_prompt_style(df_chat_analysis)
    if prompt_df.empty:
        st.warning("No prompts available for phase comparison in current filters.")
        return

    available_prompt_phases = sorted(prompt_df["Phase"].dropna().unique().tolist())
    if len(available_prompt_phases) < 2:
        st.warning("Load at least two phases to use Phase-to-Phase comparison.")
        return

    col_phase_1, col_phase_2 = st.columns(2)
    with col_phase_1:
        phase_a = st.selectbox("Select Baseline Phase", options=available_prompt_phases, index=0)
    with col_phase_2:
        default_phase_idx = 1 if len(available_prompt_phases) > 1 else 0
        phase_b = st.selectbox("Select Comparison Phase", options=available_prompt_phases, index=default_phase_idx)

    if phase_a == phase_b:
        st.info("Choose two different phases to compare.")
        return

    df_phase_chat_a = df_chat_analysis[df_chat_analysis["phase"] == phase_a].copy()
    df_phase_chat_b = df_chat_analysis[df_chat_analysis["phase"] == phase_b].copy()
    df_phase_a = prompt_df[prompt_df["Phase"] == phase_a].copy()
    df_phase_b = prompt_df[prompt_df["Phase"] == phase_b].copy()

    prompts_a = len(df_phase_a)
    prompts_b = len(df_phase_b)
    dur_a = phase_duration_days(df_phase_chat_a)
    dur_b = phase_duration_days(df_phase_chat_b)
    err_a = error_rate_percent(df_phase_chat_a)
    err_b = error_rate_percent(df_phase_chat_b)
    tokens_prompt_a = tokens_per_prompt(df_phase_chat_a)
    tokens_prompt_b = tokens_per_prompt(df_phase_chat_b)
    int_per_day_a = avg_interactions_per_day(df_phase_chat_a)
    int_per_day_b = avg_interactions_per_day(df_phase_chat_b)
    top_share_a, top_model_a = top_model_share_percent(df_phase_chat_a)
    top_share_b, top_model_b = top_model_share_percent(df_phase_chat_b)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prompts (Comparison)", f"{prompts_b}", f"{prompts_b - prompts_a}", help=f"Baseline {phase_a}: {prompts_a}")
    c2.metric(
        "Complexity Shift",
        f"{safe_mean(df_phase_b['Complexity Score']):.2f}",
        f"{(safe_mean(df_phase_b['Complexity Score']) - safe_mean(df_phase_a['Complexity Score'])):+.2f}",
        help=f"Average complexity in {phase_b} vs {phase_a}",
    )
    c3.metric(
        "Troubleshooting Rate",
        f"{safe_rate(df_phase_b, 'Is Troubleshooting'):.1f}%",
        f"{(safe_rate(df_phase_b, 'Is Troubleshooting') - safe_rate(df_phase_a, 'Is Troubleshooting')):+.1f}%",
        help="Normalized by prompt count",
    )
    c4.metric(
        "Error Rate",
        f"{err_b:.1f}%",
        f"{(err_b - err_a):+.1f}%",
        help="Flagged reverts per code-producing responses (normalized)",
    )

    st.caption("Primary values in this section are normalized by prompt count or phase span so phase sizes are comparable.")
    st.write("### Side-by-Side Phase Summary")
    summary_df = pd.DataFrame(
        [
            {"Metric": "Phase Length (days)", phase_a: f"{dur_a:.2f}", phase_b: f"{dur_b:.2f}"},
            {"Metric": "Interactions / Day", phase_a: f"{int_per_day_a:.2f}", phase_b: f"{int_per_day_b:.2f}"},
            {"Metric": "Tokens / Prompt", phase_a: f"{tokens_prompt_a:.1f}", phase_b: f"{tokens_prompt_b:.1f}"},
            {
                "Metric": "Top Model Usage Share",
                phase_a: f"{top_share_a:.1f}% ({top_model_a})",
                phase_b: f"{top_share_b:.1f}% ({top_model_b})",
            },
            {
                "Metric": "Troubleshooting Prompt Rate",
                phase_a: f"{safe_rate(df_phase_a, 'Is Troubleshooting'):.1f}%",
                phase_b: f"{safe_rate(df_phase_b, 'Is Troubleshooting'):.1f}%",
            },
            {
                "Metric": "Avg Prompt Word Count",
                phase_a: f"{safe_mean(df_phase_a['Word Count']):.1f}",
                phase_b: f"{safe_mean(df_phase_b['Word Count']):.1f}",
            },
        ]
    )
    st.dataframe(summary_df, width="stretch", hide_index=True)
    st.divider()

    descriptor_rows = [
        {"Metric": "Inquisitive", phase_a: safe_rate(df_phase_a, "Is Inquisitive"), phase_b: safe_rate(df_phase_b, "Is Inquisitive")},
        {"Metric": "Polite", phase_a: safe_rate(df_phase_a, "Is Polite"), phase_b: safe_rate(df_phase_b, "Is Polite")},
        {"Metric": "Direct/Commanding", phase_a: safe_rate(df_phase_a, "Is Direct"), phase_b: safe_rate(df_phase_b, "Is Direct")},
        {"Metric": "Detailed Request", phase_a: safe_rate(df_phase_a, "Is Detailed"), phase_b: safe_rate(df_phase_b, "Is Detailed")},
        {"Metric": "Troubleshooting", phase_a: safe_rate(df_phase_a, "Is Troubleshooting"), phase_b: safe_rate(df_phase_b, "Is Troubleshooting")},
    ]
    df_descriptor_rates = pd.DataFrame(descriptor_rows).melt(id_vars=["Metric"], var_name="Phase", value_name="Rate")
    fig_descriptor_rates = px.bar(
        df_descriptor_rates,
        x="Metric",
        y="Rate",
        color="Phase",
        barmode="group",
        title="Descriptor Rates by Phase (%)",
    )
    st.plotly_chart(fig_descriptor_rates, width="stretch")

    model_dist = pd.concat([df_phase_chat_a.assign(PhaseLabel=phase_a), df_phase_chat_b.assign(PhaseLabel=phase_b)])
    model_share = model_dist.groupby(["PhaseLabel", "model"]).size().reset_index(name="count")
    model_share["share_pct"] = model_share.groupby("PhaseLabel")["count"].transform(lambda x: x / x.sum() * 100.0)
    fig_model_share = px.bar(
        model_share,
        x="model",
        y="share_pct",
        color="PhaseLabel",
        barmode="group",
        title="Model Usage Comparison (% of prompts)",
    )
    st.plotly_chart(fig_model_share, width="stretch")

    col_dist_1, col_dist_2 = st.columns(2)
    with col_dist_1:
        df_complexity_compare = pd.concat([df_phase_a.assign(PhaseLabel=phase_a), df_phase_b.assign(PhaseLabel=phase_b)])
        fig_complexity = px.box(
            df_complexity_compare,
            x="PhaseLabel",
            y="Complexity Score",
            color="PhaseLabel",
            title="Complexity Distribution by Phase",
        )
        st.plotly_chart(fig_complexity, width="stretch")

    with col_dist_2:
        df_len_bucket = pd.concat([df_phase_a.assign(PhaseLabel=phase_a), df_phase_b.assign(PhaseLabel=phase_b)])
        length_counts = df_len_bucket.groupby(["PhaseLabel", "Prompt Length Bucket"]).size().reset_index(name="count")
        length_counts["share_pct"] = length_counts.groupby("PhaseLabel")["count"].transform(lambda x: x / x.sum() * 100.0)
        fig_length = px.bar(
            length_counts,
            x="Prompt Length Bucket",
            y="share_pct",
            color="PhaseLabel",
            barmode="group",
            category_orders={"Prompt Length Bucket": ["Short", "Medium", "Long"]},
            title="Prompt Length Profile (% of prompts)",
        )
        st.plotly_chart(fig_length, width="stretch")

    st.write("### Daily Interaction Comparison")
    daily_a = df_phase_chat_a.copy()
    daily_b = df_phase_chat_b.copy()
    daily_a["date"] = daily_a["timestamp"].dt.date
    daily_b["date"] = daily_b["timestamp"].dt.date
    daily_counts_a = daily_a.groupby("date").size().reset_index(name="count")
    daily_counts_b = daily_b.groupby("date").size().reset_index(name="count")

    if not daily_counts_a.empty:
        min_date_a = daily_counts_a["date"].min()
        daily_counts_a["Relative Day"] = (pd.to_datetime(daily_counts_a["date"]) - pd.to_datetime(min_date_a)).dt.days + 1
        daily_counts_a["Phase"] = phase_a
        daily_counts_a["Daily Share %"] = (daily_counts_a["count"] / daily_counts_a["count"].sum()) * 100.0

    if not daily_counts_b.empty:
        min_date_b = daily_counts_b["date"].min()
        daily_counts_b["Relative Day"] = (pd.to_datetime(daily_counts_b["date"]) - pd.to_datetime(min_date_b)).dt.days + 1
        daily_counts_b["Phase"] = phase_b
        daily_counts_b["Daily Share %"] = (daily_counts_b["count"] / daily_counts_b["count"].sum()) * 100.0

    daily_compare = pd.concat([daily_counts_a, daily_counts_b], ignore_index=True)
    if not daily_compare.empty:
        fig_daily_compare = px.line(
            daily_compare,
            x="Relative Day",
            y="Daily Share %",
            color="Phase",
            markers=True,
            title="Daily Interaction Share by Relative Day (%)",
        )
        st.plotly_chart(fig_daily_compare, width="stretch")
    else:
        st.info("Not enough data to build a daily interaction comparison.")

    st.write("### Prompts Driving Troubleshooting Delta")
    focus_cols = ["Phase", "Session", "User", "Timestamp", "Word Count", "Complexity Score", "Prompt"]
    focus_df = pd.concat([df_phase_a[df_phase_a["Is Troubleshooting"]].head(10), df_phase_b[df_phase_b["Is Troubleshooting"]].head(10)])[focus_cols].sort_values(
        "Timestamp", ascending=False
    )
    st.dataframe(focus_df, width="stretch", hide_index=True)


def render_prompt_analysis_tab(df_chat_analysis: pd.DataFrame) -> None:
    st.subheader("Prompt Analysis & Styling")
    st.write("This tab takes in all the extracted user prompts and categorizes them based on their tone, length, and style descriptors.")

    if df_chat_analysis.empty:
        st.warning("Please select at least one session in 'Analysis Filters' to view prompt analysis.")
        return

    prompt_df = analyze_prompt_style(df_chat_analysis)
    if prompt_df.empty:
        st.info("No text prompts found in the selected sessions.")
        return

    p_col1, p_col2, p_col3 = st.columns(3)
    p_col1.metric("Total Prompts Analyzed", len(prompt_df))
    p_col2.metric("Average Word Count", f"{prompt_df['Word Count'].mean():.1f}")
    p_col3.metric("Average Complexity Score", f"{prompt_df['Complexity Score'].mean():.1f}")

    st.divider()

    st.write("### Prompt Repository")
    st.dataframe(
        prompt_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Prompt": st.column_config.TextColumn(width="large"),
            "Complexity Score": st.column_config.ProgressColumn(format="%d", min_value=1, max_value=10),
        },
    )

    st.divider()
    st.write("### Tone & Style Breakdown")

    all_desc = []
    for desc in prompt_df["Style Descriptors"]:
        if desc:
            all_desc.extend([d.strip() for d in desc.split(",")])

    if all_desc:
        desc_counts = pd.Series(all_desc).value_counts().reset_index()
        desc_counts.columns = ["Descriptor", "Count"]

        fig_desc = px.bar(desc_counts, x="Count", y="Descriptor", orientation="h", title="Distribution of Prompt Tone/Style")
        fig_desc.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_desc, width="stretch")
