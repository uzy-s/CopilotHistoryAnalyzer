import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import json

from views.shared import *

def render_comparative_tab(df_chat_analysis: pd.DataFrame) -> None:
    """Render the Comparative Analysis tab.

    Args:
        df_chat_analysis: Session-filtered chat dataframe including phase labels.

    Returns:
        None. Renders Streamlit UI elements directly.
    """
    st.subheader("Comparative Analysis")

    # Prompt-level features power most cross-phase comparisons.
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

    # Split both chat-level and prompt-level views by selected phases.
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

    # Top-line normalized KPIs for quick phase comparison.
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

    # Convert boolean descriptor flags into comparable percentage rates by phase.
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

    # Model mix is normalized by phase to avoid sample-size bias.
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
        # Distribution chart helps detect spread, not just average shifts.
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
        # Prompt length mix is normalized to percentages per phase.
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
        # Daily shares are aligned to relative day numbers for phase-over-phase pacing.
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
    # Show recent troubleshooting prompts from both phases for qualitative review.
    focus_cols = ["Phase", "Session", "User", "Timestamp", "Word Count", "Complexity Score", "Prompt"]
    focus_df = pd.concat([df_phase_a[df_phase_a["Is Troubleshooting"]].head(10), df_phase_b[df_phase_b["Is Troubleshooting"]].head(10)])[focus_cols].sort_values(
        "Timestamp", ascending=False
    )
    st.dataframe(focus_df, width="stretch", hide_index=True)



