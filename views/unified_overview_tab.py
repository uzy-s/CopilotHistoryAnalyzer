import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import json

from views.shared import *

def render_unified_overview_tab(
    df_universal_metrics: pd.DataFrame,
    df_universal_all: pd.DataFrame,
    df_git: pd.DataFrame | None = None,
    df_ritm_phase: pd.DataFrame | None = None,
    df_ritm_notes: pd.DataFrame | None = None,
) -> None:
    """Render a phase-oriented unified overview dashboard."""
    st.subheader("Unified Overview")
    st.write("Cross-phase dashboard for the normalized universal dataset.")
    st.caption(
        "Natural prompts and visible prompts are separated here. Heuristic feature runs and prompt success "
        "are derived from visible prompts only so redacted Phase 3 Kiro prompts do not overstate delivery. "
        "Measured AI output prefers structured applied editor/tool edits when the logs preserve them."
    )

    phase_df = _aggregate_universal_phase_metrics(df_universal_metrics)
    if phase_df.empty:
        st.info("No universal metrics loaded. Select universal files in the sidebar.")
        return

    total_human_lines_series = phase_df["total_lines_written_by_humans"].dropna()
    total_human_lines = total_human_lines_series.sum() if not total_human_lines_series.empty else None

    card1, card2, card3, card4 = st.columns(4)
    card1.metric("Phases Loaded", f"{phase_df['phase'].nunique():.0f}")
    card2.metric("Contributors", f"{df_universal_metrics['user_id'].nunique():.0f}" if "user_id" in df_universal_metrics else "0")
    card3.metric("Natural Prompts", f"{phase_df['total_prompts'].fillna(0).sum():.0f}")
    card4.metric("Visible Prompts", f"{phase_df['total_visible_prompts'].fillna(0).sum():.0f}")

    card5, card6, card7, card8 = st.columns(4)
    card5.metric("Total Work Hours", f"{phase_df['man_hours'].fillna(0).sum():.2f}")
    card6.metric("Team Man-Days", f"{phase_df['man_days'].fillna(0).sum():.0f}")
    card7.metric(
        "Measured AI Output Lines",
        f"{phase_df['total_lines_written_by_selected_model_agent'].fillna(0).sum():.0f}",
    )
    card8.metric("Feature Runs (Heuristic)", f"{phase_df['completed_features'].fillna(0).sum():.0f}")

    ritm_phase_df = _prepare_ritm_phase_df(df_ritm_phase if df_ritm_phase is not None else pd.DataFrame())
    if not ritm_phase_df.empty:
        st.divider()
        st.write("### RITM Quality Evaluation")
        st.caption("Requirement adherence and documented quality notes parsed from the Michelin RITM PDF.")
        _render_ritm_graphics(ritm_phase_df, df_ritm_notes if df_ritm_notes is not None else pd.DataFrame(), "unified_overview", compact=True)

    st.divider()
    st.write("### Phase Delivery Profile")

    col1, col2 = st.columns(2)
    with col1:
        effort_df = phase_df.melt(
            id_vars="phase",
            value_vars=["man_hours", "man_days"],
            var_name="metric",
            value_name="value",
        )
        effort_df["metric"] = effort_df["metric"].replace({"man_hours": "Work Hours", "man_days": "Team Man-Days"})
        fig_effort = px.bar(
            effort_df,
            x="phase",
            y="value",
            color="metric",
            barmode="group",
            title="Team Effort by Phase",
            color_discrete_map=SIGNAL_COLORS,
        )
        st.plotly_chart(_style_figure(fig_effort), width="stretch", key="unified_overview_effort")

    with col2:
        output_df = phase_df.melt(
            id_vars="phase",
            value_vars=["total_prompts", "total_visible_prompts", "completed_features"],
            var_name="metric",
            value_name="value",
        )
        output_df["metric"] = output_df["metric"].replace(
            {
                "total_prompts": "Natural Prompts",
                "total_visible_prompts": "Visible Prompts",
                "completed_features": "Heuristic Feature Runs",
            }
        )
        fig_output = px.bar(
            output_df,
            x="phase",
            y="value",
            color="metric",
            barmode="group",
            title="Prompt Load vs Delivery Runs",
            color_discrete_map=SIGNAL_COLORS,
        )
        st.plotly_chart(_style_figure(fig_output), width="stretch", key="unified_overview_output")

    st.write("### Efficiency & Quality")

    col3, col4 = st.columns(2)
    with col3:
        success_df = phase_df.copy()
        success_df["prompt_success_rate_pct"] = success_df["prompt_success_rate"].fillna(0) * 100.0
        fig_success = px.bar(
            success_df,
            x="phase",
            y="prompt_success_rate_pct",
            color="phase",
            color_discrete_map=PHASE_COLORS,
            title="Visible Prompt Success Rate by Phase",
            labels={"prompt_success_rate_pct": "Success Rate (%)", "phase": "Phase"},
        )
        st.plotly_chart(_style_figure(fig_success), width="stretch", key="unified_overview_success")

    with col4:
        feature_df = phase_df.copy()
        fig_feature = px.bar(
            feature_df,
            x="phase",
            y="prompt_to_feature_ratio",
            color="phase",
            color_discrete_map=PHASE_COLORS,
            title="Visible Prompts per Heuristic Feature Run",
            labels={"prompt_to_feature_ratio": "Visible Prompts per Run", "phase": "Phase"},
        )
        st.plotly_chart(_style_figure(fig_feature), width="stretch", key="unified_overview_feature_ratio")

    st.write("### Contribution & Coverage")

    col5, col6 = st.columns(2)
    with col5:
        contribution_cols = [
            "total_lines_written_by_selected_model_agent",
            "number_of_ai_lines_needing_human_revision",
        ]
        if phase_df["total_lines_written_by_humans"].notna().any():
            contribution_cols.append("total_lines_written_by_humans")

        contribution_df = phase_df.melt(
            id_vars="phase",
            value_vars=contribution_cols,
            var_name="metric",
            value_name="value",
        )
        contribution_df["metric"] = contribution_df["metric"].replace(
            {
                "total_lines_written_by_selected_model_agent": "Measured AI Output Lines",
                "number_of_ai_lines_needing_human_revision": "AI Lines Revised",
                "total_lines_written_by_humans": "Human Lines",
            }
        )
        fig_contribution = px.bar(
            contribution_df,
            x="phase",
            y="value",
            color="metric",
            barmode="group",
            title="Contribution & Revision Mix by Phase",
            color_discrete_map=SIGNAL_COLORS,
        )
        st.plotly_chart(_style_figure(fig_contribution), width="stretch", key="unified_overview_contribution")

        if total_human_lines is not None:
            st.caption(f"Human lines currently loaded from external data: {total_human_lines:.0f}.")

    with col6:
        token_df = phase_df.melt(
            id_vars="phase",
            value_vars=["total_actual_tokens", "total_estimated_tokens", "total_metering_usage"],
            var_name="metric",
            value_name="value",
        )
        token_df["metric"] = token_df["metric"].replace(
            {
                "total_actual_tokens": "Actual Tokens",
                "total_estimated_tokens": "Estimated Tokens",
                "total_metering_usage": "Kiro Credits",
            }
        )
        fig_token = px.bar(
            token_df,
            x="phase",
            y="value",
            color="metric",
            barmode="group",
            title="Token & Credit Coverage by Phase",
            color_discrete_map=SIGNAL_COLORS,
        )
        st.plotly_chart(_style_figure(fig_token), width="stretch", key="unified_overview_token")

    if not df_universal_all.empty:
        st.divider()
        st.write("### Unified Timeline")

        daily_activity = df_universal_all.copy()
        daily_activity["date"] = daily_activity["timestamp"].dt.date
        daily_activity = (
            daily_activity.groupby(["date", "phase"]).size().reset_index(name="interaction_count")
        )
        daily_activity = _sort_phase_frame(daily_activity)

        fig_timeline = px.line(
            daily_activity,
            x="date",
            y="interaction_count",
            color="phase",
            markers=True,
            color_discrete_map=PHASE_COLORS,
            title="Daily Universal Activity by Phase",
            labels={"interaction_count": "Interactions", "date": "Date"},
        )

        if df_git is not None and not df_git.empty:
            git_daily = df_git.copy()
            git_daily["date"] = git_daily["timestamp"].dt.date
            git_daily = git_daily.groupby("date").size().reset_index(name="commit_count")
            fig_timeline.add_trace(
                go.Scatter(
                    x=git_daily["date"],
                    y=git_daily["commit_count"],
                    mode="lines+markers",
                    name="Git Commits",
                    yaxis="y2",
                )
            )
            fig_timeline.update_layout(
                yaxis2=dict(
                    title="Git Commits",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                )
            )

        st.plotly_chart(_style_figure(fig_timeline), width="stretch", key="unified_overview_timeline")

    st.divider()
    st.write("### Phase Summary Table")
    summary_df = phase_df.copy()
    summary_df["prompt_visibility_rate"] = summary_df["prompt_visibility_rate"].apply(
        lambda v: f"{v * 100.0:.1f}%" if pd.notna(v) else "N/A"
    )
    summary_df["prompt_success_rate"] = summary_df["prompt_success_rate"].apply(
        lambda v: f"{v * 100.0:.1f}%" if pd.notna(v) else "N/A"
    )
    summary_df["prompt_to_feature_ratio"] = summary_df["prompt_to_feature_ratio"].apply(
        lambda v: f"{v:.2f}" if pd.notna(v) else "N/A"
    )
    summary_df["average_context_usage_percentage"] = summary_df["average_context_usage_percentage"].apply(
        lambda v: f"{v:.2f}%" if pd.notna(v) else "N/A"
    )
    if not ritm_phase_df.empty:
        summary_df = summary_df.merge(
            ritm_phase_df[["phase", "requirement_adherence_percent", "documented_issue_count"]],
            on="phase",
            how="left",
        )
        summary_df["requirement_adherence_percent"] = summary_df["requirement_adherence_percent"].apply(
            lambda v: f"{v:.2f}%" if pd.notna(v) else "N/A"
        )
        summary_df["documented_issue_count"] = summary_df["documented_issue_count"].apply(
            lambda v: f"{v:.0f}" if pd.notna(v) else "0"
        )
    summary_df = summary_df.rename(
        columns={
            "phase": "Phase",
            "contributors": "Contributors",
            "man_hours": "Work Hours",
            "man_days": "Team Man-Days",
            "total_prompts": "Natural Prompts",
            "total_visible_prompts": "Visible Prompts",
            "total_redacted_prompts": "Redacted Prompts",
            "completed_features": "Heuristic Feature Runs",
            "prompt_success_rate": "Prompt Success",
            "prompt_to_feature_ratio": "Visible Prompts per Run",
            "total_lines_written_by_selected_model_agent": "AI Lines",
            "number_of_ai_lines_needing_human_revision": "AI Lines Revised",
            "total_lines_written_by_humans": "Human Lines",
            "total_metering_usage": "Kiro Credits",
            "average_context_usage_percentage": "Avg Context Usage",
            "requirement_adherence_percent": "RITM Adherence",
            "documented_issue_count": "RITM Issue Notes",
        }
    )
    st.dataframe(summary_df, width="stretch", hide_index=True)



