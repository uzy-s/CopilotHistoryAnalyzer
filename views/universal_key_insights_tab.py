import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import json

from views.shared import *

def render_universal_key_insights_tab(
    df_universal_metrics: pd.DataFrame,
    df_universal_all: pd.DataFrame,
    df_ritm_phase: pd.DataFrame | None = None,
    df_ritm_notes: pd.DataFrame | None = None,
) -> None:
    """Render key insights and overview-document KPI coverage."""
    st.subheader("Unified Key Insights")
    st.write("Key findings and KPI coverage aligned to the capstone overview document.")
    st.caption(
        "Prompt success and prompt-to-feature views below are based on visible natural prompts. "
        "Redacted Phase 3 prompts still count toward workload, but not toward visible-prompt success heuristics."
    )

    phase_df = _aggregate_universal_phase_metrics(df_universal_metrics)
    if phase_df.empty:
        st.info("No universal metrics loaded. Select universal files in the sidebar.")
        return

    insights: list[str] = []

    success_candidates = phase_df.dropna(subset=["prompt_success_rate"])
    if not success_candidates.empty:
        best_success = success_candidates.sort_values("prompt_success_rate", ascending=False).iloc[0]
        insights.append(
            f"Highest prompt success rate: {best_success['phase']} at {best_success['prompt_success_rate'] * 100.0:.1f}%."
        )

    feature_candidates = phase_df[
        phase_df["prompt_to_feature_ratio"].notna() & (phase_df["prompt_to_feature_ratio"] > 0)
    ]
    if not feature_candidates.empty:
        best_efficiency = feature_candidates.sort_values("prompt_to_feature_ratio").iloc[0]
        insights.append(
            f"Lowest visible-prompts-per-feature-run burden: {best_efficiency['phase']} at {best_efficiency['prompt_to_feature_ratio']:.2f}."
        )

    output_candidates = phase_df.dropna(subset=["total_lines_written_by_selected_model_agent"])
    if not output_candidates.empty:
        highest_output = output_candidates.sort_values("total_lines_written_by_selected_model_agent", ascending=False).iloc[0]
        insights.append(
            f"Highest AI output volume: {highest_output['phase']} with {highest_output['total_lines_written_by_selected_model_agent']:.0f} AI-written lines."
        )

    effort_candidates = phase_df.dropna(subset=["man_hours"])
    if not effort_candidates.empty:
        highest_effort = effort_candidates.sort_values("man_hours", ascending=False).iloc[0]
        insights.append(
            f"Most team effort recorded: {highest_effort['phase']} with {highest_effort['man_hours']:.2f} work hours."
        )

    visibility_candidates = phase_df.dropna(subset=["prompt_visibility_rate"])
    if not visibility_candidates.empty:
        best_visibility = visibility_candidates.sort_values("prompt_visibility_rate", ascending=False).iloc[0]
        insights.append(
            f"Highest prompt visibility coverage: {best_visibility['phase']} at {best_visibility['prompt_visibility_rate'] * 100.0:.1f}% visible prompts."
        )

    ritm_phase_df = _prepare_ritm_phase_df(df_ritm_phase if df_ritm_phase is not None else pd.DataFrame())
    if not ritm_phase_df.empty:
        best_ritm = ritm_phase_df.sort_values("requirement_adherence_score", ascending=False).iloc[0]
        insights.append(
            f"Highest requirement adherence: {best_ritm['phase']} at {best_ritm['requirement_adherence_percent']:.2f}%."
        )
        most_notes = ritm_phase_df.sort_values("documented_issue_count", ascending=False).iloc[0]
        if pd.notna(most_notes["documented_issue_count"]) and float(most_notes["documented_issue_count"]) > 0:
            insights.append(
                f"Most documented RITM issues: {most_notes['phase']} with {int(most_notes['documented_issue_count'])} noted gaps."
            )

    st.write("### Key Findings")
    if insights:
        for insight in insights:
            st.markdown(f"- {insight}")
    else:
        st.info("Not enough data is loaded yet to compute key findings.")

    st.divider()
    st.write("### Overview Document KPI Coverage")

    human_available = "total_lines_written_by_humans" in df_universal_metrics.columns and not df_universal_metrics["total_lines_written_by_humans"].dropna().empty
    overview_coverage = pd.DataFrame(
        [
            {"Metric": "Total lines written by humans", "Availability": "Available" if human_available else "External / Not loaded", "Source": "Universal metric when provided"},
            {"Metric": "Total lines written by selected model / agent", "Availability": "Available", "Source": "Universal metrics"},
            {"Metric": "Number of AI lines needing human revision", "Availability": "Available", "Source": "Universal metrics (heuristic-compatible)"},
            {"Metric": "Total work hours", "Availability": "Available", "Source": "Universal metrics from sessionized timestamps"},
            {"Metric": "Team man-days", "Availability": "Available", "Source": "Universal metrics from timestamps"},
            {"Metric": "Prompt-to-feature ratio", "Availability": "Available", "Source": "Visible-prompt heuristic from universal metrics"},
            {"Metric": "Prompt success rate", "Availability": "Available", "Source": "Visible-prompt heuristic from universal metrics"},
            {"Metric": "Requirement adherence score", "Availability": "Available" if not ritm_phase_df.empty else "Not yet loaded", "Source": "Parsed from the RITM PDF"},
            {"Metric": "Scope overreach", "Availability": "Not yet loaded", "Source": "Requires evaluation input"},
            {"Metric": "NASA-TLX", "Availability": "Not yet loaded", "Source": "Requires survey input"},
            {"Metric": "Kiro credits", "Availability": "Available", "Source": "Exact Kiro metering usage"},
            {"Metric": "Copilot exact tokens", "Availability": "Available", "Source": "Actual token counts when present in source logs"},
        ]
    )
    st.dataframe(overview_coverage, width="stretch", hide_index=True)

    st.divider()
    st.write("### Overview KPI Charts")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        fig_quality = px.bar(
            phase_df,
            x="phase",
            y="prompt_success_rate",
            color="phase",
            color_discrete_map=PHASE_COLORS,
            title="Visible Prompt Success Rate by Phase",
            labels={"prompt_success_rate": "Success Rate", "phase": "Phase"},
        )
        fig_quality.update_yaxes(tickformat=".0%")
        st.plotly_chart(_style_figure(fig_quality), width="stretch", key="universal_insights_quality")

    with chart_col2:
        fig_ratio = px.bar(
            phase_df,
            x="phase",
            y="prompt_to_feature_ratio",
            color="phase",
            color_discrete_map=PHASE_COLORS,
            title="Visible Prompts per Heuristic Feature Run",
            labels={"prompt_to_feature_ratio": "Visible Prompts per Run", "phase": "Phase"},
        )
        st.plotly_chart(_style_figure(fig_ratio), width="stretch", key="universal_insights_ratio")

    chart_col3, chart_col4 = st.columns(2)
    with chart_col3:
        output_mix = phase_df.melt(
            id_vars="phase",
            value_vars=[
                "total_lines_written_by_selected_model_agent",
                "number_of_ai_lines_needing_human_revision",
            ],
            var_name="metric",
            value_name="value",
        )
        output_mix["metric"] = output_mix["metric"].replace(
            {
                "total_lines_written_by_selected_model_agent": "AI Lines",
                "number_of_ai_lines_needing_human_revision": "AI Lines Revised",
            }
        )
        fig_output_mix = px.bar(
            output_mix,
            x="phase",
            y="value",
            color="metric",
            barmode="group",
            title="AI Output and Revision Pressure",
            color_discrete_map=SIGNAL_COLORS,
        )
        st.plotly_chart(_style_figure(fig_output_mix), width="stretch", key="universal_insights_output_mix")

    with chart_col4:
        fig_effort = px.bar(
            phase_df.melt(
                id_vars="phase",
                value_vars=["man_hours", "man_days"],
                var_name="metric",
                value_name="value",
            ).replace({"metric": {"man_hours": "Work Hours", "man_days": "Team Man-Days"}}),
            x="phase",
            y="value",
            color="metric",
            barmode="group",
            title="Effort Metrics from the Overview Document",
            color_discrete_map=SIGNAL_COLORS,
        )
        st.plotly_chart(_style_figure(fig_effort), width="stretch", key="universal_insights_effort")

    if not ritm_phase_df.empty:
        st.divider()
        st.write("### RITM Quality Snapshot")
        _render_ritm_graphics(
            ritm_phase_df,
            df_ritm_notes if df_ritm_notes is not None else pd.DataFrame(),
            "universal_insights",
            compact=True,
        )

    user_phase_df = _aggregate_universal_user_phase_metrics(df_universal_metrics)
    if not user_phase_df.empty:
        st.divider()
        st.write("### User-by-Phase Distribution")

        user_col1, user_col2 = st.columns(2)
        with user_col1:
            fig_user_hours = px.bar(
                user_phase_df,
                x="user_id",
                y="man_hours",
                color="phase",
                barmode="group",
                color_discrete_map=PHASE_COLORS,
                title="Work Hours by User and Phase",
                labels={"user_id": "User", "man_hours": "Work Hours"},
            )
            st.plotly_chart(_style_figure(fig_user_hours), width="stretch", key="universal_insights_user_hours")

        with user_col2:
            fig_user_lines = px.bar(
                user_phase_df,
                x="user_id",
                y="total_lines_written_by_selected_model_agent",
                color="phase",
                barmode="group",
                color_discrete_map=PHASE_COLORS,
                title="AI Lines by User and Phase",
                labels={"user_id": "User", "total_lines_written_by_selected_model_agent": "AI Lines"},
            )
            st.plotly_chart(_style_figure(fig_user_lines), width="stretch", key="universal_insights_user_lines")

    st.divider()
    st.write("### Overview KPI Table")
    overview_table = phase_df[
        [
            "phase",
            "contributors",
            "man_hours",
            "man_days",
            "total_prompts",
            "total_visible_prompts",
            "total_redacted_prompts",
            "completed_features",
            "prompt_success_rate",
            "prompt_to_feature_ratio",
            "total_lines_written_by_selected_model_agent",
            "number_of_ai_lines_needing_human_revision",
            "total_lines_written_by_humans",
            "total_metering_usage",
        ]
    ].copy()
    overview_table["prompt_success_rate"] = overview_table["prompt_success_rate"].apply(
        lambda v: f"{v * 100.0:.1f}%" if pd.notna(v) else "N/A"
    )
    overview_table["prompt_to_feature_ratio"] = overview_table["prompt_to_feature_ratio"].apply(
        lambda v: f"{v:.2f}" if pd.notna(v) else "N/A"
    )
    overview_table["total_lines_written_by_humans"] = overview_table["total_lines_written_by_humans"].apply(
        lambda v: f"{v:.0f}" if pd.notna(v) else "N/A"
    )
    overview_table = overview_table.rename(
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
        }
    )
    st.dataframe(overview_table, width="stretch", hide_index=True)

    if not df_universal_all.empty:
        st.divider()
        st.write("### Phase Interaction Distribution")
        prompt_counts = (
            df_universal_all.groupby(["phase", "suspected_user"]).size().reset_index(name="interaction_count")
        )
        prompt_counts = _sort_phase_frame(prompt_counts)
        fig_interactions = px.bar(
            prompt_counts,
            x="suspected_user",
            y="interaction_count",
            color="phase",
            barmode="group",
            color_discrete_map=PHASE_COLORS,
            title="Universal Turn Count by User and Phase",
            labels={"suspected_user": "User", "interaction_count": "Interactions"},
        )
        st.plotly_chart(_style_figure(fig_interactions), width="stretch", key="universal_insights_interactions")



