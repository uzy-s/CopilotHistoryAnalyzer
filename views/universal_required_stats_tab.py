import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import json

from views.shared import *

def _render_phase_metric_chart(
    phase_df: pd.DataFrame,
    metric_col: str,
    title: str,
    y_label: str,
    chart_key: str,
    *,
    percent: bool = False,
    decimals: int = 0,
    empty_message: str | None = None,
) -> None:
    """Render a single slide-ready phase comparison chart for one metric."""
    if phase_df.empty or metric_col not in phase_df.columns:
        st.info(empty_message or f"{title} is not available in the current selection.")
        return

    chart_df = phase_df[["phase", metric_col]].copy()
    chart_df = chart_df[chart_df[metric_col].notna()]
    chart_df = _sort_phase_frame(chart_df)

    if chart_df.empty:
        st.info(empty_message or f"{title} is not available in the current selection.")
        return

    plot_col = metric_col
    if percent:
        plot_col = "_plot_value"
        chart_df[plot_col] = chart_df[metric_col] * 100.0

    fig = px.bar(
        chart_df,
        x="phase",
        y=plot_col,
        color="phase",
        color_discrete_map=PHASE_COLORS,
        title=title,
        labels={"phase": "Phase", plot_col: y_label},
    )
    fig.update_layout(showlegend=False)
    fig.update_traces(
        marker_line_color="white",
        marker_line_width=1.5,
        textposition="outside",
        cliponaxis=False,
    )

    if percent:
        fig.update_traces(texttemplate="%{y:.1f}%")
        axis_max = max(100.0, chart_df[plot_col].max() * 1.18 if chart_df[plot_col].max() > 0 else 100.0)
        fig.update_yaxes(range=[0, axis_max], ticksuffix="%")
    else:
        if decimals <= 0:
            fig.update_traces(texttemplate="%{y:,.0f}")
        elif decimals == 1:
            fig.update_traces(texttemplate="%{y:,.1f}")
        else:
            fig.update_traces(texttemplate="%{y:,.2f}")

        axis_max = chart_df[plot_col].max() * 1.18 if chart_df[plot_col].max() > 0 else 1.0
        fig.update_yaxes(range=[0, axis_max])

    st.plotly_chart(_style_figure(fig), width="stretch", key=chart_key)



def render_universal_required_stats_tab(
    df_universal_metrics: pd.DataFrame,
    df_git: pd.DataFrame | None = None,
    df_ritm_phase: pd.DataFrame | None = None,
    df_ritm_notes: pd.DataFrame | None = None,
) -> None:
    """Render required capstone metrics in a presentation-friendly layout."""
    st.subheader("Required Presentation Metrics")
    st.write(
        "This tab is aligned to the Michelin capstone overview and keeps each shared phase metric on its own chart for slide-ready comparison."
    )

    if df_universal_metrics.empty:
        st.info("No universal metrics loaded. Select universal files in the sidebar.")
        return

    df_universal_metrics = _prepare_universal_metrics_df(df_universal_metrics)
    phase_df = _aggregate_universal_phase_metrics(df_universal_metrics)

    if phase_df.empty:
        st.info("No aggregated phase metrics are available from the selected universal files.")
        return

    ritm_phase_df = _prepare_ritm_phase_df(df_ritm_phase if df_ritm_phase is not None else pd.DataFrame())

    human_lines_available = phase_df["total_lines_written_by_humans"].notna().any()
    revision_metric_available = phase_df["number_of_ai_lines_needing_human_revision"].notna().any()
    total_ai_lines = phase_df["total_lines_written_by_selected_model_agent"].fillna(0).sum()
    total_tracking_human_written_lines = phase_df["tracking_human_written_lines"].fillna(0).sum()
    total_tracking_human_edit_lines = phase_df["tracking_human_edit_lines"].fillna(0).sum()
    total_tracking_ai_removed_lines = phase_df["tracking_ai_removed_lines"].fillna(0).sum()
    tracked_user_phase_pairs = (
        int(df_universal_metrics["tracking_sheet_count"].fillna(0).gt(0).sum())
        if "tracking_sheet_count" in df_universal_metrics.columns
        else 0
    )
    total_copilot_applied_edit_lines = phase_df["total_copilot_applied_edit_lines"].fillna(0).sum()
    total_copilot_applied_edit_calls = phase_df["total_copilot_applied_edit_calls"].fillna(0).sum()
    total_copilot_edited_file_events = phase_df["total_copilot_edited_file_events"].fillna(0).sum()

    requirement_rows = [
        {
            "Metric": "Total lines written by selected model / agent",
            "Capstone Scope": "Shared (Phase 1-3)",
            "Availability": "Available",
            "Presentation Note": "Charted below as a single phase comparison.",
        },
        {
            "Metric": "Total work hours",
            "Capstone Scope": "Shared (Phase 1-3)",
            "Availability": "Available",
            "Presentation Note": "Charted below as a single phase comparison.",
        },
        {
            "Metric": "Team man-days",
            "Capstone Scope": "Shared (Phase 1-3)",
            "Availability": "Available",
            "Presentation Note": "Charted below as a single phase comparison.",
        },
        {
            "Metric": "Prompt-to-feature ratio",
            "Capstone Scope": "Shared (Phase 1-3)",
            "Availability": "Available",
            "Presentation Note": "Weighted from visible prompts for cleaner cross-phase comparison.",
        },
        {
            "Metric": "Prompt success rate",
            "Capstone Scope": "Shared (Phase 1-3)",
            "Availability": "Available",
            "Presentation Note": "Weighted from visible prompts to avoid redacted-log distortion.",
        },
        {
            "Metric": "Total lines written by humans",
            "Capstone Scope": "Phase 1 only",
            "Availability": (
                "Available (tracking workbook supplement)"
                if tracked_user_phase_pairs > 0
                else "Available"
            )
            if human_lines_available
            else "Not loaded in current universal exports",
            "Presentation Note": (
                "Included in the phase summary table and backfilled from tracking workbooks where the universal export was null."
                if human_lines_available
                else "Current universal JSON files preserve this field as null."
            ),
        },
        {
            "Metric": "Number of AI lines needing human revision",
            "Capstone Scope": "Phase 1 only in the brief",
            "Availability": "Available (supplemental heuristic view)" if revision_metric_available else "Not loaded",
            "Presentation Note": "Shown below as a supplemental chart for revision pressure.",
        },
        {
            "Metric": "Merge request rejections and causes",
            "Capstone Scope": "All phases",
            "Availability": "Not yet loaded",
            "Presentation Note": "Requires review / merge metadata outside the current universal files.",
        },
        {
            "Metric": "NASA-TLX",
            "Capstone Scope": "Post-phase survey",
            "Availability": "Not yet loaded",
            "Presentation Note": "Requires survey input rather than chat or git data.",
        },
        {
            "Metric": "Requirement adherence score",
            "Capstone Scope": "All phases",
            "Availability": "Available" if not ritm_phase_df.empty else "Not yet loaded",
            "Presentation Note": "Parsed from the RITM PDF." if not ritm_phase_df.empty else "Requires RITM scoring input.",
        },
        {
            "Metric": "Scope overreach",
            "Capstone Scope": "All phases",
            "Availability": "Not yet loaded",
            "Presentation Note": "Requires evaluation input against the scoped requirement set.",
        },
    ]
    loaded_required_items = sum(1 for row in requirement_rows if row["Availability"].startswith("Available"))

    st.write("### Requirement Coverage")
    overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
    overview_col1.metric("Phases Loaded", f"{phase_df['phase'].nunique():.0f}")
    overview_col2.metric("Contributors", f"{df_universal_metrics['user_id'].nunique():.0f}")
    overview_col3.metric("Shared Charts Ready", "5/5")
    overview_col4.metric("Required Items Loaded", f"{loaded_required_items}/{len(requirement_rows)}")
    st.dataframe(pd.DataFrame(requirement_rows), width="stretch", hide_index=True)

    readiness_col1, readiness_col2, readiness_col3 = st.columns(3)
    readiness_col1.metric("Measured AI Output Lines", f"{total_ai_lines:.0f}")
    readiness_col2.metric("Copilot Applied Edit Lines", f"{total_copilot_applied_edit_lines:.0f}")
    readiness_col3.metric("Copilot Edit Calls", f"{total_copilot_applied_edit_calls:.0f}")
    if tracked_user_phase_pairs > 0:
        tracking_col1, tracking_col2, tracking_col3, tracking_col4 = st.columns(4)
        tracking_col1.metric("Tracked User-Phase Pairs", f"{tracked_user_phase_pairs:.0f}")
        tracking_col2.metric("Workbook Human Lines", f"{total_tracking_human_written_lines:.0f}")
        tracking_col3.metric("Workbook Human Edit Lines", f"{total_tracking_human_edit_lines:.0f}")
        tracking_col4.metric("Workbook AI Lines Removed", f"{total_tracking_ai_removed_lines:.0f}")

    st.caption(
        "Shared capstone metrics are charted one-per-figure below so each slide can lift a single graph without combining unrelated measures. "
        "Measured AI output uses structured applied editor/tool edits when preserved by the logs, otherwise fenced assistant code."
    )

    st.divider()
    st.write("### Shared Phase Metrics")
    st.caption(
        "Prompt success rate and prompt-to-feature ratio are derived from visible natural prompts, which keeps redacted logs from artificially skewing the phase comparison."
    )

    shared_row_1_left, shared_row_1_right = st.columns(2)
    with shared_row_1_left:
        _render_phase_metric_chart(
            phase_df,
            "total_lines_written_by_selected_model_agent",
            "Measured AI Output Lines by Phase",
            "Measured AI Output Lines",
            "required_shared_ai_lines",
            decimals=0,
        )
    with shared_row_1_right:
        _render_phase_metric_chart(
            phase_df,
            "man_hours",
            "Work Hours by Phase",
            "Work Hours",
            "required_shared_work_hours",
            decimals=2,
        )

    shared_row_2_left, shared_row_2_right = st.columns(2)
    with shared_row_2_left:
        _render_phase_metric_chart(
            phase_df,
            "man_days",
            "Team Man-Days by Phase",
            "Team Man-Days",
            "required_shared_man_days",
            decimals=0,
        )
    with shared_row_2_right:
        _render_phase_metric_chart(
            phase_df,
            "prompt_to_feature_ratio",
            "Prompt-to-Feature Ratio by Phase",
            "Visible Prompts per Run",
            "required_shared_prompt_to_feature",
            decimals=2,
        )

    _render_phase_metric_chart(
        phase_df,
        "prompt_success_rate",
        "Prompt Success Rate by Phase",
        "Prompt Success Rate (%)",
        "required_shared_prompt_success",
        percent=True,
    )

    st.write("### Required Metrics Graphics")
    graphic_left, graphic_right = st.columns(2)
    with graphic_left:
        required_output_mix_df = phase_df.melt(
            id_vars="phase",
            value_vars=["total_assistant_fallback_code_lines", "total_tool_code_lines"],
            var_name="metric",
            value_name="value",
        )
        required_output_mix_df["metric"] = required_output_mix_df["metric"].replace(
            {
                "total_assistant_fallback_code_lines": "Assistant Fallback Code Lines",
                "total_tool_code_lines": "Structured Edit Lines",
            }
        )
        fig_required_output_mix = px.bar(
            required_output_mix_df,
            x="phase",
            y="value",
            color="metric",
            barmode="stack",
            title="Measured Output Composition by Phase",
            color_discrete_map=SIGNAL_COLORS,
        )
        st.plotly_chart(
            _style_figure(fig_required_output_mix),
            width="stretch",
            key="required_output_mix_phase",
        )

    with graphic_right:
        fig_required_copilot = px.bar(
            phase_df,
            x="phase",
            y="total_copilot_applied_edit_lines",
            color="phase",
            color_discrete_map=PHASE_COLORS,
            title="Copilot Applied Edit Lines by Phase",
            labels={
                "phase": "Phase",
                "total_copilot_applied_edit_lines": "Applied Edit Lines",
            },
        )
        st.plotly_chart(
            _style_figure(fig_required_copilot),
            width="stretch",
            key="required_copilot_activity_phase",
        )
        required_call_parts = []
        for _, row in phase_df.iterrows():
            required_call_parts.append(
                f"{row['phase']}: {int(row['total_copilot_applied_edit_calls'] or 0)} edit calls / "
                f"{int(row['total_copilot_edited_file_events'] or 0)} edited-file events"
        )
        st.caption("Support counts for the bar above: " + "; ".join(required_call_parts))

    if tracked_user_phase_pairs > 0:
        st.write("### Workbook-Tracked Edit Activity")
        workbook_edit_left, workbook_edit_right = st.columns(2)
        with workbook_edit_left:
            _render_phase_metric_chart(
                phase_df,
                "tracking_human_edit_lines",
                "Workbook-Tracked Human Edit Lines by Phase",
                "Human Edit Lines",
                "required_tracking_human_edit_lines",
                decimals=0,
            )
        with workbook_edit_right:
            _render_phase_metric_chart(
                phase_df,
                "tracking_ai_removed_lines",
                "Workbook-Tracked AI Lines Removed by Phase",
                "AI Lines Removed",
                "required_tracking_ai_removed_lines",
                decimals=0,
            )
        st.caption(
            "These workbook-only metrics capture manual cleanup and edit pressure that the chat exports do not always preserve directly."
        )

    st.divider()
    st.write("### Phase-Specific and Supplemental Metrics")
    supplemental_left, supplemental_right = st.columns(2)
    with supplemental_left:
        _render_phase_metric_chart(
            phase_df,
            "number_of_ai_lines_needing_human_revision",
            "AI Lines Needing Human Revision",
            "AI Lines Revised",
            "required_supplemental_revision_lines",
            decimals=0,
            empty_message="The current selection does not include revision-line metrics.",
        )
        st.caption("The capstone brief explicitly calls this out for Phase 1; the universal pipeline currently exposes it as a supplemental cross-phase heuristic.")

    with supplemental_right:
        if human_lines_available:
            _render_phase_metric_chart(
                phase_df,
                "total_lines_written_by_humans",
                "Human Lines by Phase",
                "Human Lines",
                "required_supplemental_human_lines",
                decimals=0,
            )
            st.caption("This metric is a Phase 1 requirement in the capstone brief.")
        else:
            st.info(
                "Phase 1 human-authored line totals are part of the capstone brief, but the current universal exports do not yet populate that field."
            )
            st.caption(
                "If you add a human-line source later, this tab is already structured to surface it beside the other required metrics."
            )

    if not ritm_phase_df.empty:
        st.divider()
        st.write("### RITM Requirement Adherence")
        st.caption(
            "These charts come directly from the Michelin RITM PDF and provide the presentable quality/adherence view that complements the prompt and delivery metrics."
        )
        _render_ritm_graphics(
            ritm_phase_df,
            df_ritm_notes if df_ritm_notes is not None else pd.DataFrame(),
            "required_metrics",
            compact=False,
        )

    st.divider()
    st.write("### Presentation Summary Table")
    summary_df = phase_df[
        [
            "phase",
            "total_lines_written_by_selected_model_agent",
            "total_copilot_applied_edit_lines",
            "total_copilot_applied_edit_calls",
            "man_hours",
            "man_days",
            "prompt_to_feature_ratio",
            "prompt_success_rate",
            "number_of_ai_lines_needing_human_revision",
            "total_lines_written_by_humans",
            "tracking_human_edit_lines",
            "tracking_ai_removed_lines",
        ]
    ].copy()
    summary_df = summary_df.rename(
        columns={
            "phase": "Phase",
            "total_lines_written_by_selected_model_agent": "Measured AI Output Lines",
            "total_copilot_applied_edit_lines": "Copilot Applied Edit Lines",
            "total_copilot_applied_edit_calls": "Copilot Edit Calls",
            "man_hours": "Work Hours",
            "man_days": "Team Man-Days",
            "prompt_to_feature_ratio": "Prompts per Run",
            "prompt_success_rate": "Prompt Success Rate",
            "number_of_ai_lines_needing_human_revision": "AI Lines Revised",
            "total_lines_written_by_humans": "Human Lines",
            "tracking_human_edit_lines": "Workbook Human Edit Lines",
            "tracking_ai_removed_lines": "Workbook AI Lines Removed",
        }
    )
    summary_df["Measured AI Output Lines"] = summary_df["Measured AI Output Lines"].apply(
        lambda v: f"{v:.0f}" if pd.notna(v) else "N/A"
    )
    summary_df["Copilot Applied Edit Lines"] = summary_df["Copilot Applied Edit Lines"].apply(
        lambda v: f"{v:.0f}" if pd.notna(v) else "0"
    )
    summary_df["Copilot Edit Calls"] = summary_df["Copilot Edit Calls"].apply(
        lambda v: f"{v:.0f}" if pd.notna(v) else "0"
    )
    summary_df["Work Hours"] = summary_df["Work Hours"].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "N/A")
    summary_df["Team Man-Days"] = summary_df["Team Man-Days"].apply(lambda v: f"{v:.0f}" if pd.notna(v) else "N/A")
    summary_df["Prompts per Run"] = summary_df["Prompts per Run"].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "N/A")
    summary_df["Prompt Success Rate"] = summary_df["Prompt Success Rate"].apply(
        lambda v: f"{v * 100.0:.1f}%" if pd.notna(v) else "N/A"
    )
    summary_df["AI Lines Revised"] = summary_df["AI Lines Revised"].apply(
        lambda v: f"{v:.0f}" if pd.notna(v) else "N/A"
    )
    summary_df["Human Lines"] = summary_df["Human Lines"].apply(
        lambda v: f"{v:.0f}" if pd.notna(v) else "Not loaded"
    )
    summary_df["Workbook Human Edit Lines"] = summary_df["Workbook Human Edit Lines"].apply(
        lambda v: f"{v:.0f}" if pd.notna(v) else "0"
    )
    summary_df["Workbook AI Lines Removed"] = summary_df["Workbook AI Lines Removed"].apply(
        lambda v: f"{v:.0f}" if pd.notna(v) else "0"
    )
    if not ritm_phase_df.empty:
        summary_df = summary_df.merge(
            ritm_phase_df[["phase", "requirement_adherence_percent", "documented_issue_count"]],
            left_on="Phase",
            right_on="phase",
            how="left",
        ).drop(columns="phase")
        summary_df["requirement_adherence_percent"] = summary_df["requirement_adherence_percent"].apply(
            lambda v: f"{v:.2f}%" if pd.notna(v) else "N/A"
        )
        summary_df["documented_issue_count"] = summary_df["documented_issue_count"].apply(
            lambda v: f"{v:.0f}" if pd.notna(v) else "0"
        )
        summary_df = summary_df.rename(
            columns={
                "requirement_adherence_percent": "RITM Adherence",
                "documented_issue_count": "RITM Issue Notes",
            }
        )
    st.dataframe(summary_df, width="stretch", hide_index=True)

    st.divider()
    st.write("### Optional Platform Coverage")

    credits_total = df_universal_metrics["total_metering_usage"].fillna(0).sum()
    actual_tokens_total = (
        df_universal_metrics["total_actual_prompt_tokens"].fillna(0).sum()
        + df_universal_metrics["total_actual_completion_tokens"].fillna(0).sum()
    )
    estimated_tokens_total = (
        df_universal_metrics["total_estimated_prompt_tokens"].fillna(0).sum()
        + df_universal_metrics["total_estimated_completion_tokens"].fillna(0).sum()
    )
    avg_context_usage = df_universal_metrics["average_context_usage_percentage"].dropna()
    total_assistant_code = df_universal_metrics["total_assistant_code_lines"].fillna(0).sum()
    total_tool_code = df_universal_metrics["total_tool_code_lines"].fillna(0).sum()
    structured_output_share = _safe_divide(total_tool_code, total_ai_lines) or 0.0
    metering_unit_values = [v for v in df_universal_metrics["metering_unit"].dropna().tolist() if str(v).strip()]
    metering_unit = metering_unit_values[0] if metering_unit_values else "credit"

    optional_col1, optional_col2, optional_col3, optional_col4 = st.columns(4)
    optional_col1.metric(
        "Kiro Credits",
        f"{credits_total:.3f}",
        help=f"Exact metering total from Kiro logs in {metering_unit}s.",
    )
    optional_col2.metric(
        "Actual Tokens",
        f"{actual_tokens_total:.0f}",
        help="Exact token totals when the source logs expose them directly.",
    )
    optional_col3.metric(
        "Estimated Tokens",
        f"{estimated_tokens_total:.0f}",
        help="Estimated totals used for sources like Kiro when exact tokens are unavailable.",
    )
    optional_col4.metric(
        "Avg Context Usage",
        f"{(avg_context_usage.mean() if len(avg_context_usage) > 0 else 0.0):.2f}%",
        help="Average context usage percentage across selected universal files.",
    )

    optional_col5, optional_col6, optional_col7, optional_col8 = st.columns(4)
    optional_col5.metric(
        "Assistant Code Lines",
        f"{total_assistant_code:.0f}",
        help="Code lines detected in assistant fenced/code text.",
    )
    optional_col6.metric(
        "Structured Edit Lines",
        f"{total_tool_code:.0f}",
        help="Structured edit lines reconstructed from Copilot applied editor changes and Kiro tool edits.",
    )
    optional_col7.metric(
        "Copilot Applied Edit Lines",
        f"{total_copilot_applied_edit_lines:.0f}",
        help="Applied editor-change lines reconstructed from Copilot edit tools.",
    )
    optional_col8.metric(
        "Structured Output Share",
        f"{structured_output_share * 100.0:.1f}%",
        help="Share of measured AI output coming from structured edits rather than fenced assistant code.",
    )

    st.divider()
    st.write("### Appendix: Per User / Phase Breakdown")
    display_df = df_universal_metrics[
        [
            "user_id",
            "phase",
            "man_hours",
            "man_days",
            "total_prompts",
            "total_visible_prompts",
            "total_redacted_prompts",
            "prompt_visibility_rate",
            "prompt_success_ratio",
            "prompt_success_rate",
            "prompt_to_feature_ratio",
            "total_lines_written_by_selected_model_agent",
            "total_copilot_applied_edit_lines",
            "total_copilot_applied_edit_calls",
            "total_copilot_edited_file_events",
            "number_of_ai_lines_needing_human_revision",
            "total_lines_written_by_humans",
            "tracking_human_written_lines",
            "tracking_human_edit_lines",
            "tracking_human_deleted_lines",
            "tracking_ai_generated_lines",
            "tracking_ai_removed_lines",
            "tracking_ai_edited_lines",
            "tracking_missing_features",
            "tracking_sheet_count",
            "tracking_sheet_names",
            "total_actual_prompt_tokens",
            "total_actual_completion_tokens",
            "total_estimated_prompt_tokens",
            "total_estimated_completion_tokens",
            "total_metering_usage",
            "metering_unit",
            "average_context_usage_percentage",
            "file_path",
        ]
    ].copy()

    for pct_col in ["prompt_visibility_rate", "prompt_success_ratio", "prompt_success_rate"]:
        display_df[pct_col] = display_df[pct_col].apply(
            lambda v: f"{v * 100.0:.1f}%" if pd.notna(v) else "N/A"
        )

    display_df = display_df.rename(
        columns={
            "user_id": "User",
            "phase": "Phase",
            "man_hours": "Work Hours",
            "man_days": "Team Man-Days",
            "total_prompts": "Natural Prompts",
            "total_visible_prompts": "Visible Prompts",
            "total_redacted_prompts": "Redacted Prompts",
            "prompt_visibility_rate": "Prompt Visibility",
            "prompt_success_ratio": "Prompt Success Ratio",
            "prompt_success_rate": "Prompt Success Rate",
            "prompt_to_feature_ratio": "Visible Prompts per Run",
            "total_lines_written_by_selected_model_agent": "Measured AI Output Lines",
            "total_copilot_applied_edit_lines": "Copilot Applied Edit Lines",
            "total_copilot_applied_edit_calls": "Copilot Edit Calls",
            "total_copilot_edited_file_events": "Copilot Edited File Events",
            "number_of_ai_lines_needing_human_revision": "AI Lines Revised",
            "total_lines_written_by_humans": "Human Lines",
            "tracking_human_written_lines": "Workbook Human Lines",
            "tracking_human_edit_lines": "Workbook Human Edit Lines",
            "tracking_human_deleted_lines": "Workbook Human Deleted Lines",
            "tracking_ai_generated_lines": "Workbook AI Lines",
            "tracking_ai_removed_lines": "Workbook AI Lines Removed",
            "tracking_ai_edited_lines": "Workbook AI Lines Edited",
            "tracking_missing_features": "Workbook Missing Features",
            "tracking_sheet_count": "Workbook Sheet Count",
            "tracking_sheet_names": "Workbook Sheets",
            "total_actual_prompt_tokens": "Actual Prompt Tokens",
            "total_actual_completion_tokens": "Actual Completion Tokens",
            "total_estimated_prompt_tokens": "Estimated Prompt Tokens",
            "total_estimated_completion_tokens": "Estimated Completion Tokens",
            "total_metering_usage": "Kiro Credits",
            "metering_unit": "Credit Unit",
            "average_context_usage_percentage": "Avg Context Usage",
            "file_path": "Source File",
        }
    )
    st.dataframe(display_df, width="stretch", hide_index=True)

    st.divider()
    st.write("### Appendix: Token Method Coverage")

    coverage_rows: list[dict[str, str]] = []
    for _, row in df_universal_metrics.iterrows():
        coverage_rows.append(
            {
                "user_id": str(row.get("user_id") or "Unknown"),
                "phase": str(row.get("phase") or "Unknown"),
                "token_count_methods": str(row.get("token_count_methods") or "{}"),
                "tool_call_counts": str(row.get("tool_call_counts") or "{}"),
            }
        )

    st.dataframe(pd.DataFrame(coverage_rows), width="stretch", hide_index=True)

    if df_git is not None and not df_git.empty:
        st.divider()
        st.write("### Appendix: Git Repository Overlay")

        git_col1, git_col2, git_col3 = st.columns(3)
        git_col1.metric("Git Commits", f"{len(df_git):.0f}")
        git_col2.metric("Git Insertions", f"{df_git['insertions'].fillna(0).sum():.0f}")
        git_col3.metric("Git Deletions", f"{df_git['deletions'].fillna(0).sum():.0f}")

        git_daily = df_git.copy()
        git_daily["date"] = git_daily["timestamp"].dt.date
        git_daily_counts = git_daily.groupby("date").size().reset_index(name="commit_count")
        fig_git = px.line(
            git_daily_counts,
            x="date",
            y="commit_count",
            markers=True,
            title="Git Commit History Over Time",
        )
        st.plotly_chart(_style_figure(fig_git), width="stretch")


