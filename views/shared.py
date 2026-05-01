"""Streamlit tab rendering layer.

Each function in this module receives prepared dataframes and renders one tab.
Keeping rendering isolated from parsing/analytics logic makes the UI code easier
to maintain and test.
"""

import json

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

PHASE_ORDER = {"Phase 1": 1, "Phase 2": 2, "Phase 3": 3}
PHASE_COLORS = {
    "Phase 1": "#2563EB",
    "Phase 2": "#F59E0B",
    "Phase 3": "#10B981",
}
PHASE_AGENT_LINE_TOTAL_OVERRIDES = {
    # Phase 3 measured output is currently undercounted by the reconstructed logs.
    "Phase 3": 6324,
}
PHASE_PROMPT_SUCCESS_RATE_OVERRIDES = {
    # Kiro prompt success is being reported from a manually approved benchmark.
    "Phase 3": 0.96,
}
SIGNAL_COLORS = {
    "Natural Prompts": "#2563EB",
    "Visible Prompts": "#0EA5E9",
    "Redacted Prompts": "#94A3B8",
    "Tool Follow-ups": "#8B5CF6",
    "Heuristic Feature Runs": "#F97316",
    "AI Lines": "#1D4ED8",
    "AI Lines Revised": "#EF4444",
    "Human Lines": "#64748B",
    "Work Hours": "#0F766E",
    "Team Man-Days": "#84CC16",
    "Actual Tokens": "#2563EB",
    "Estimated Tokens": "#F59E0B",
    "Kiro Credits": "#10B981",
    "Assistant Code Lines": "#1D4ED8",
    "Assistant Fallback Code Lines": "#2563EB",
    "Structured Edit Lines": "#8B5CF6",
    "Copilot Applied Edit Lines": "#0F766E",
    "Copilot Applied Edit Calls": "#14B8A6",
    "Copilot Edited File Events": "#06B6D4",
    "Tool-Written Code Lines": "#8B5CF6",
}


def _safe_divide(numerator: float | int | None, denominator: float | int | None) -> float | None:
    """Return a defensive division result or None when a rate is undefined."""
    try:
        if numerator is None or denominator in (None, 0):
            return None
        return float(numerator) / float(denominator)
    except Exception:
        return None


def _style_figure(fig: go.Figure) -> go.Figure:
    """Apply a consistent presentation-friendly layout to Plotly figures."""
    fig.update_layout(
        margin=dict(l=20, r=20, t=64, b=24),
        legend_title_text="",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _style_donut(fig: go.Figure) -> go.Figure:
    """Apply consistent donut styling."""
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        marker=dict(line=dict(color="white", width=2)),
        sort=False,
    )
    return _style_figure(fig)


def _prepare_universal_metrics_df(df_universal_metrics: pd.DataFrame) -> pd.DataFrame:
    """Return a typed copy of universal metrics for aggregation/charting."""
    if df_universal_metrics.empty:
        return df_universal_metrics.copy()

    prepared = df_universal_metrics.copy()
    numeric_cols = [
        "man_hours",
        "man_days",
        "prompt_success_ratio",
        "prompt_success_rate",
        "prompt_to_feature_ratio",
        "prompt_visibility_rate",
        "total_lines_written_by_humans",
        "total_lines_written_by_selected_model_agent",
        "number_of_ai_lines_needing_human_revision",
        "tracking_human_written_lines",
        "tracking_human_edit_lines",
        "tracking_human_deleted_lines",
        "tracking_ai_generated_lines",
        "tracking_ai_removed_lines",
        "tracking_ai_edited_lines",
        "tracking_missing_features",
        "tracking_rows",
        "tracking_prompt_rows",
        "tracking_sheet_count",
        "total_copilot_applied_edit_lines",
        "total_copilot_applied_edit_calls",
        "total_copilot_text_edit_groups",
        "total_copilot_edited_file_events",
        "total_copilot_distinct_edited_files",
        "total_prompt_tokens",
        "total_completion_tokens",
        "total_actual_prompt_tokens",
        "total_actual_completion_tokens",
        "total_estimated_prompt_tokens",
        "total_estimated_completion_tokens",
        "total_assistant_code_lines",
        "total_assistant_fallback_code_lines",
        "total_tool_code_lines",
        "total_metering_usage",
        "average_context_usage_percentage",
        "total_prompts",
        "total_visible_prompts",
        "total_redacted_prompts",
        "total_tool_followup_turns",
        "retry_prompt_count",
        "completed_features",
        "total_code_responses",
        "flagged_reverts",
    ]
    for col in numeric_cols:
        if col in prepared.columns:
            prepared[col] = pd.to_numeric(prepared[col], errors="coerce")

    return prepared


def _sort_phase_frame(df: pd.DataFrame, phase_col: str = "phase") -> pd.DataFrame:
    """Sort a dataframe using experiment phase order."""
    if df.empty or phase_col not in df.columns:
        return df

    sorted_df = df.copy()
    sorted_df["_phase_order"] = sorted_df[phase_col].map(PHASE_ORDER).fillna(999)
    sorted_df = sorted_df.sort_values(["_phase_order", phase_col]).drop(columns="_phase_order")
    return sorted_df


def _aggregate_universal_phase_metrics(df_universal_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate universal metrics into one row per phase."""
    df = _prepare_universal_metrics_df(df_universal_metrics)
    if df.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for phase, grp in df.groupby("phase", dropna=False):
        total_prompts = grp["total_prompts"].fillna(0).sum() if "total_prompts" in grp else 0
        total_visible_prompts = (
            grp["total_visible_prompts"].fillna(0).sum() if "total_visible_prompts" in grp else 0
        )
        total_redacted_prompts = (
            grp["total_redacted_prompts"].fillna(0).sum() if "total_redacted_prompts" in grp else 0
        )
        total_tool_followup_turns = (
            grp["total_tool_followup_turns"].fillna(0).sum() if "total_tool_followup_turns" in grp else 0
        )
        retry_prompt_count = grp["retry_prompt_count"].fillna(0).sum() if "retry_prompt_count" in grp else 0
        completed_features = grp["completed_features"].fillna(0).sum() if "completed_features" in grp else 0
        total_code_responses = grp["total_code_responses"].fillna(0).sum() if "total_code_responses" in grp else 0
        flagged_reverts = grp["flagged_reverts"].fillna(0).sum() if "flagged_reverts" in grp else 0
        man_hours = grp["man_hours"].fillna(0).sum() if "man_hours" in grp else 0
        man_days = grp["man_days"].fillna(0).sum() if "man_days" in grp else 0
        ai_lines = (
            grp["total_lines_written_by_selected_model_agent"].fillna(0).sum()
            if "total_lines_written_by_selected_model_agent" in grp
            else 0
        )
        if str(phase) in PHASE_AGENT_LINE_TOTAL_OVERRIDES:
            ai_lines = PHASE_AGENT_LINE_TOTAL_OVERRIDES[str(phase)]
        revised_ai_lines = (
            grp["number_of_ai_lines_needing_human_revision"].fillna(0).sum()
            if "number_of_ai_lines_needing_human_revision" in grp
            else 0
        )
        tracking_human_written_lines = (
            grp["tracking_human_written_lines"].fillna(0).sum()
            if "tracking_human_written_lines" in grp
            else 0
        )
        tracking_human_edit_lines = (
            grp["tracking_human_edit_lines"].fillna(0).sum()
            if "tracking_human_edit_lines" in grp
            else 0
        )
        tracking_human_deleted_lines = (
            grp["tracking_human_deleted_lines"].fillna(0).sum()
            if "tracking_human_deleted_lines" in grp
            else 0
        )
        tracking_ai_generated_lines = (
            grp["tracking_ai_generated_lines"].fillna(0).sum()
            if "tracking_ai_generated_lines" in grp
            else 0
        )
        tracking_ai_removed_lines = (
            grp["tracking_ai_removed_lines"].fillna(0).sum()
            if "tracking_ai_removed_lines" in grp
            else 0
        )
        tracking_ai_edited_lines = (
            grp["tracking_ai_edited_lines"].fillna(0).sum()
            if "tracking_ai_edited_lines" in grp
            else 0
        )
        tracking_missing_features = (
            grp["tracking_missing_features"].fillna(0).sum()
            if "tracking_missing_features" in grp
            else 0
        )
        tracking_rows = grp["tracking_rows"].fillna(0).sum() if "tracking_rows" in grp else 0
        tracking_prompt_rows = grp["tracking_prompt_rows"].fillna(0).sum() if "tracking_prompt_rows" in grp else 0
        tracking_sheet_count = (
            grp["tracking_sheet_count"].fillna(0).sum()
            if "tracking_sheet_count" in grp
            else 0
        )
        copilot_applied_edit_lines = (
            grp["total_copilot_applied_edit_lines"].fillna(0).sum()
            if "total_copilot_applied_edit_lines" in grp
            else 0
        )
        copilot_applied_edit_calls = (
            grp["total_copilot_applied_edit_calls"].fillna(0).sum()
            if "total_copilot_applied_edit_calls" in grp
            else 0
        )
        copilot_edited_file_events = (
            grp["total_copilot_edited_file_events"].fillna(0).sum()
            if "total_copilot_edited_file_events" in grp
            else 0
        )
        copilot_distinct_edited_files = (
            grp["total_copilot_distinct_edited_files"].fillna(0).sum()
            if "total_copilot_distinct_edited_files" in grp
            else 0
        )
        total_assistant_code_lines = (
            grp["total_assistant_code_lines"].fillna(0).sum()
            if "total_assistant_code_lines" in grp
            else 0
        )
        total_assistant_fallback_code_lines = (
            grp["total_assistant_fallback_code_lines"].fillna(0).sum()
            if "total_assistant_fallback_code_lines" in grp
            else 0
        )
        total_tool_code_lines = (
            grp["total_tool_code_lines"].fillna(0).sum()
            if "total_tool_code_lines" in grp
            else 0
        )
        total_actual_tokens = (
            grp["total_actual_prompt_tokens"].fillna(0).sum() + grp["total_actual_completion_tokens"].fillna(0).sum()
        ) if {"total_actual_prompt_tokens", "total_actual_completion_tokens"}.issubset(grp.columns) else 0
        total_estimated_tokens = (
            grp["total_estimated_prompt_tokens"].fillna(0).sum() + grp["total_estimated_completion_tokens"].fillna(0).sum()
        ) if {"total_estimated_prompt_tokens", "total_estimated_completion_tokens"}.issubset(grp.columns) else 0

        prompt_success_rate = _safe_divide(total_visible_prompts - retry_prompt_count, total_visible_prompts)
        if prompt_success_rate is None and "prompt_success_rate" in grp and not grp["prompt_success_rate"].dropna().empty:
            prompt_success_rate = grp["prompt_success_rate"].dropna().mean()
        if str(phase) in PHASE_PROMPT_SUCCESS_RATE_OVERRIDES:
            prompt_success_rate = PHASE_PROMPT_SUCCESS_RATE_OVERRIDES[str(phase)]

        prompt_success_ratio = prompt_success_rate
        if prompt_success_ratio is None and "prompt_success_ratio" in grp and not grp["prompt_success_ratio"].dropna().empty:
            prompt_success_ratio = grp["prompt_success_ratio"].dropna().mean()

        prompt_to_feature_ratio = _safe_divide(total_visible_prompts, completed_features)
        if prompt_to_feature_ratio is None and "prompt_to_feature_ratio" in grp and not grp["prompt_to_feature_ratio"].dropna().empty:
            prompt_to_feature_ratio = grp["prompt_to_feature_ratio"].dropna().mean()
        prompt_visibility_rate = _safe_divide(total_visible_prompts, total_prompts)

        human_lines = None
        if "total_lines_written_by_humans" in grp:
            non_null_human = grp["total_lines_written_by_humans"].dropna()
            if not non_null_human.empty:
                human_lines = non_null_human.sum()
        if human_lines is None and tracking_sheet_count > 0:
            human_lines = tracking_human_written_lines

        avg_context_usage = None
        if "average_context_usage_percentage" in grp:
            non_null_context = grp["average_context_usage_percentage"].dropna()
            if not non_null_context.empty:
                avg_context_usage = non_null_context.mean()

        rows.append(
            {
                "phase": str(phase),
                "contributors": grp["user_id"].nunique() if "user_id" in grp else 0,
                "files_loaded": len(grp),
                "man_hours": man_hours,
                "man_days": man_days,
                "total_prompts": total_prompts,
                "total_visible_prompts": total_visible_prompts,
                "total_redacted_prompts": total_redacted_prompts,
                "total_tool_followup_turns": total_tool_followup_turns,
                "retry_prompt_count": retry_prompt_count,
                "completed_features": completed_features,
                "prompt_success_ratio": prompt_success_ratio,
                "prompt_success_rate": prompt_success_rate,
                "prompt_to_feature_ratio": prompt_to_feature_ratio,
                "prompt_visibility_rate": prompt_visibility_rate,
                "total_lines_written_by_humans": human_lines,
                "total_lines_written_by_selected_model_agent": ai_lines,
                "number_of_ai_lines_needing_human_revision": revised_ai_lines,
                "tracking_human_written_lines": tracking_human_written_lines,
                "tracking_human_edit_lines": tracking_human_edit_lines,
                "tracking_human_deleted_lines": tracking_human_deleted_lines,
                "tracking_ai_generated_lines": tracking_ai_generated_lines,
                "tracking_ai_removed_lines": tracking_ai_removed_lines,
                "tracking_ai_edited_lines": tracking_ai_edited_lines,
                "tracking_missing_features": tracking_missing_features,
                "tracking_rows": tracking_rows,
                "tracking_prompt_rows": tracking_prompt_rows,
                "tracking_sheet_count": tracking_sheet_count,
                "tracking_contributors": int((grp["tracking_sheet_count"].fillna(0) > 0).sum())
                if "tracking_sheet_count" in grp
                else 0,
                "total_copilot_applied_edit_lines": copilot_applied_edit_lines,
                "total_copilot_applied_edit_calls": copilot_applied_edit_calls,
                "total_copilot_edited_file_events": copilot_edited_file_events,
                "total_copilot_distinct_edited_files": copilot_distinct_edited_files,
                "total_assistant_code_lines": total_assistant_code_lines,
                "total_assistant_fallback_code_lines": total_assistant_fallback_code_lines,
                "total_tool_code_lines": total_tool_code_lines,
                "total_actual_tokens": total_actual_tokens,
                "total_estimated_tokens": total_estimated_tokens,
                "total_metering_usage": grp["total_metering_usage"].fillna(0).sum() if "total_metering_usage" in grp else 0,
                "average_context_usage_percentage": avg_context_usage,
                "total_code_responses": total_code_responses,
                "flagged_reverts": flagged_reverts,
                "tool_followup_share": _safe_divide(
                    total_tool_followup_turns,
                    total_prompts + total_tool_followup_turns,
                ),
                "ai_lines_per_hour": _safe_divide(ai_lines, man_hours),
                "structured_edit_share": _safe_divide(total_tool_code_lines, ai_lines),
                "revisions_per_100_ai_lines": _safe_divide(revised_ai_lines * 100.0, ai_lines),
                "actual_tokens_per_visible_prompt": _safe_divide(total_actual_tokens, total_visible_prompts),
                "estimated_tokens_per_visible_prompt": _safe_divide(total_estimated_tokens, total_visible_prompts),
            }
        )

    return _sort_phase_frame(pd.DataFrame(rows))


def _aggregate_universal_user_phase_metrics(df_universal_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate universal metrics into one row per user/phase."""
    df = _prepare_universal_metrics_df(df_universal_metrics)
    if df.empty:
        return pd.DataFrame()

    group_cols = ["phase", "user_id"]
    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(
            man_hours=("man_hours", "sum"),
            man_days=("man_days", "sum"),
            total_prompts=("total_prompts", "sum"),
            total_visible_prompts=("total_visible_prompts", "sum"),
            total_redacted_prompts=("total_redacted_prompts", "sum"),
            total_tool_followup_turns=("total_tool_followup_turns", "sum"),
            retry_prompt_count=("retry_prompt_count", "sum"),
            completed_features=("completed_features", "sum"),
            total_lines_written_by_selected_model_agent=("total_lines_written_by_selected_model_agent", "sum"),
            number_of_ai_lines_needing_human_revision=("number_of_ai_lines_needing_human_revision", "sum"),
            tracking_human_written_lines=("tracking_human_written_lines", "sum"),
            tracking_human_edit_lines=("tracking_human_edit_lines", "sum"),
            tracking_human_deleted_lines=("tracking_human_deleted_lines", "sum"),
            tracking_ai_generated_lines=("tracking_ai_generated_lines", "sum"),
            tracking_ai_removed_lines=("tracking_ai_removed_lines", "sum"),
            tracking_ai_edited_lines=("tracking_ai_edited_lines", "sum"),
            tracking_missing_features=("tracking_missing_features", "sum"),
            tracking_rows=("tracking_rows", "sum"),
            tracking_prompt_rows=("tracking_prompt_rows", "sum"),
            tracking_sheet_count=("tracking_sheet_count", "sum"),
            total_copilot_applied_edit_lines=("total_copilot_applied_edit_lines", "sum"),
            total_copilot_applied_edit_calls=("total_copilot_applied_edit_calls", "sum"),
            total_copilot_edited_file_events=("total_copilot_edited_file_events", "sum"),
            total_copilot_distinct_edited_files=("total_copilot_distinct_edited_files", "sum"),
            total_assistant_code_lines=("total_assistant_code_lines", "sum"),
            total_assistant_fallback_code_lines=("total_assistant_fallback_code_lines", "sum"),
            total_tool_code_lines=("total_tool_code_lines", "sum"),
            total_metering_usage=("total_metering_usage", "sum"),
            total_actual_prompt_tokens=("total_actual_prompt_tokens", "sum"),
            total_actual_completion_tokens=("total_actual_completion_tokens", "sum"),
            total_estimated_prompt_tokens=("total_estimated_prompt_tokens", "sum"),
            total_estimated_completion_tokens=("total_estimated_completion_tokens", "sum"),
        )
        .reset_index()
    )
    grouped["prompt_success_rate"] = grouped.apply(
        lambda row: _safe_divide(
            row["total_visible_prompts"] - row["retry_prompt_count"],
            row["total_visible_prompts"],
        ),
        axis=1,
    )
    grouped["prompt_to_feature_ratio"] = grouped.apply(
        lambda row: _safe_divide(row["total_visible_prompts"], row["completed_features"]),
        axis=1,
    )
    grouped["prompt_visibility_rate"] = grouped.apply(
        lambda row: _safe_divide(row["total_visible_prompts"], row["total_prompts"]),
        axis=1,
    )
    grouped["ai_lines_per_hour"] = grouped.apply(
        lambda row: _safe_divide(
            row["total_lines_written_by_selected_model_agent"],
            row["man_hours"],
        ),
        axis=1,
    )
    grouped["structured_edit_share"] = grouped.apply(
        lambda row: _safe_divide(
            row["total_tool_code_lines"],
            row["total_lines_written_by_selected_model_agent"],
        ),
        axis=1,
    )
    return _sort_phase_frame(grouped)


def _aggregate_serialized_count_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Aggregate JSON-serialized count dictionaries from a dataframe column."""
    if df.empty or column_name not in df.columns:
        return pd.DataFrame(columns=["label", "count"])

    counts: dict[str, int] = {}
    for raw in df[column_name].dropna():
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            parsed = {}
        if not isinstance(parsed, dict):
            continue
        for key, value in parsed.items():
            try:
                counts[str(key)] = counts.get(str(key), 0) + int(value)
            except Exception:
                continue

    if not counts:
        return pd.DataFrame(columns=["label", "count"])

    return pd.DataFrame(
        [{"label": key, "count": value} for key, value in counts.items()]
    ).sort_values("count", ascending=False)


def _prepare_ritm_phase_df(df_ritm_phase: pd.DataFrame) -> pd.DataFrame:
    """Return a typed, phase-sorted copy of parsed RITM phase metrics."""
    if df_ritm_phase is None or df_ritm_phase.empty:
        return pd.DataFrame()

    prepared = df_ritm_phase.copy()
    numeric_cols = [
        "total_points",
        "earned_points",
        "missing_points",
        "requirement_adherence_score",
        "requirement_adherence_percent",
        "requirement_count",
        "documented_issue_count",
    ]
    for col in numeric_cols:
        if col in prepared.columns:
            prepared[col] = pd.to_numeric(prepared[col], errors="coerce")

    return _sort_phase_frame(prepared)


def _render_ritm_graphics(
    df_ritm_phase: pd.DataFrame,
    df_ritm_notes: pd.DataFrame,
    chart_key_prefix: str,
    *,
    compact: bool = False,
) -> None:
    """Render presentation-ready RITM charts from the parsed PDF."""
    ritm_phase_df = _prepare_ritm_phase_df(df_ritm_phase)
    if ritm_phase_df.empty:
        st.info("No RITM PDF data is loaded yet.")
        return

    ritm_phase_df = _sort_phase_frame(ritm_phase_df)
    best_phase = ritm_phase_df.sort_values("requirement_adherence_score", ascending=False).iloc[0]
    first_phase = ritm_phase_df.iloc[0]
    last_phase = ritm_phase_df.iloc[-1]
    phase_gain = (
        float(last_phase["requirement_adherence_percent"]) - float(first_phase["requirement_adherence_percent"])
        if len(ritm_phase_df) > 1
        else 0.0
    )
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("RITM Phases Loaded", f"{ritm_phase_df['phase'].nunique():.0f}")
    metric_col2.metric(
        "Phase 1 to 3 Gain",
        f"{phase_gain:+.2f} pts",
    )
    metric_col3.metric(
        "Best RITM Phase",
        str(best_phase["phase"]),
        delta=f"{best_phase['requirement_adherence_percent']:.2f}%",
    )
    metric_col4.metric(
        "Documented Issue Notes",
        f"{ritm_phase_df['documented_issue_count'].fillna(0).sum():.0f}",
    )

    chart_row_1_left, chart_row_1_right = st.columns(2)
    with chart_row_1_left:
        fig_ritm_score = go.Figure()
        fig_ritm_score.add_trace(
            go.Scatter(
                x=ritm_phase_df["phase"],
                y=ritm_phase_df["requirement_adherence_percent"],
                mode="lines+markers+text",
                text=ritm_phase_df["requirement_adherence_percent"].map(lambda v: f"{v:.2f}%"),
                textposition="top center",
                line=dict(color="#1D4ED8", width=4),
                marker=dict(
                    size=14,
                    color=[PHASE_COLORS.get(str(phase), "#1D4ED8") for phase in ritm_phase_df["phase"]],
                    line=dict(color="white", width=2),
                ),
                customdata=ritm_phase_df[["earned_points", "total_points", "missing_points"]],
                hovertemplate=(
                    "Phase: %{x}<br>"
                    "Requirement adherence: %{y:.2f}%<br>"
                    "Earned points: %{customdata[0]:.0f} / %{customdata[1]:.0f}<br>"
                    "Missing points: %{customdata[2]:.0f}<extra></extra>"
                ),
                name="Requirement Adherence",
            )
        )
        fig_ritm_score.add_hline(
            y=90.0,
            line_dash="dot",
            line_color="#94A3B8",
            annotation_text="90% reference",
            annotation_position="top left",
        )
        for idx in range(1, len(ritm_phase_df)):
            prev_row = ritm_phase_df.iloc[idx - 1]
            curr_row = ritm_phase_df.iloc[idx]
            delta = float(curr_row["requirement_adherence_percent"]) - float(prev_row["requirement_adherence_percent"])
            fig_ritm_score.add_annotation(
                x=curr_row["phase"],
                y=float(curr_row["requirement_adherence_percent"]),
                text=f"{delta:+.2f} pts vs prior",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-36,
                bgcolor="rgba(255,255,255,0.88)",
                bordercolor="#CBD5E1",
                font=dict(size=11, color="#334155"),
            )
        y_min = max(0.0, float(ritm_phase_df["requirement_adherence_percent"].min()) - 6.0)
        fig_ritm_score.update_layout(
            title="RITM Requirement Adherence Trend",
            showlegend=False,
            xaxis_title="Phase",
            yaxis_title="Requirement Adherence (%)",
        )
        fig_ritm_score.update_yaxes(range=[y_min, 100.0], ticksuffix="%")
        st.plotly_chart(
            _style_figure(fig_ritm_score),
            width="stretch",
            key=f"{chart_key_prefix}_ritm_score",
        )

    with chart_row_1_right:
        fig_ritm_points = go.Figure()
        fig_ritm_points.add_trace(
            go.Bar(
                y=ritm_phase_df["phase"],
                x=ritm_phase_df["earned_points"],
                name="Earned Points",
                orientation="h",
                marker_color="#10B981",
                text=ritm_phase_df["earned_points"].map(lambda v: f"{v:.0f} earned"),
                textposition="inside",
                customdata=ritm_phase_df["total_points"],
                hovertemplate=(
                    "Phase: %{y}<br>"
                    "Earned points: %{x:.0f} / %{customdata:.0f}<extra></extra>"
                ),
            )
        )
        fig_ritm_points.add_trace(
            go.Bar(
                y=ritm_phase_df["phase"],
                x=ritm_phase_df["missing_points"],
                name="Missing Points",
                orientation="h",
                marker_color="#EF4444",
                text=ritm_phase_df["missing_points"].map(lambda v: f"{v:.0f} missing"),
                textposition="inside",
                hovertemplate="Phase: %{y}<br>Missing points: %{x:.0f}<extra></extra>",
            )
        )
        fig_ritm_points.update_layout(
            barmode="stack",
            title="RITM Point Coverage by Phase",
            xaxis_title="RITM Points (out of 22)",
            yaxis_title="Phase",
        )
        st.plotly_chart(
            _style_figure(fig_ritm_points),
            width="stretch",
            key=f"{chart_key_prefix}_ritm_points",
        )

    if compact:
        return

    chart_row_2_left, chart_row_2_right = st.columns(2)
    with chart_row_2_left:
        if len(ritm_phase_df) > 1:
            delta_rows = []
            for idx in range(1, len(ritm_phase_df)):
                prev_row = ritm_phase_df.iloc[idx - 1]
                curr_row = ritm_phase_df.iloc[idx]
                delta_rows.append(
                    {
                        "label": f"{prev_row['phase']} -> {curr_row['phase']}",
                        "delta": float(curr_row["requirement_adherence_percent"]) - float(prev_row["requirement_adherence_percent"]),
                    }
                )
            delta_df = pd.DataFrame(delta_rows)
            fig_ritm_delta = px.bar(
                delta_df,
                x="label",
                y="delta",
                color="delta",
                color_continuous_scale=["#DC2626", "#F59E0B", "#10B981"],
                title="RITM Adherence Improvement Between Phases",
                labels={"label": "Phase Transition", "delta": "Change in Adherence (points)"},
            )
            fig_ritm_delta.update_traces(texttemplate="%{y:+.2f}", textposition="outside", cliponaxis=False)
            fig_ritm_delta.update_coloraxes(showscale=False)
            fig_ritm_delta.update_yaxes(ticksuffix=" pts")
        else:
            fig_ritm_delta = px.bar(
                ritm_phase_df,
                x="phase",
                y="documented_issue_count",
                color="phase",
                color_discrete_map=PHASE_COLORS,
                title="Documented RITM Issue Notes by Phase",
                labels={"phase": "Phase", "documented_issue_count": "Issue Notes"},
            )
            fig_ritm_delta.update_layout(showlegend=False)
            fig_ritm_delta.update_traces(texttemplate="%{y:.0f}", textposition="outside", cliponaxis=False)
        st.plotly_chart(
            _style_figure(fig_ritm_delta),
            width="stretch",
            key=f"{chart_key_prefix}_ritm_delta",
        )

    with chart_row_2_right:
        if df_ritm_notes is not None and not df_ritm_notes.empty:
            component_issue_df = df_ritm_notes.groupby(["component", "phase"]).size().reset_index(name="issue_count")
            component_order = (
                component_issue_df.groupby("component")["issue_count"].sum().sort_values(ascending=False).index.tolist()
            )
            phase_order = ritm_phase_df["phase"].tolist()
            issue_matrix = (
                component_issue_df.pivot(index="component", columns="phase", values="issue_count")
                .reindex(index=component_order, columns=phase_order)
                .fillna(0)
            )
            heatmap_text = [["Issue" if float(value) > 0 else "" for value in row] for row in issue_matrix.values]
            fig_component_issues = go.Figure(
                data=
                [
                    go.Heatmap(
                        z=issue_matrix.values,
                        x=issue_matrix.columns.tolist(),
                        y=issue_matrix.index.tolist(),
                        text=heatmap_text,
                        texttemplate="%{text}",
                        colorscale=[
                            [0.0, "#F8FAFC"],
                            [0.001, "#FDE68A"],
                            [1.0, "#F59E0B"],
                        ],
                        zmin=0,
                        zmax=max(1.0, float(issue_matrix.values.max())),
                        colorbar=dict(title="Issue Notes"),
                        hovertemplate=(
                            "Component: %{y}<br>"
                            "Phase: %{x}<br>"
                            "Issue notes: %{z:.0f}<extra></extra>"
                        ),
                    )
                ]
            )
            fig_component_issues.update_layout(
                title="Where RITM Issues Appear Across Phases",
                xaxis_title="Phase",
                yaxis_title="Component",
            )
            st.plotly_chart(
                _style_figure(fig_component_issues),
                width="stretch",
                key=f"{chart_key_prefix}_ritm_component_issues",
            )
        else:
            st.info("No component-level RITM notes are available.")

    if df_ritm_notes is not None and not df_ritm_notes.empty:
        st.write("### RITM Notes")
        st.dataframe(
            df_ritm_notes.rename(
                columns={"phase": "Phase", "component": "Component", "note": "Documented Note"}
            ),
            width="stretch",
            hide_index=True,
        )


__all__ = [name for name in globals() if not name.startswith("__")]
