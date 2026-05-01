import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import json

from views.shared import *

def render_universal_dashboard(
    df_universal_metrics: pd.DataFrame,
    df_universal_all: pd.DataFrame,
    df_git: pd.DataFrame | None = None,
    df_ritm_phase: pd.DataFrame | None = None,
    df_ritm_notes: pd.DataFrame | None = None,
    df_chat_analysis: pd.DataFrame | None = None,
) -> None:
    """Render a unified dashboard combining universal metrics, insights, and timeline."""
    st.subheader("Universal Metrics and Timeline Dashboard")

    # Section: Universal Comparative Analysis
    st.write("### Universal Comparative Analysis")
    phase_df = _aggregate_universal_phase_metrics(df_universal_metrics)
    if phase_df.empty:
        st.info("No universal metrics loaded. Select universal files in the sidebar.")
        return

    user_phase_df = _aggregate_universal_user_phase_metrics(df_universal_metrics)
    st.caption("Comparative charts across the normalized Phase 1, Phase 2, and Phase 3 datasets.")

    st.write("### Phase Signal & Output Share")
    top_left, top_right = st.columns(2)

    with top_left:
        prompt_phase_df = phase_df[phase_df["total_prompts"].fillna(0) > 0][["phase", "total_prompts"]]
        if not prompt_phase_df.empty:
            fig_prompt_phase = px.pie(
                prompt_phase_df,
                names="phase",
                values="total_prompts",
                color="phase",
                color_discrete_map=PHASE_COLORS,
                hole=0.48,
                title="Natural Prompt Share by Phase",
            )
            st.plotly_chart(
                _style_donut(fig_prompt_phase),
                width="stretch",
                key="universal_comp_prompt_phase",
            )
        else:
            st.info("No natural prompt totals are available.")

    with top_right:
        ai_phase_df = phase_df[
            phase_df["total_lines_written_by_selected_model_agent"].fillna(0) > 0
        ][["phase", "total_lines_written_by_selected_model_agent"]]
        if not ai_phase_df.empty:
            fig_ai_phase = px.pie(
                ai_phase_df,
                names="phase",
                values="total_lines_written_by_selected_model_agent",
                color="phase",
                color_discrete_map=PHASE_COLORS,
                hole=0.48,
                title="Measured AI Output Share by Phase",
            )
            st.plotly_chart(
                _style_donut(fig_ai_phase),
                width="stretch",
                key="universal_comp_ai_phase",
            )
        else:
            st.info("No measurable AI output totals are available.")

    st.write("### Output Composition & Copilot Editor Changes")
    output_left, output_right = st.columns(2)

    with output_left:
        output_mix_df = phase_df.melt(
            id_vars="phase",
            value_vars=["total_assistant_fallback_code_lines", "total_tool_code_lines"],
            var_name="metric",
            value_name="value",
        )
        output_mix_df["metric"] = output_mix_df["metric"].replace(
            {
                "total_assistant_fallback_code_lines": "Assistant Fallback Code Lines",
                "total_tool_code_lines": "Structured Edit Lines",
            }
        )
        fig_output_mix = px.bar(
            output_mix_df,
            x="phase",
            y="value",
            color="metric",
            barmode="stack",
            title="Measured AI Output Composition by Phase",
            color_discrete_map=SIGNAL_COLORS,
        )
        st.plotly_chart(
            _style_figure(fig_output_mix),
            width="stretch",
            key="universal_comp_output_mix",
        )

    with output_right:
        copilot_edit_phase_df = phase_df[
            phase_df["total_copilot_applied_edit_lines"].fillna(0) > 0
        ][["phase", "total_copilot_applied_edit_lines"]]
        if not copilot_edit_phase_df.empty:
            fig_copilot_phase = px.pie(
                copilot_edit_phase_df,
                names="phase",
                values="total_copilot_applied_edit_lines",
                color="phase",
                color_discrete_map=PHASE_COLORS,
                hole=0.48,
                title="Copilot Applied Editor Edit Share by Phase",
            )
            st.plotly_chart(
                _style_donut(fig_copilot_phase),
                width="stretch",
                key="universal_comp_copilot_edit_share",
            )
        else:
            st.info("No Copilot applied editor-change totals are available in the selected files.")

    st.write("### Prompt Signal Quality")
    signal_left, signal_right = st.columns(2)

    with signal_left:
        signal_df = phase_df.melt(
            id_vars="phase",
            value_vars=[
                "total_visible_prompts",
                "total_redacted_prompts",
                "total_tool_followup_turns",
            ],
            var_name="metric",
            value_name="value",
        )
        signal_df["metric"] = signal_df["metric"].replace(
            {
                "total_visible_prompts": "Visible Prompts",
                "total_redacted_prompts": "Redacted Prompts",
                "total_tool_followup_turns": "Tool Follow-ups",
            }
        )
        fig_signal = px.bar(
            signal_df,
            x="phase",
            y="value",
            color="metric",
            barmode="stack",
            title="Prompt Signal Composition by Phase",
            color_discrete_map=SIGNAL_COLORS,
        )
        st.plotly_chart(
            _style_figure(fig_signal),
            width="stretch",
            key="universal_comp_signal_mix",
        )

    with signal_right:
        visibility_df = phase_df.copy()
        visibility_df["prompt_visibility_pct"] = visibility_df["prompt_visibility_rate"].fillna(0) * 100.0
        visibility_df["tool_followup_pct"] = visibility_df["tool_followup_share"].fillna(0) * 100.0
        fig_visibility = px.scatter(
            visibility_df,
            x="prompt_visibility_pct",
            y="tool_followup_pct",
            size="total_prompts",
            color="phase",
            color_discrete_map=PHASE_COLORS,
            text="phase",
            title="Prompt Visibility vs Agent Loop Intensity",
            labels={
                "prompt_visibility_pct": "Visible Prompt Share (%)",
                "tool_followup_pct": "Tool Follow-up Share (%)",
                "total_prompts": "Natural Prompts",
            },
        )
        fig_visibility.update_traces(textposition="top center")
        st.plotly_chart(
            _style_figure(fig_visibility),
            width="stretch",
            key="universal_comp_visibility_bubble",
        )

    st.write("### Efficiency & Quality Profile")
    efficiency_left, efficiency_right = st.columns(2)
    with efficiency_left:
        fig_success = px.bar(
            phase_df,
            x="phase",
            y="prompt_success_rate",
            color="phase",
            color_discrete_map=PHASE_COLORS,
            title="Visible Prompt Success Rate by Phase",
            labels={"prompt_success_rate": "Success Rate", "phase": "Phase"},
        )
        fig_success.update_yaxes(tickformat=".0%")
        st.plotly_chart(
            _style_figure(fig_success),
            width="stretch",
            key="universal_comp_success_bar",
        )

    with efficiency_right:
        fig_ratio = px.bar(
            phase_df,
            x="phase",
            y="prompt_to_feature_ratio",
            color="phase",
            color_discrete_map=PHASE_COLORS,
            title="Visible Prompts per Heuristic Feature Run",
            labels={"prompt_to_feature_ratio": "Visible Prompts per Run", "phase": "Phase"},
        )
        st.plotly_chart(
            _style_figure(fig_ratio),
            width="stretch",
            key="universal_comp_ratio_bar",
        )

    efficiency_bottom_left, efficiency_bottom_right = st.columns(2)
    with efficiency_bottom_left:
        fig_throughput = px.bar(
            phase_df,
            x="phase",
            y="ai_lines_per_hour",
            color="phase",
            color_discrete_map=PHASE_COLORS,
            title="Measured AI Throughput per Work Hour",
            labels={"ai_lines_per_hour": "AI Lines per Hour", "phase": "Phase"},
        )
        st.plotly_chart(
            _style_figure(fig_throughput),
            width="stretch",
            key="universal_comp_throughput_bar",
        )

    with efficiency_bottom_right:
        fig_revision_density = px.bar(
            phase_df,
            x="phase",
            y="revisions_per_100_ai_lines",
            color="phase",
            color_discrete_map=PHASE_COLORS,
            title="AI Revision Density per 100 AI Lines",
            labels={"revisions_per_100_ai_lines": "Revised AI Lines / 100", "phase": "Phase"},
        )
        st.plotly_chart(
            _style_figure(fig_revision_density),
            width="stretch",
            key="universal_comp_revision_density",
        )

    st.write("### Copilot Editor Activity")
    copilot_left, copilot_right = st.columns(2)

    with copilot_left:
        fig_copilot_activity = px.bar(
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
            _style_figure(fig_copilot_activity),
            width="stretch",
            key="universal_comp_copilot_activity",
        )
        phase_call_parts = []
        for _, row in phase_df.iterrows():
            phase_call_parts.append(
                f"{row['phase']}: {int(row['total_copilot_applied_edit_calls'] or 0)} edit calls / "
                f"{int(row['total_copilot_edited_file_events'] or 0)} edited-file events"
            )
        st.caption("Copilot activity support counts: " + "; ".join(phase_call_parts))

    with copilot_right:
        copilot_tool_df = _aggregate_serialized_count_column(
            df_universal_metrics,
            "copilot_edit_tool_counts",
        ).head(8)
        if not copilot_tool_df.empty:
            fig_copilot_tools = px.pie(
                copilot_tool_df,
                names="label",
                values="count",
                hole=0.48,
                title="Copilot Edit Tool Mix",
            )
            st.plotly_chart(
                _style_donut(fig_copilot_tools),
                width="stretch",
                key="universal_comp_copilot_tool_mix",
            )
        else:
            st.info("No Copilot edit tool usage is available in the selected files.")

    st.write("### User Share Within a Phase")
    available_phases = phase_df["phase"].dropna().tolist()
    selected_phase = st.selectbox(
        "Focus phase for user comparison",
        options=available_phases,
        index=0 if available_phases else None,
        key="universal_comparative_phase",
    )

    selected_user_phase_df = user_phase_df[user_phase_df["phase"] == selected_phase].copy()

    user_left, user_right = st.columns(2)

    with user_left:
        user_prompt_df = selected_user_phase_df[
            selected_user_phase_df["total_prompts"].fillna(0) > 0
        ][["user_id", "total_prompts", "total_visible_prompts", "total_redacted_prompts"]]
        if not user_prompt_df.empty:
            fig_user_prompt = px.pie(
                user_prompt_df,
                names="user_id",
                values="total_prompts",
                hole=0.48,
                title=f"Natural Prompt Share by User ({selected_phase})",
            )
            st.plotly_chart(
                _style_donut(fig_user_prompt),
                width="stretch",
                key="universal_comp_user_prompt",
            )
        else:
            st.info("No natural prompt totals are available for this phase.")

        if not user_prompt_df.empty:
            parts = [
                (
                    f"{row['user_id']}: "
                    f"{int(row['total_visible_prompts'])} visible / "
                    f"{int(row['total_redacted_prompts'])} redacted"
                )
                for _, row in user_prompt_df.iterrows()
            ]
            st.caption("Prompt visibility in this phase: " + ", ".join(parts))

    with user_right:
        user_ai_df = selected_user_phase_df[
            selected_user_phase_df["total_lines_written_by_selected_model_agent"].fillna(0) > 0
        ][["user_id", "total_lines_written_by_selected_model_agent"]]
        if not user_ai_df.empty:
            fig_user_ai = px.pie(
                user_ai_df,
                names="user_id",
                values="total_lines_written_by_selected_model_agent",
                hole=0.48,
                title=f"Measured AI Output Share by User ({selected_phase})",
            )
            st.plotly_chart(
                _style_donut(fig_user_ai),
                width="stretch",
                key="universal_comp_user_ai",
            )
        else:
            st.info("No AI output totals available for this phase.")

        zero_output_users = selected_user_phase_df[
            selected_user_phase_df["total_lines_written_by_selected_model_agent"].fillna(0) <= 0
        ]["user_id"].dropna().astype(str).tolist()
        if zero_output_users:
            st.caption(
                "Users with zero measurable AI output in this phase: "
                + ", ".join(sorted(zero_output_users))
                + ". This usually means the selected logs did not preserve reconstructable code-line output for them."
            )

    copilot_user_df = selected_user_phase_df[
        selected_user_phase_df["total_copilot_applied_edit_lines"].fillna(0) > 0
    ][["user_id", "total_copilot_applied_edit_lines"]]
    if not copilot_user_df.empty:
        fig_user_copilot = px.bar(
            copilot_user_df,
            x="user_id",
            y="total_copilot_applied_edit_lines",
            color="user_id",
            title=f"Copilot Applied Edit Lines by User ({selected_phase})",
            labels={
                "user_id": "User",
                "total_copilot_applied_edit_lines": "Applied Edit Lines",
            },
        )
        st.plotly_chart(
            _style_figure(fig_user_copilot),
            width="stretch",
            key="universal_comp_user_copilot_lines",
        )

    st.write("### Coverage & Tooling")
    coverage_left, coverage_right = st.columns(2)

    with coverage_left:
        token_mix_df = phase_df.melt(
            id_vars="phase",
            value_vars=["total_actual_tokens", "total_estimated_tokens"],
            var_name="metric",
            value_name="value",
        )
        token_mix_df["metric"] = token_mix_df["metric"].replace(
            {
                "total_actual_tokens": "Actual Tokens",
                "total_estimated_tokens": "Estimated Tokens",
            }
        )
        fig_token_mix = px.bar(
            token_mix_df,
            x="phase",
            y="value",
            color="metric",
            barmode="group",
            title="Actual vs Estimated Token Coverage by Phase",
            color_discrete_map=SIGNAL_COLORS,
        )
        st.plotly_chart(
            _style_figure(fig_token_mix),
            width="stretch",
            key="universal_comp_token_mix",
        )

    with coverage_right:
        tool_call_df = _aggregate_serialized_count_column(df_universal_metrics, "tool_call_counts").head(8)
        if not tool_call_df.empty:
            fig_tool_calls = px.pie(
                tool_call_df,
                names="label",
                values="count",
                hole=0.48,
                title="Top Universal Tool Call Categories",
            )
            st.plotly_chart(
                _style_donut(fig_tool_calls),
                width="stretch",
                key="universal_comp_tool_calls",
            )
        else:
            st.info("No tool call coverage data loaded.")

    st.write("### Mode-Specific Spotlight")
    spotlight_left, spotlight_right = st.columns(2)
    with spotlight_left:
        copilot_exact_df = phase_df[
            phase_df["actual_tokens_per_visible_prompt"].fillna(0) > 0
        ][["phase", "actual_tokens_per_visible_prompt"]]
        if not copilot_exact_df.empty:
            fig_exact_tokens = px.bar(
                copilot_exact_df,
                x="phase",
                y="actual_tokens_per_visible_prompt",
                color="phase",
                color_discrete_map=PHASE_COLORS,
                title="Exact Tokens per Visible Prompt (Copilot-heavy phases)",
                labels={
                    "actual_tokens_per_visible_prompt": "Exact Tokens per Visible Prompt",
                    "phase": "Phase",
                },
            )
            st.plotly_chart(
                _style_figure(fig_exact_tokens),
                width="stretch",
                key="universal_comp_exact_tokens_phase",
            )
        else:
            st.info("No exact-token phases are available in the selected files.")

    with spotlight_right:
        kiro_credit_df = user_phase_df[
            (user_phase_df["phase"] == "Phase 3")
            & (user_phase_df["total_metering_usage"].fillna(0) > 0)
        ][["user_id", "total_metering_usage"]]
        if not kiro_credit_df.empty:
            fig_kiro_credit = px.pie(
                kiro_credit_df,
                names="user_id",
                values="total_metering_usage",
                hole=0.48,
                title="Kiro Credit Share by User (Phase 3)",
            )
            st.plotly_chart(
                _style_donut(fig_kiro_credit),
                width="stretch",
                key="universal_comp_kiro_credit_user",
            )
        else:
            st.info("No exact Kiro credit totals are available in the selected files.")

    st.write("### Phase Comparison Table")
    comparative_table = phase_df[
        [
            "phase",
            "contributors",
            "files_loaded",
            "man_hours",
            "total_prompts",
            "total_visible_prompts",
            "total_redacted_prompts",
            "completed_features",
            "prompt_visibility_rate",
            "prompt_success_rate",
            "prompt_to_feature_ratio",
            "ai_lines_per_hour",
            "revisions_per_100_ai_lines",
            "total_lines_written_by_selected_model_agent",
            "total_copilot_applied_edit_lines",
            "total_copilot_applied_edit_calls",
            "number_of_ai_lines_needing_human_revision",
            "total_actual_tokens",
            "total_estimated_tokens",
            "total_metering_usage",
        ]
    ].copy()
    comparative_table["prompt_visibility_rate"] = comparative_table["prompt_visibility_rate"].apply(
        lambda v: f"{v * 100.0:.1f}%" if pd.notna(v) else "N/A"
    )
    comparative_table["prompt_success_rate"] = comparative_table["prompt_success_rate"].apply(
        lambda v: f"{v * 100.0:.1f}%" if pd.notna(v) else "N/A"
    )
    comparative_table["prompt_to_feature_ratio"] = comparative_table["prompt_to_feature_ratio"].apply(
        lambda v: f"{v:.2f}" if pd.notna(v) else "N/A"
    )
    comparative_table["ai_lines_per_hour"] = comparative_table["ai_lines_per_hour"].apply(
        lambda v: f"{v:.1f}" if pd.notna(v) else "N/A"
    )
    comparative_table["revisions_per_100_ai_lines"] = comparative_table["revisions_per_100_ai_lines"].apply(
        lambda v: f"{v:.1f}" if pd.notna(v) else "N/A"
    )
    comparative_table = comparative_table.rename(
        columns={
            "phase": "Phase",
            "contributors": "Contributors",
            "files_loaded": "Files Loaded",
            "man_hours": "Work Hours",
            "man_days": "Team Man-Days",
            "total_prompts": "Natural Prompts",
            "total_visible_prompts": "Visible Prompts",
            "total_redacted_prompts": "Redacted Prompts",
            "completed_features": "Heuristic Feature Runs",
            "prompt_visibility_rate": "Prompt Visibility",
            "prompt_success_rate": "Prompt Success",
            "prompt_to_feature_ratio": "Visible Prompts per Run",
            "total_lines_written_by_selected_model_agent": "AI Lines",
            "number_of_ai_lines_needing_human_revision": "AI Lines Revised",
            "total_actual_tokens": "Actual Tokens",
            "total_estimated_tokens": "Estimated Tokens",
            "total_metering_usage": "Kiro Credits",
            "average_context_usage_percentage": "Avg Context Usage",
            "requirement_adherence_percent": "RITM Adherence",
            "documented_issue_count": "RITM Issue Notes",
        }
    )
    st.dataframe(comparative_table, width="stretch", hide_index=True)



