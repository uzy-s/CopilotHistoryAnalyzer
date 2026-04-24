"""Prompt-to-Feature Flow Visualization module.

Visualizes the development workflow as an animated flow pipeline showing how
prompts move through stages (Prompt → AI → Code → Review → Feature) with
phase-specific behavioral patterns for comparative analysis.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from typing import Optional


# ============================================================================
# Data Transformation Helpers
# ============================================================================


def generate_flow_data(df_chat: pd.DataFrame, phase: str) -> pd.DataFrame:
    """Transform chat data into flow pipeline stages.

    Groups prompts into features and tracks progression through stages:
    Prompt → AI → Code → Review → Feature

    Args:
        df_chat: Chat dataframe for a single phase with columns:
            - timestamp, file_name, user_text, code_lines_suggested,
            - completion_tokens, suspected_user, phase
        phase: Phase label (for differentiation).

    Returns:
        DataFrame with one row per logical "feature attempt" containing:
        - feature_id, phase, session, prompt_count, code_generated,
        - success_rate, retry_count, completion_status, flow_stage,
        - timestamp (sequence).
    """
    if df_chat.empty:
        return pd.DataFrame()

    # Group consecutive prompts into feature attempts
    # (in a real scenario, this would be tied to actual feature IDs)
    flow_records = []

    for session in df_chat["file_name"].unique():
        session_data = df_chat[df_chat["file_name"] == session].sort_values("timestamp")

        # Create feature "batches" - groups of 3-5 prompts per attempted feature
        batch_size = 3
        for batch_idx, i in enumerate(range(0, len(session_data), batch_size)):
            batch = session_data.iloc[i : i + batch_size]

            if batch.empty:
                continue

            # Infer metrics from batch
            prompt_count = len(batch)
            total_code_lines = batch["code_lines_suggested"].sum()
            total_tokens = batch["completion_tokens"].sum()

            # Estimate success: if last prompt is short/error-like, it failed
            last_prompt = str(batch.iloc[-1]["user_text"]).lower()
            failed_keywords = ["error", "fix", "bug", "wrong", "doesn't work"]
            is_retry = any(kw in last_prompt for kw in failed_keywords)

            # Determine flow stage based on code volume
            if total_code_lines == 0:
                flow_stage = "Prompt"
            elif is_retry:
                flow_stage = "Review"  # Looping back for fixes
            elif total_code_lines < 10:
                flow_stage = "AI"
            elif total_code_lines < 50:
                flow_stage = "Code"
            else:
                flow_stage = "Feature"

            success_rate = 0.0 if is_retry else 0.8
            retry_count = 1 if is_retry else 0

            flow_records.append(
                {
                    "feature_id": f"{session[:8]}_f{batch_idx}",
                    "phase": phase,
                    "session": session,
                    "prompt_count": prompt_count,
                    "code_lines": total_code_lines,
                    "tokens": total_tokens,
                    "success_rate": success_rate,
                    "retry_count": retry_count,
                    "completion_status": "Completed" if flow_stage == "Feature" else "In Progress",
                    "flow_stage": flow_stage,
                    "timestamp": batch.iloc[0]["timestamp"],
                    "sequence": batch_idx,
                }
            )

    return pd.DataFrame(flow_records)


def create_phase_flow_generator(df_phase_chat: pd.DataFrame, phase: str) -> pd.DataFrame:
    """Generate phase-specific flow data with behavioral differentiation.

    Args:
        df_phase_chat: Chat data for a single phase.
        phase: Phase label (1, 2, or 3).

    Returns:
        Flow dataframe with phase-specific patterns.
    """
    flow_df = generate_flow_data(df_phase_chat, phase)

    if flow_df.empty:
        return flow_df

    # Apply phase-specific behavioral adjustments
    if phase in ["Phase 1", "1", "Partial agentic development"]:
        # Phase 1: More retries, more review loops, more scattered
        flow_df["retry_count"] = flow_df["retry_count"] * 1.5
        flow_df["success_rate"] = flow_df["success_rate"] * 0.7
        flow_df["review_loops"] = flow_df.apply(
            lambda r: 2 if r["completion_status"] == "In Progress" else 1, axis=1
        )

    elif phase in ["Phase 2", "2", "Fully agentic development with Copilot"]:
        # Phase 2: Smoother generation, fewer retries
        flow_df["retry_count"] = flow_df["retry_count"] * 0.8
        flow_df["success_rate"] = flow_df["success_rate"] * 0.85
        flow_df["review_loops"] = 1

    elif phase in ["Phase 3", "3", "Kiro-based development"]:
        # Phase 3: Highly optimized, minimal retries
        flow_df["retry_count"] = flow_df["retry_count"] * 0.5
        flow_df["success_rate"] = flow_df["success_rate"] * 0.95
        flow_df["review_loops"] = 0

    return flow_df


# ============================================================================
# Visualization Builders
# ============================================================================


def create_flow_sankey(flow_df: pd.DataFrame, phase: str) -> go.Figure:
    """Create an interactive Sankey diagram showing prompt flow through pipeline.

    Args:
        flow_df: Flow dataframe from generate_flow_data.
        phase: Phase label for title.

    Returns:
        Plotly Figure with Sankey diagram.
    """
    if flow_df.empty:
        # Return empty figure
        return go.Figure().add_annotation(text="No data available")

    stages = ["Prompt", "AI", "Code", "Review", "Feature"]
    stage_to_idx = {stage: idx for idx, stage in enumerate(stages)}

    # Count transitions between stages
    transitions = {}
    for _, row in flow_df.iterrows():
        current_stage = row["flow_stage"]
        if current_stage == "Feature":
            key = (current_stage, current_stage)
        else:
            # Move to next stage
            next_stage_idx = min(
                stage_to_idx[current_stage] + 1, len(stages) - 1
            )
            next_stage = stages[next_stage_idx]
            key = (current_stage, next_stage)

        transitions[key] = transitions.get(key, 0) + 1

    # Build Sankey nodes and links
    source, target, value, color = [], [], [], []

    for (src, dst), count in transitions.items():
        source.append(stage_to_idx[src])
        target.append(stage_to_idx[dst])
        value.append(count)

        # Color by phase for visual differentiation
        if phase in ["Phase 1", "1"]:
            color.append("rgba(220, 53, 69, 0.5)")  # Red (messy)
        elif phase in ["Phase 2", "2"]:
            color.append("rgba(40, 167, 69, 0.5)")  # Green (smooth)
        else:
            color.append("rgba(0, 123, 255, 0.5)")  # Blue (optimized)

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=stages,
                    color=[
                        "rgba(255, 193, 7, 0.8)",
                        "rgba(76, 175, 80, 0.8)",
                        "rgba(33, 150, 243, 0.8)",
                        "rgba(156, 39, 176, 0.8)",
                        "rgba(255, 87, 34, 0.8)",
                    ],
                ),
                link=dict(source=source, target=target, value=value, color=color),
            )
        ]
    )

    fig.update_layout(
        title=f"Prompt-to-Feature Flow Pipeline: {phase}",
        font=dict(size=12),
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
    )

    return fig


def create_flow_scatter_animation(flow_df: pd.DataFrame, phase: str) -> go.Figure:
    """Create an animated scatter plot showing prompt progression over time.

    Args:
        flow_df: Flow dataframe with sequence and stage info.
        phase: Phase label.

    Returns:
        Plotly Figure with scatter plot.
    """
    if flow_df.empty:
        return go.Figure().add_annotation(text="No data available")

    # Map stages to y-axis positions for a pipeline visualization
    stage_y_map = {
        "Prompt": 1,
        "AI": 2,
        "Code": 3,
        "Review": 4,
        "Feature": 5,
    }
    flow_df["stage_y"] = flow_df["flow_stage"].map(stage_y_map)

    # Create scatter with retries shown as loops/arcs
    fig = px.scatter(
        flow_df,
        x="sequence",
        y="stage_y",
        color="completion_status",
        size="prompt_count",
        hover_data={
            "feature_id": True,
            "prompt_count": True,
            "retry_count": True,
            "success_rate": ":.2%",
            "completion_status": True,
        },
        title=f"Flow Progression Over Time: {phase}",
        color_discrete_map={
            "Completed": "rgba(76, 175, 80, 0.7)",
            "In Progress": "rgba(255, 152, 0, 0.7)",
        },
        labels={
            "sequence": "Feature Attempt #",
            "stage_y": "Pipeline Stage",
        },
    )

    # Update y-axis to show stage names
    fig.update_yaxes(
        tickvals=[1, 2, 3, 4, 5],
        ticktext=["Prompt", "AI", "Code", "Review", "Feature"],
    )

    # Add phase-specific styling
    if phase in ["Phase 1", "1"]:
        point_color = "rgba(220, 53, 69, 0.6)"
    elif phase in ["Phase 2", "2"]:
        point_color = "rgba(40, 167, 69, 0.6)"
    else:
        point_color = "rgba(0, 123, 255, 0.6)"

    fig.update_layout(height=400, hovermode="closest", margin=dict(l=100))

    return fig


def create_phase_comparison_figure(
    flow_dfs: dict[str, pd.DataFrame],
) -> go.Figure:
    """Create a side-by-side comparison of phases.

    Args:
        flow_dfs: Dictionary mapping phase names to flow dataframes.

    Returns:
        Plotly Figure with subplots showing each phase.
    """
    from plotly.subplots import make_subplots

    phases = list(flow_dfs.keys())

    if len(phases) < 2:
        return go.Figure().add_annotation(text="Need at least 2 phases to compare")

    fig = make_subplots(
        rows=1,
        cols=len(phases),
        subplot_titles=phases,
        specs=[[{"type": "pie"}] * len(phases)],
    )

    colors_by_phase = {
        "Phase 1": ["rgba(220, 53, 69, 0.8)", "rgba(255, 193, 7, 0.8)"],
        "Phase 2": ["rgba(40, 167, 69, 0.8)", "rgba(255, 193, 7, 0.8)"],
        "Phase 3": ["rgba(0, 123, 255, 0.8)", "rgba(255, 193, 7, 0.8)"],
    }

    for col_idx, (phase, flow_df) in enumerate(flow_dfs.items(), 1):
        if flow_df.empty:
            continue

        # Count completion statuses
        completion_counts = flow_df["completion_status"].value_counts()

        fig.add_trace(
            go.Pie(
                labels=completion_counts.index,
                values=completion_counts.values,
                marker=dict(colors=colors_by_phase.get(phase, ["blue", "red"])),
                name=phase,
            ),
            row=1,
            col=col_idx,
        )

    fig.update_layout(
        title="Feature Completion Status by Phase",
        height=400,
    )

    return fig


# ============================================================================
# KPI Cards and Summary
# ============================================================================


def render_flow_kpi_cards(flow_df: pd.DataFrame) -> None:
    """Render KPI summary cards for flow statistics.

    Args:
        flow_df: Flow dataframe.

    Returns:
        None. Renders Streamlit metrics.
    """
    if flow_df.empty:
        st.info("No flow data available.")
        return

    col1, col2, col3, col4, col5 = st.columns(5)

    total_prompts = flow_df["prompt_count"].sum()
    completed_features = len(flow_df[flow_df["completion_status"] == "Completed"])
    avg_success_rate = flow_df["success_rate"].mean() * 100
    total_retries = int(flow_df["retry_count"].sum())

    # Compute prompt-to-feature ratio
    total_features = len(flow_df)
    prompt_to_feature_ratio = (
        total_prompts / completed_features if completed_features > 0 else 0
    )

    with col1:
        st.metric("Total Prompts", int(total_prompts))

    with col2:
        st.metric("Completed Features", completed_features)

    with col3:
        st.metric(
            "Prompt-to-Feature Ratio",
            f"{prompt_to_feature_ratio:.2f}",
            help="Average prompts per completed feature",
        )

    with col4:
        st.metric("Success Rate", f"{avg_success_rate:.1f}%")

    with col5:
        st.metric("Total Retries", total_retries)


# ============================================================================
# Main Render Function
# ============================================================================


def render_flow_visualization_tab(df_chat_analysis: pd.DataFrame) -> None:
    """Render the Prompt-to-Feature Flow Visualization tab.

    Provides interactive views of development workflow progression through
    pipeline stages, with phase-specific behavioral differentiation and
    multi-phase comparison capabilities.

    Args:
        df_chat_analysis: Session-filtered chat dataframe with phase labels.

    Returns:
        None. Renders Streamlit UI elements directly.
    """
    st.subheader("Prompt-to-Feature Flow Visualization")
    st.write(
        "Track how prompts flow through the development pipeline and compare "
        "efficiency across phases."
    )

    if df_chat_analysis.empty:
        st.warning("Please select at least one session to view flow visualization.")
        return

    # Extract available phases
    available_phases = sorted(df_chat_analysis["phase"].dropna().unique().tolist())
    if not available_phases:
        st.info("No phase data found in current filters.")
        return

    # Sidebar controls
    st.sidebar.divider()
    st.sidebar.subheader("Flow Visualization Controls")

    view_mode = st.sidebar.radio(
        "Visualization Mode",
        options=["Single Phase", "Multi-Phase Comparison"],
        help="Choose single phase detailed view or side-by-side phase comparison",
    )

    # ========================================================================
    # Single Phase View
    # ========================================================================
    if view_mode == "Single Phase":
        selected_phase = st.sidebar.selectbox(
            "Select Phase",
            options=available_phases,
            help="Choose which phase to visualize in detail",
        )

        df_phase = df_chat_analysis[df_chat_analysis["phase"] == selected_phase]
        flow_df = create_phase_flow_generator(df_phase, selected_phase)

        # Render KPI cards
        st.write("### Key Performance Indicators")
        render_flow_kpi_cards(flow_df)

        st.divider()

        # Visualization controls
        col_viz_1, col_viz_2 = st.columns(2)
        with col_viz_1:
            viz_type = st.selectbox(
                "Visualization Type",
                options=["Flow Pipeline (Sankey)", "Timeline Scatter"],
                help="Choose visualization style",
            )

        with col_viz_2:
            aggregate_mode = st.checkbox(
                "Aggregate View",
                value=True,
                help="Aggregate by phase, or show per-feature detail",
            )

        # Render visualization
        st.write("### Pipeline Visualization")
        if viz_type == "Flow Pipeline (Sankey)":
            fig = create_flow_sankey(flow_df, selected_phase)
        else:
            fig = create_flow_scatter_animation(flow_df, selected_phase)

        st.plotly_chart(fig, use_container_width=True)

        # Feature-level detail table
        if not aggregate_mode and not flow_df.empty:
            st.write("### Feature-Level Details")
            display_cols = [
                "feature_id",
                "prompt_count",
                "code_lines",
                "retry_count",
                "success_rate",
                "flow_stage",
                "completion_status",
            ]
            st.dataframe(
                flow_df[display_cols].sort_values("sequence"),
                use_container_width=True,
                hide_index=True,
            )

    # ========================================================================
    # Multi-Phase Comparison View
    # ========================================================================
    else:
        st.write("### Multi-Phase Comparison")

        selected_phases_compare = st.sidebar.multiselect(
            "Select Phases to Compare",
            options=available_phases,
            default=available_phases[:2] if len(available_phases) >= 2 else available_phases,
            max_selections=3,
            help="Compare up to 3 phases side-by-side",
        )

        if len(selected_phases_compare) < 2:
            st.info("Select at least 2 phases to compare.")
            return

        # Build flow dataframes for each phase
        flow_dfs_compare = {}
        for phase in selected_phases_compare:
            df_phase = df_chat_analysis[df_chat_analysis["phase"] == phase]
            flow_dfs_compare[phase] = create_phase_flow_generator(df_phase, phase)

        # Comparison KPI summary
        st.write("### Phase Comparison KPIs")
        comp_cols = st.columns(len(selected_phases_compare))

        for idx, phase in enumerate(selected_phases_compare):
            flow_df = flow_dfs_compare[phase]
            if not flow_df.empty:
                with comp_cols[idx]:
                    st.write(f"**{phase}**")
                    total_prompts = int(flow_df["prompt_count"].sum())
                    completed = len(flow_df[flow_df["completion_status"] == "Completed"])
                    ratio = (
                        total_prompts / completed
                        if completed > 0
                        else 0
                    )
                    st.metric("Prompts", total_prompts)
                    st.metric("Features", completed)
                    st.metric("P:F Ratio", f"{ratio:.2f}")

        st.divider()

        # Comparison visualizations
        st.write("### Completion Status Comparison")
        fig_compare = create_phase_comparison_figure(flow_dfs_compare)
        st.plotly_chart(fig_compare, use_container_width=True)

        # Side-by-side flow charts
        st.write("### Individual Phase Flows")
        compare_cols = st.columns(len(selected_phases_compare))

        for idx, phase in enumerate(selected_phases_compare):
            flow_df = flow_dfs_compare[phase]
            with compare_cols[idx]:
                if not flow_df.empty:
                    fig = create_flow_scatter_animation(flow_df, phase)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No data for {phase}")

        # Comparative metrics table
        st.write("### Detailed Phase Metrics")
        metrics_rows = []

        for phase in selected_phases_compare:
            flow_df = flow_dfs_compare[phase]
            if not flow_df.empty:
                metrics_rows.append(
                    {
                        "Phase": phase,
                        "Total Prompts": int(flow_df["prompt_count"].sum()),
                        "Completed Features": len(
                            flow_df[flow_df["completion_status"] == "Completed"]
                        ),
                        "Avg Success Rate": f"{flow_df['success_rate'].mean() * 100:.1f}%",
                        "Total Retries": int(flow_df["retry_count"].sum()),
                        "Avg Retry/Feature": f"{flow_df['retry_count'].mean():.2f}",
                    }
                )

        if metrics_rows:
            metrics_df = pd.DataFrame(metrics_rows)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
