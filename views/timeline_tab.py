import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import json

from views.shared import *

def render_timeline_tab(df_chat_analysis: pd.DataFrame, df_git: pd.DataFrame) -> None:
    """Render the Development Timeline tab.

    Args:
        df_chat_analysis: Session-filtered chat dataframe for analysis.
        df_git: Optional git commit dataframe.

    Returns:
        None. Renders Streamlit UI elements directly.
    """
    st.subheader("Development History Timeline")

    # Build one normalized event table so both sources can share charts.
    timeline_data = []

    # Chat events contribute one interaction row each.
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

    # Git events are appended into the same timeline schema.
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

    # Scatter timeline shows event type and relative code volume at each timestamp.
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
    # Velocity line chart is useful for seeing burst periods over time.
    fig_velocity = px.line(
        df_timeline.sort_values("timestamp"),
        x="timestamp",
        y="code_volume",
        color="type",
        title="Code Volume Over Time",
    )
    st.plotly_chart(fig_velocity, width="stretch")

    st.write("### Daily Activity Volume")
    # Aggregate by day/type for a compact daily intensity view.
    df_timeline["date"] = df_timeline["timestamp"].dt.date
    daily_counts = df_timeline.groupby(["date", "type"]).size().reset_index(name="count")

    fig_daily = px.bar(daily_counts, x="date", y="count", color="type", title="Daily Interactions vs Commits")
    st.plotly_chart(fig_daily, width="stretch")



