import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import json

from views.shared import *

def render_chat_history_tab(df_chat_all: pd.DataFrame, key_prefix: str = "chat_history") -> None:
    """Render the Chat History tab with a hierarchical selection menu.

    Args:
        df_chat_all: Full chat dataframe across all loaded sessions.
        key_prefix: Unique prefix for Streamlit widget keys to avoid collisions.

    Returns:
        None. Renders Streamlit UI elements directly.
    """
    st.subheader("Chat Session Explorer")

    if df_chat_all.empty:
        st.info("No chat data available.")
        return

    # Use columns to build the hierarchical selection
    col1, col2, col3 = st.columns(3)

    # 1. Select Phase
    available_phases = sorted(df_chat_all["phase"].dropna().unique().tolist())
    if not available_phases:
        st.warning("No phases found in the data.")
        return

    with col1:
        selected_phase = st.selectbox(
            "Select Phase",
            options=available_phases,
            key=f"{key_prefix}_phase_select"
        )

    df_phase = df_chat_all[df_chat_all["phase"] == selected_phase]

    # 2. Select User
    available_users = sorted(df_phase["suspected_user"].dropna().unique().tolist())
    if not available_users:
        st.warning("No users found for this phase.")
        return
    
    with col2:
        selected_user = st.selectbox(
            "Select User",
            options=available_users,
            key=f"{key_prefix}_user_select"
        )
        
    df_user = df_phase[df_phase["suspected_user"] == selected_user]

    # 3. Select Session / File Name
    # Create display labels that include the first timestamp of the file for better context
    unique_sessions = df_user[["file_name", "timestamp"]].drop_duplicates(
        subset=["file_name"], keep="first"
    ).sort_values("timestamp")
    
    session_map: dict[str, str] = {}
    display_options = []
    
    for _, row in unique_sessions.iterrows():
        start_time_str = row["timestamp"].strftime("%Y-%m-%d %H:%M")
        label = f"{start_time_str} | {row['file_name']}"
        display_options.append(label)
        session_map[label] = row["file_name"]

    with col3:
        selected_display_label = st.selectbox(
            "Select Chat Session",
            options=display_options,
            key=f"{key_prefix}_session_select"
        )

    selected_chat_file = session_map.get(selected_display_label)

    st.divider()

    # Render the active transcript.
    if selected_chat_file:
        df_chat_view = df_user[df_user["file_name"] == selected_chat_file]

        st.caption(f"Viewing Session: **{selected_chat_file}** | User: **{selected_user}** | Phase: **{selected_phase}**")

        if df_chat_view.empty:
            st.info("No messages in this session.")

        with st.container(height=600):
            for _, row in df_chat_view.iterrows():
                # Render the original user prompt.
                with st.chat_message("user"):
                    st.markdown(f"**{row['suspected_user']}**")
                    st.write(row["user_text"])
                    st.caption(f"{row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")

                # Render the assistant response with compact metadata header.
                assistant_header = f"{row['model']} | Code Lines: {row['code_lines_suggested']}"
                with st.chat_message("assistant"):
                    st.markdown(f"**{assistant_header}**")
                    st.markdown(row["assistant_text"])
                    st.caption(f"Tokens: {row['completion_tokens']}")

                st.divider()




