import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
from git import Repo

st.set_page_config(page_title="Copilot History Analyzer", layout="wide")

st.title("Copilot History Analyzer")

# --- Sidebar ---
st.sidebar.header("Configuration")
uploaded_files = st.sidebar.file_uploader("Upload chatTemplate.json", type="json", accept_multiple_files=True)
repo_path = st.sidebar.text_input("Local Git Repository Path (Optional)", help="Path to the root of your local git repository to correlate chat with commits.")

import re

# --- Data Processing Functions ---

@st.cache_data
def parse_chat_data(files):
    all_requests = []
    
    for uploaded_file in files:
        try:
            data = json.load(uploaded_file)
            requests = data.get("requests", [])
            
            # Try to identify user from file paths in the first few requests
            suspected_user = "Unknown User"
            for req in requests:
                if suspected_user != "Unknown User":
                    break
                
                # Check variableData for file paths
                variables = req.get("variableData", {}).get("variables", [])
                for var in variables:
                    val = var.get("value", {})
                    path = val.get("fsPath") or val.get("path")
                    if path:
                        # Regex for Windows/Mac/Linux home directories
                        match = re.search(r'[/\\](?:Users|home)[/\\]([^/\\\\]+)', path, re.IGNORECASE)
                        if match:
                            suspected_user = match.group(1)
                            break
            
            for req in requests:
                timestamp = req.get("timestamp")
                if not timestamp:
                    continue
                
                # Convert ms to datetime
                dt = datetime.fromtimestamp(timestamp / 1000.0)
                
                # User Message
                user_msg = req.get("message", {}).get("text", "")
                
                # Assistant Response
                response_parts = req.get("response", [])
                assistant_msg = ""
                model_name = "Unknown"
                metrics = {}
                
                for part in response_parts:
                    # Extract model info if available in result (not here per se, handled below)
                    
                    val = part.get("value")
                    if val and isinstance(val, str):
                        if part.get("kind") == "thinking":
                            continue # Skip thinking blocks for now
                        assistant_msg += val
                
                # Extract metadata from the request object directly as per example
                result = req.get("result", {})
                timings = result.get("timings", {}) # Get timings
                
                if result:
                    details = result.get("details", "")
                    if details:
                        model_name = details
                    
                    usage = result.get("usage", {})
                    metrics = usage
                
                # Context Files
                referenced_files = []
                variables = req.get("variableData", {}).get("variables", [])
                for var in variables:
                    val = var.get("value", {})
                    # Try to get file name from path
                    path = val.get("fsPath") or val.get("path")
                    if path:
                        try:
                            file_name = os.path.basename(path)
                            referenced_files.append(file_name)
                        except:
                            pass

                # Calculate Code Lines and Languages
                code_lines = 0
                languages = []
                if "```" in assistant_msg:
                    lines = assistant_msg.split('\n')
                    in_block = False
                    for line in lines:
                        if line.strip().startswith("```"):
                            if not in_block:
                                # Entering block
                                lang = line.strip().replace("```", "").strip()
                                if lang:
                                    languages.append(lang)
                            in_block = not in_block
                        elif in_block:
                            code_lines += 1

                all_requests.append({
                    "timestamp": dt,
                    "user_text": user_msg,
                    "assistant_text": assistant_msg,
                    "model": model_name,
                    "completion_tokens": metrics.get("completionTokens", 0),
                    "prompt_tokens": metrics.get("promptTokens", 0),
                    "code_lines_suggested": code_lines,
                    "file_name": uploaded_file.name,
                    "suspected_user": suspected_user,
                    "latency_ms": timings.get("totalElapsed", 0),
                    "ttft_ms": timings.get("firstProgress", 0), # Time to First Token ~ Thinking Time
                    "referenced_files": referenced_files,
                    "languages": languages,
                    "edited_file_events": len(req.get("editedFileEvents", []) or []),
                    "checkpoints_restored": 1 if any(p.get("kind") == "undoStop" for p in response_parts) else 0
                })
                
        except Exception as e:
            st.error(f"Error parsing {uploaded_file.name}: {e}")
            
    return pd.DataFrame(all_requests)

def parse_git_history(path):
    commits_data = []
    try:
        repo = Repo(path)
        for commit in repo.iter_commits():
            commits_data.append({
                "timestamp": datetime.fromtimestamp(commit.committed_date),
                "author": commit.author.name,
                "message": commit.message,
                "insertions": commit.stats.total['insertions'],
                "deletions": commit.stats.total['deletions'],
                "files": commit.stats.files
            })
    except Exception as e:
        st.error(f"Error reading git repo: {e}")
    
    return pd.DataFrame(commits_data)

def calculate_success_metrics(df):
    """
    Heuristic for Success/Reverts:
    1. Iterate through messages in chronological order per session.
    2. If an Assistant message has code suggestions (>0 lines).
    3. Look at the VERY NEXT User message.
    4. If User message is short (< 15 words) and contains negative keywords -> Revert/Fail.
    """
    total_code_responses = 0
    flagged_reverts = 0
    negative_keywords = ["error", "fix", "no", "wrong", "fail", "broken", "bug", "issue", "doesn't work", "didn't work", "restore"]
    
    unique_sessions = df['file_name'].unique()
    
    for session in unique_sessions:
        # Get session rows sorted by time
        session_df = df[df['file_name'] == session].sort_values("timestamp")
        
        # We need to pair (Assistant Response) -> (Next User Message)
        # However, our dataframe structure is 1 row = 1 Request (User -> Assistant)
        # So: Row[i].Assistant_Text  vs  Row[i+1].User_Text
        
        rows = list(session_df.iterrows())
        for i in range(len(rows) - 1):
            curr_idx, curr_row = rows[i]
            next_idx, next_row = rows[i+1]
            
            if curr_row['code_lines_suggested'] > 0:
                total_code_responses += 1
                
                # Check next user message (the prompt of the NEXT request)
                next_user_msg = str(next_row['user_text']).lower()
                word_count = len(next_user_msg.split())
                
                # Heuristic check
                if word_count < 20 and any(k in next_user_msg for k in negative_keywords):
                    flagged_reverts += 1
    
    return total_code_responses, flagged_reverts

# --- Main App Logic ---

if uploaded_files:
    df_chat_all = parse_chat_data(uploaded_files)
    
    if not df_chat_all.empty:
        # Sort by time
        df_chat_all = df_chat_all.sort_values("timestamp")
        
        # --- Sidebar ---
        st.sidebar.divider()
        st.sidebar.subheader("Chat History View")
        
        # Create unique session labels with User where available
        # But we need a mapping back to file name for filtering
        session_map = {}
        unique_sessions = df_chat_all[["file_name", "suspected_user"]].drop_duplicates()
        
        display_options = []
        for _, row in unique_sessions.iterrows():
            label = f"{row['file_name']} (User: {row['suspected_user']})"
            display_options.append(label)
            session_map[label] = row['file_name']
            
        display_options.sort()
        
        selected_display_label = st.sidebar.radio(
            "Select Chat Session to View",
            options=display_options,
            index=0 if display_options else None
        )
        
        # Get selected file name for Chat View
        selected_chat_file = session_map.get(selected_display_label)

        # --- Analysis Filters (Moved below Chat History selection) ---
        st.sidebar.divider()
        st.sidebar.subheader("Analysis Filters")
        
        # We can reuse the labels if desired, or just use file names
        # User requested filtering for analysis/stats separately
        all_sessions = sorted(df_chat_all["file_name"].unique().tolist())
        
        with st.sidebar.expander("Select Sessions for Analysis", expanded=True):
             selected_sessions_analysis = st.multiselect(
                "Filter statistics by session:",
                options=all_sessions,
                default=all_sessions
            )
        
        # Filter Data for Stats/Timeline
        df_chat_analysis = df_chat_all[df_chat_all["file_name"].isin(selected_sessions_analysis)]
        
        # Git Data
        df_git = pd.DataFrame()
        if repo_path and os.path.isdir(repo_path):
            df_git = parse_git_history(repo_path)
            if not df_git.empty:
                df_git = df_git.sort_values("timestamp")

        # --- TABS ---
        tab1, tab2, tab3 = st.tabs(["Chat History", "Statistics", "Development Timeline"])
        
        with tab1:
            st.subheader("Recreated Chat Session")
            
            if selected_chat_file:
                df_chat_view = df_chat_all[df_chat_all["file_name"] == selected_chat_file]
                
                # Show session metadata
                user_info = df_chat_view["suspected_user"].iloc[0] if not df_chat_view.empty else "Unknown"
                st.caption(f"Viewing Session: **{selected_chat_file}** | User: **{user_info}**")
                
                if df_chat_view.empty:
                    st.info("No messages in this session.")
                
                # Scrollable container for chat history
                with st.container(height=600):
                    for index, row in df_chat_view.iterrows():
                        # User Message
                        with st.chat_message("user"):
                            st.markdown(f"**{row['suspected_user']}**")
                            st.write(row["user_text"])
                            st.caption(f"{row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                        
                        # Assistant Message
                        # Format: Model Name | Code Lines
                        assistant_header = f"{row['model']} | Code Lines: {row['code_lines_suggested']}"
                        with st.chat_message("assistant"):
                            st.markdown(f"**{assistant_header}**")
                            st.markdown(row["assistant_text"])
                            st.caption(f"Tokens: {row['completion_tokens']}")
                        
                        st.divider()
            else:
                st.info("Select a chat session from the sidebar to view.")

        with tab2:
            st.subheader("Statistics: Who Created Content?")
            
            if df_chat_analysis.empty:
                 st.warning("Please select at least one session in 'Analysis Filters' to view statistics.")
            else:
                col1, col2 = st.columns(2)
                
                # --- Calculates ---
                total_code, flagged_reverts = calculate_success_metrics(df_chat_analysis)
                success_rate = 100
                if total_code > 0:
                    success_rate = ((total_code - flagged_reverts) / total_code) * 100

                with col1:
                    st.write("### AI Contribution & Quality")
                    total_code_lines = df_chat_analysis["code_lines_suggested"].sum()
                    total_tokens = df_chat_analysis["completion_tokens"].sum()
                    
                    c1, c2 = st.columns(2)
                    c1.metric("Total Code Lines", total_code_lines)
                    c1.metric("Total Tokens", total_tokens)
                    c2.metric("Success Rate (Est.)", f"{success_rate:.1f}%", help="Based on lack of negative follow-up prompts (e.g. 'fix', 'error')")
                    c2.metric("Flagged Reverts", flagged_reverts, help="Number of responses followed immediately by a correction request.")
                    
                    # Pie chart of Models
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
                        
                        # Author breakdown
                        author_counts = df_git["author"].value_counts()
                        fig_author = px.pie(author_counts, values=author_counts.values, names=author_counts.index, title="Code Commits by Author")
                        st.plotly_chart(fig_author)
                        
                        # Comparison
                        st.write("### Volume Comparison")
                        # Naive comparison: AI Suggested vs Git Insertions
                        # Note: Git insertions include things user wrote themselves OR copied from AI.
                        # We cannot strictly separate them without more granular tracking.
                        
                        fig_comp = go.Figure(data=[
                            go.Bar(name='AI Suggested Lines', x=['Code Volume'], y=[total_code_lines]),
                            go.Bar(name='Git Insertions', x=['Code Volume'], y=[total_insertions])
                        ])
                        fig_comp.update_layout(title="AI Suggestions vs Committed Code Volume")
                        st.plotly_chart(fig_comp)
                    else:
                        st.info("Enter a valid Git Repository path to see Human/Git statistics.")
                
                # --- Advanced Analytics Section ---
                st.divider()
                st.subheader("Deep Dive Analytics")
                
                col3, col4 = st.columns(2)
                
                with col3:
                    st.write("### Response Latency by Model")
                    if "latency_ms" in df_chat_analysis.columns:
                         # Filter out zero latencies if any (failed requests)
                         df_latency = df_chat_analysis[df_chat_analysis["latency_ms"] > 0].copy()
                         if not df_latency.empty:
                             try:
                                # Convert to seconds for better readability (avoids 'k' suffix for thousands of ms)
                                df_latency["latency_s"] = df_latency["latency_ms"] / 1000.0
                                df_latency["ttft_s"] = df_latency["ttft_ms"] / 1000.0

                                fig_latency = px.box(
                                    df_latency, 
                                    x="model", 
                                    y="latency_s", 
                                    points="all", 
                                    title="Total Response Latency (seconds)",
                                    hover_data={"latency_s": ":.2f", "model": False} # Clean hover: show only value
                                )
                                st.plotly_chart(fig_latency, use_container_width=True)
                                
                                # Thinking Time
                                df_ttft = df_latency[df_latency["ttft_s"] > 0]
                                if not df_ttft.empty:
                                    fig_ttft = px.box(
                                        df_ttft, 
                                        x="model", 
                                        y="ttft_s", 
                                        points="all", 
                                        title="Thinking Time / TTFT (seconds)",
                                         hover_data={"ttft_s": ":.2f", "model": False}
                                    )
                                    st.plotly_chart(fig_ttft, use_container_width=True)
                             except Exception as e:
                                 st.error(f"Error plotting latency: {e}")
                
                with col4:
                    st.write("### Languages & Context")
                    
                    # 1. Languages
                    all_langs = []
                    for langs in df_chat_analysis["languages"]:
                        if isinstance(langs, list):
                            all_langs.extend(langs)
                    
                    if all_langs:
                        lang_counts = pd.Series(all_langs).value_counts().reset_index()
                        lang_counts.columns = ["Language", "Count"]
                        fig_langs = px.bar(lang_counts, x="Language", y="Count", title="Top Programming Languages Generated")
                        st.plotly_chart(fig_langs, use_container_width=True)
                    else:
                        st.info("No code blocks detected.")

                    # 2. Context Files
                    all_files = []
                    for files in df_chat_analysis.get("referenced_files", []):
                        if isinstance(files, list):
                            all_files.extend(files)
                    
                    if all_files:
                        file_counts = pd.Series(all_files).value_counts().head(10).reset_index()
                        file_counts.columns = ["File Name", "References"]
                        fig_files = px.bar(file_counts, x="References", y="File Name", orientation='h', title="Top 10 Context Files")
                        fig_files.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig_files, use_container_width=True)
                    else:
                        st.info("No file context data found.")
                
                # --- Editor Events Section ---
                st.divider()
                st.subheader("Editor Events & Reliability")
                
                col5, col6 = st.columns(2)
                
                with col5:
                     # Checkpoints / Undo Operations
                     total_checkpoints = df_chat_analysis.get("checkpoints_restored", pd.Series([0]*len(df_chat_analysis))).sum()
                     editor_edits = df_chat_analysis.get("edited_file_events", pd.Series([0]*len(df_chat_analysis))).sum()
                     
                     st.metric("Checkpoints Restored", total_checkpoints, help="Number of times an 'undoStop' event was recorded, likely indicating a revert or checkpoint restoration.")
                     st.metric("File Edit Events", editor_edits, help="Total number of file edit events triggered by the AI agent.")
                     
                with col6:
                    # Timeline of these events
                    if total_checkpoints > 0 or editor_edits > 0:
                        df_events = df_chat_analysis[ (df_chat_analysis["checkpoints_restored"] > 0) | (df_chat_analysis["edited_file_events"] > 0) ]
                        if not df_events.empty:
                            fig_events = px.scatter(df_events, x="timestamp", y="model", size="edited_file_events", color="checkpoints_restored", title="Timeline of Edits and Restores")
                            st.plotly_chart(fig_events, use_container_width=True)
                        else:
                             st.info("No events to plot.")
                    else:
                         st.info("No checkpoint or edit events found in these sessions.")

        with tab3:
            st.subheader("Development History Timeline")
            
            # Combine data for timeline
            timeline_data = []
            
            for _, row in df_chat_analysis.iterrows():
                timeline_data.append({
                    "timestamp": row["timestamp"],
                    "type": "Chat Interaction",
                    "count": 1,
                    "code_volume": row["code_lines_suggested"],
                    "session": row["file_name"],
                    "details": f"Model: {row['model']} | Session: {row['file_name']}"
                })
            
            if not df_git.empty:
                for _, row in df_git.iterrows():
                    timeline_data.append({
                        "timestamp": row["timestamp"],
                        "type": "Git Commit",
                        "count": 1,
                        "code_volume": row["insertions"],
                        "session": "Git Repo",
                        "details": f"Author: {row['author']} | Insertions: {row['insertions']}"
                    })
            
            df_timeline = pd.DataFrame(timeline_data)
            
            if not df_timeline.empty:
                # Scatter plot timeline
                fig_timeline = px.scatter(
                    df_timeline, 
                    x="timestamp", 
                    y="type", 
                    size="code_volume",
                    color="session", # Changed color to session to differentiate
                    symbol="type",   # Use symbol to differentiate Type
                    hover_data=["details"],
                    title="Activity Timeline (Color=Session, Symbol=Type)"
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
                
                # Code Velocity
                st.write("### Code Velocity (AI Suggestions vs User Commits)")
                fig_velocity = px.line(df_timeline.sort_values("timestamp"), x="timestamp", y="code_volume", color="type", title="Code Volume Over Time")
                st.plotly_chart(fig_velocity, use_container_width=True)
                
                # Daily activity histogram
                st.write("### Daily Activity Volume")
                df_timeline['date'] = df_timeline['timestamp'].dt.date
                daily_counts = df_timeline.groupby(['date', 'type']).size().reset_index(name='count')
                
                fig_daily = px.bar(
                    daily_counts, 
                    x="date", 
                    y="count", 
                    color="type", 
                    title="Daily Interactions vs Commits"
                )
                st.plotly_chart(fig_daily, use_container_width=True)

    else:
        st.warning("No valid chat requests found in the file.")
else:
    st.info("Please upload a 'chatTemplate.json' file to begin.")
