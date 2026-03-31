import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import hmac
from git import Repo

st.set_page_config(page_title="Copilot History Analyzer", layout="wide")


def require_password():
    expected_password = None
    try:
        expected_password = st.secrets.get("APP_PASSWORD")
    except Exception:
        expected_password = None
    if not expected_password:
        expected_password = os.getenv("APP_PASSWORD")
    if not expected_password:
        st.error("App password is not configured. Set APP_PASSWORD in Streamlit secrets or as an environment variable.")
        st.stop()

    def on_password_submit():
        entered_password = st.session_state.get("app_password_input", "")
        st.session_state["password_ok"] = hmac.compare_digest(entered_password, expected_password)
        st.session_state["app_password_input"] = ""

    if st.session_state.get("password_ok", False):
        return

    st.title("Copilot History Analyzer")
    st.subheader("Sign in")
    st.text_input(
        "Password",
        type="password",
        key="app_password_input",
        on_change=on_password_submit,
    )

    if "password_ok" in st.session_state and not st.session_state["password_ok"]:
        st.error("Incorrect password")

    st.stop()


require_password()

st.title("Copilot History Analyzer")

# --- Sidebar ---
st.sidebar.header("Configuration")

# Local Data Discovery
DATA_DIR = "data"
available_phases = []
if os.path.exists(DATA_DIR):
    available_phases = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]

selected_phases = st.sidebar.multiselect(
    "Select Phases to Pool Data From",
    options=available_phases,
    default=available_phases,
)

uploaded_files = st.sidebar.file_uploader("Or Upload chatTemplate.json manually", type="json", accept_multiple_files=True)
repo_path = st.sidebar.text_input("Local Git Repository Path (Optional)", help="Path to the root of your local git repository to correlate chat with commits.")

import re

# --- Data Processing Functions ---

@st.cache_data
def parse_chat_data(files):
    all_requests = []

    def infer_phase_name(source_path):
        if not source_path or not isinstance(source_path, str):
            return "Uploaded"
        normalized = os.path.normpath(source_path)
        parts = normalized.split(os.sep)
        if "data" in parts:
            data_idx = parts.index("data")
            if data_idx + 1 < len(parts):
                return parts[data_idx + 1]
        return "Uploaded"
    
    for uploaded_file in files:
        try:
            if isinstance(uploaded_file, str):
                with open(uploaded_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                file_name = os.path.basename(uploaded_file)
                source_path = uploaded_file
            else:
                data = json.load(uploaded_file)
                file_name = uploaded_file.name
                source_path = uploaded_file.name

            phase_name = infer_phase_name(source_path)
                
            requests = data.get("requests", [])
            if not isinstance(requests, list):
                requests = []
            
            # Try to identify user from file paths in the first few requests
            suspected_user = "Unknown User"
            for req in requests:
                if not isinstance(req, dict):
                    continue
                if suspected_user != "Unknown User":
                    break
                
                # Check variableData for file paths
                variable_data = req.get("variableData", {})
                if not isinstance(variable_data, dict):
                    variable_data = {}
                variables = variable_data.get("variables", [])
                if not isinstance(variables, list):
                    variables = []
                for var in variables:
                    if not isinstance(var, dict):
                        continue
                    val = var.get("value", {})
                    if not isinstance(val, dict):
                        continue
                    path = val.get("fsPath") or val.get("path")
                    if path:
                        # Regex for Windows/Mac/Linux home directories
                        match = re.search(r'[/\\](?:Users|home)[/\\]([^/\\\\]+)', path, re.IGNORECASE)
                        if match:
                            suspected_user = match.group(1)
                            break
            
            for req in requests:
                if not isinstance(req, dict):
                    continue
                timestamp = req.get("timestamp")
                if not timestamp:
                    continue
                
                # Convert ms to datetime
                dt = datetime.fromtimestamp(timestamp / 1000.0)
                
                # User Message
                message_obj = req.get("message", {})
                if not isinstance(message_obj, dict):
                    message_obj = {}
                user_msg = message_obj.get("text", "")
                
                # Assistant Response
                response_parts = req.get("response", [])
                if not isinstance(response_parts, list):
                    response_parts = []
                assistant_msg = ""
                model_name = "Unknown"
                metrics = {}
                
                for part in response_parts:
                    if not isinstance(part, dict):
                        continue
                    # Extract model info if available in result (not here per se, handled below)
                    
                    val = part.get("value")
                    if val and isinstance(val, str):
                        if part.get("kind") == "thinking":
                            continue # Skip thinking blocks for now
                        assistant_msg += val
                
                # Extract metadata from the request object directly as per example
                result = req.get("result", {})
                if not isinstance(result, dict):
                    result = {}
                timings = result.get("timings", {}) # Get timings
                if not isinstance(timings, dict):
                    timings = {}
                
                if result:
                    details = result.get("details", "")
                    if details:
                        model_name = details
                    
                    usage = result.get("usage", {})
                    if not isinstance(usage, dict):
                        usage = {}
                    metrics = usage
                
                # Context Files
                referenced_files = []
                variable_data = req.get("variableData", {})
                if not isinstance(variable_data, dict):
                    variable_data = {}
                variables = variable_data.get("variables", [])
                if not isinstance(variables, list):
                    variables = []
                for var in variables:
                    if not isinstance(var, dict):
                        continue
                    val = var.get("value", {})
                    if not isinstance(val, dict):
                        continue
                    # Try to get file name from path
                    path = val.get("fsPath") or val.get("path")
                    if path:
                        try:
                            referenced_file_name = os.path.basename(path)
                            referenced_files.append(referenced_file_name)
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
                    "file_name": file_name,
                    "suspected_user": suspected_user,
                    "latency_ms": timings.get("totalElapsed", 0),
                    "ttft_ms": timings.get("firstProgress", 0), # Time to First Token ~ Thinking Time
                    "referenced_files": referenced_files,
                    "languages": languages,
                    "phase": phase_name,
                    "source_path": source_path,
                    "edited_file_events": len(req.get("editedFileEvents", []) or []) if isinstance(req.get("editedFileEvents", []), list) else 0,
                    "checkpoints_restored": 1 if any(isinstance(p, dict) and p.get("kind") == "undoStop" for p in response_parts) else 0
                })
                
        except Exception as e:
            file_ref = uploaded_file if isinstance(uploaded_file, str) else uploaded_file.name
            st.error(f"Error parsing {file_ref}: {e}")
            
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

def analyze_prompt_style(df):
    """
    Analyzes user prompts and assigns rankings & descriptors.
    """
    results = []
    
    for idx, row in df.iterrows():
        text = str(row.get('user_text', '')).strip()
        if not text:
            continue
            
        word_count = len(text.split())
        char_count = len(text)
        
        # Descriptors mappings
        descriptors = []
        
        # Tone/Style heuristics
        text_lower = text.lower()
        if '?' in text or any(w in text_lower.split() for w in ['how', 'what', 'why', 'where', 'when', 'who', 'explain']):
            descriptors.append("Inquisitive")
            
        if any(w in text_lower for w in ['please', 'thanks', 'thank you', 'could you', 'would you']):
            descriptors.append("Polite")
            
        if any(text_lower.startswith(w) for w in ['create', 'make', 'write', 'generate', 'add', 'update', 'fix']):
             if word_count < 15:
                 descriptors.append("Direct/Commanding")
             else:
                 descriptors.append("Detailed Request")
                 
        if any(w in text_lower for w in ['error', 'bug', 'fail', 'broken', 'issue', 'doesn\'t work']):
            descriptors.append("Troubleshooting")
            
        # Add basic fallbacks
        if len(descriptors) == 0:
            if word_count < 5:
                descriptors.append("Succinct/Short")
            else:
                descriptors.append("Conversational")
                
        # Calculate a "Complexity Score" (1-10) based on length and unique words
        unique_words = len(set(text_lower.split()))
        complexity_score = min(10, max(1, int((unique_words / 15) + (char_count / 150))))
        prompt_length_bucket = "Short" if word_count < 12 else "Medium" if word_count < 30 else "Long"

        descriptor_set = set(descriptors)
        
        results.append({
            'Session': row.get('file_name', 'Unknown'),
            'Timestamp': row.get('timestamp', ''),
            'User': row.get('suspected_user', 'Unknown'),
            'Phase': row.get('phase', 'Uploaded'),
            'Prompt': text,
            'Word Count': word_count,
            'Complexity Score': complexity_score,
            'Prompt Length Bucket': prompt_length_bucket,
            'Style Descriptors': ", ".join(descriptors),
            'Is Inquisitive': "Inquisitive" in descriptor_set,
            'Is Polite': "Polite" in descriptor_set,
            'Is Direct': "Direct/Commanding" in descriptor_set,
            'Is Detailed': "Detailed Request" in descriptor_set,
            'Is Troubleshooting': "Troubleshooting" in descriptor_set
        })
        
    return pd.DataFrame(results)

# --- Main App Logic ---

files_to_parse = list(uploaded_files) if uploaded_files else []

for selected_phase in selected_phases:
    phase_dir = os.path.join(DATA_DIR, selected_phase)
    if not os.path.isdir(phase_dir):
        continue
    # Walk through User/Sessions directories
    for root, dirs, files in os.walk(phase_dir):
        # We only expect files in the sessions directory, but we can just grab all JSON files in the phase dir
        for file in files:
            if file.endswith(".json"):
                files_to_parse.append(os.path.join(root, file))

if files_to_parse:
    df_chat_all = parse_chat_data(files_to_parse)
    
    if not df_chat_all.empty:
        # Sort by time
        df_chat_all = df_chat_all.sort_values("timestamp")
        
        # --- Sidebar ---
        st.sidebar.divider()
        st.sidebar.subheader("Chat History View")
        
        # Create unique session labels with User where available
        # But we need a mapping back to file name for filtering
        session_map = {}
        # Get unique sessions ordered by First Chat Time (cronological)
        # df_chat_all is already sorted by timestamp, so the first occurrence is the start
        unique_sessions = df_chat_all[["file_name", "suspected_user", "timestamp"]].drop_duplicates(subset=["file_name"], keep='first')
        
        display_options = []
        for _, row in unique_sessions.iterrows():
            # Add timestamp to help understand the chronological order
            start_time_str = row['timestamp'].strftime('%Y-%m-%d %H:%M')
            label = f"{start_time_str} | {row['file_name']} ({row['suspected_user']})"
            display_options.append(label)
            session_map[label] = row['file_name']
            
        # display_options are now preserved in chronological order
        
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
        # Keep them in chronological order as well
        all_sessions = df_chat_all["file_name"].unique().tolist()
        
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
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Chat History", "Statistics", "Development Timeline", "Comparative Analysis", "Prompt Analysis"])
        
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
                                st.plotly_chart(fig_latency, width="stretch")
                                
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
                                    st.plotly_chart(fig_ttft, width="stretch")
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
                        st.plotly_chart(fig_langs, width="stretch")
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
                        st.plotly_chart(fig_files, width="stretch")
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
                            st.plotly_chart(fig_events, width="stretch")
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
                st.plotly_chart(fig_timeline, width="stretch")
                
                # Code Velocity
                st.write("### Code Velocity (AI Suggestions vs User Commits)")
                fig_velocity = px.line(df_timeline.sort_values("timestamp"), x="timestamp", y="code_volume", color="type", title="Code Volume Over Time")
                st.plotly_chart(fig_velocity, width="stretch")
                
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
                st.plotly_chart(fig_daily, width="stretch")

        with tab4:
            st.subheader("Comparative Analysis")
            
            prompt_df = analyze_prompt_style(df_chat_analysis)
            if prompt_df.empty:
                st.warning("No prompts available for phase comparison in current filters.")
            else:
                available_prompt_phases = sorted(prompt_df["Phase"].dropna().unique().tolist())
                if len(available_prompt_phases) < 2:
                    st.warning("Load at least two phases to use Phase-to-Phase comparison.")
                else:
                    col_phase_1, col_phase_2 = st.columns(2)
                    with col_phase_1:
                        phase_a = st.selectbox("Select Baseline Phase", options=available_prompt_phases, index=0)
                    with col_phase_2:
                        default_phase_idx = 1 if len(available_prompt_phases) > 1 else 0
                        phase_b = st.selectbox("Select Comparison Phase", options=available_prompt_phases, index=default_phase_idx)

                    if phase_a == phase_b:
                        st.info("Choose two different phases to compare.")
                    else:
                        df_phase_chat_a = df_chat_analysis[df_chat_analysis["phase"] == phase_a].copy()
                        df_phase_chat_b = df_chat_analysis[df_chat_analysis["phase"] == phase_b].copy()
                        df_phase_a = prompt_df[prompt_df["Phase"] == phase_a].copy()
                        df_phase_b = prompt_df[prompt_df["Phase"] == phase_b].copy()

                        def safe_mean(series):
                            return float(series.mean()) if len(series) > 0 else 0.0

                        def safe_rate(df, col):
                            return float(df[col].mean() * 100.0) if len(df) > 0 else 0.0

                        def phase_duration_days(df_phase_chat):
                            if df_phase_chat.empty:
                                return 0.0
                            start_ts = df_phase_chat["timestamp"].min()
                            end_ts = df_phase_chat["timestamp"].max()
                            duration = end_ts - start_ts
                            return max(duration.total_seconds() / 86400.0, 0.0)

                        def tokens_per_prompt(df_phase_chat):
                            if df_phase_chat.empty:
                                return 0.0
                            total_tokens = df_phase_chat["completion_tokens"].sum() + df_phase_chat["prompt_tokens"].sum()
                            return float(total_tokens / len(df_phase_chat)) if len(df_phase_chat) > 0 else 0.0

                        def error_rate_percent(df_phase_chat):
                            total_code, flagged_reverts = calculate_success_metrics(df_phase_chat)
                            if total_code <= 0:
                                return 0.0
                            return (flagged_reverts / total_code) * 100.0

                        def avg_interactions_per_day(df_phase_chat):
                            if df_phase_chat.empty:
                                return 0.0
                            duration_days = phase_duration_days(df_phase_chat)
                            normalized_days = max(duration_days, 1.0)
                            return len(df_phase_chat) / normalized_days

                        def top_model_share_percent(df_phase_chat):
                            if df_phase_chat.empty:
                                return 0.0, "N/A"
                            model_counts = df_phase_chat["model"].value_counts()
                            if model_counts.empty:
                                return 0.0, "N/A"
                            top_model = model_counts.index[0]
                            top_share = (model_counts.iloc[0] / model_counts.sum()) * 100.0
                            return top_share, top_model

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
                        summary_df = pd.DataFrame([
                            {
                                "Metric": "Phase Length (days)",
                                phase_a: f"{dur_a:.2f}",
                                phase_b: f"{dur_b:.2f}",
                            },
                            {
                                "Metric": "Interactions / Day",
                                phase_a: f"{int_per_day_a:.2f}",
                                phase_b: f"{int_per_day_b:.2f}",
                            },
                            {
                                "Metric": "Tokens / Prompt",
                                phase_a: f"{tokens_prompt_a:.1f}",
                                phase_b: f"{tokens_prompt_b:.1f}",
                            },
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
                        ])
                        st.dataframe(summary_df, width="stretch", hide_index=True)
                        st.divider()

                        descriptor_rows = [
                            {"Metric": "Inquisitive", phase_a: safe_rate(df_phase_a, "Is Inquisitive"), phase_b: safe_rate(df_phase_b, "Is Inquisitive")},
                            {"Metric": "Polite", phase_a: safe_rate(df_phase_a, "Is Polite"), phase_b: safe_rate(df_phase_b, "Is Polite")},
                            {"Metric": "Direct/Commanding", phase_a: safe_rate(df_phase_a, "Is Direct"), phase_b: safe_rate(df_phase_b, "Is Direct")},
                            {"Metric": "Detailed Request", phase_a: safe_rate(df_phase_a, "Is Detailed"), phase_b: safe_rate(df_phase_b, "Is Detailed")},
                            {"Metric": "Troubleshooting", phase_a: safe_rate(df_phase_a, "Is Troubleshooting"), phase_b: safe_rate(df_phase_b, "Is Troubleshooting")},
                        ]
                        df_descriptor_rates = pd.DataFrame(descriptor_rows).melt(
                            id_vars=["Metric"],
                            var_name="Phase",
                            value_name="Rate",
                        )
                        fig_descriptor_rates = px.bar(
                            df_descriptor_rates,
                            x="Metric",
                            y="Rate",
                            color="Phase",
                            barmode="group",
                            title="Descriptor Rates by Phase (%)",
                        )
                        st.plotly_chart(fig_descriptor_rates, width="stretch")

                        model_dist = pd.concat([
                            df_phase_chat_a.assign(PhaseLabel=phase_a),
                            df_phase_chat_b.assign(PhaseLabel=phase_b),
                        ])
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
                            df_complexity_compare = pd.concat([
                                df_phase_a.assign(PhaseLabel=phase_a),
                                df_phase_b.assign(PhaseLabel=phase_b),
                            ])
                            fig_complexity = px.box(
                                df_complexity_compare,
                                x="PhaseLabel",
                                y="Complexity Score",
                                color="PhaseLabel",
                                title="Complexity Distribution by Phase",
                            )
                            st.plotly_chart(fig_complexity, width="stretch")

                        with col_dist_2:
                            df_len_bucket = pd.concat([
                                df_phase_a.assign(PhaseLabel=phase_a),
                                df_phase_b.assign(PhaseLabel=phase_b),
                            ])
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
                        focus_cols = ["Phase", "Session", "User", "Timestamp", "Word Count", "Complexity Score", "Prompt"]
                        focus_df = pd.concat([
                            df_phase_a[df_phase_a["Is Troubleshooting"]].head(10),
                            df_phase_b[df_phase_b["Is Troubleshooting"]].head(10),
                        ])[focus_cols].sort_values("Timestamp", ascending=False)
                        st.dataframe(focus_df, width="stretch", hide_index=True)

        with tab5:
            st.subheader("Prompt Analysis & Styling")
            st.write("This tab takes in all the extracted user prompts and categorizes them based on their tone, length, and style descriptors.")
            
            if df_chat_analysis.empty:
                 st.warning("Please select at least one session in 'Analysis Filters' to view prompt analysis.")
            else:
                prompt_df = analyze_prompt_style(df_chat_analysis)
                
                if not prompt_df.empty:
                    # Summary metrics
                    p_col1, p_col2, p_col3 = st.columns(3)
                    p_col1.metric("Total Prompts Analyzed", len(prompt_df))
                    p_col2.metric("Average Word Count", f"{prompt_df['Word Count'].mean():.1f}")
                    p_col3.metric("Average Complexity Score", f"{prompt_df['Complexity Score'].mean():.1f}")
                    
                    st.divider()
                    
                    # Display the spreadsheet-like table
                    st.write("### Prompt Repository")
                    st.dataframe(
                        prompt_df, 
                        width="stretch", 
                        hide_index=True,
                        column_config={
                            "Prompt": st.column_config.TextColumn(width="large"),
                            "Complexity Score": st.column_config.ProgressColumn(format="%d", min_value=1, max_value=10)
                        }
                    )
                    
                    st.divider()
                    
                    # Descriptors Distribution
                    st.write("### Tone & Style Breakdown")
                    
                    # Flatten the descriptors column
                    all_desc = []
                    for desc in prompt_df['Style Descriptors']:
                        if desc:
                            all_desc.extend([d.strip() for d in desc.split(',')])
                            
                    if all_desc:
                        desc_counts = pd.Series(all_desc).value_counts().reset_index()
                        desc_counts.columns = ["Descriptor", "Count"]
                        
                        fig_desc = px.bar(desc_counts, x="Count", y="Descriptor", orientation='h', title="Distribution of Prompt Tone/Style")
                        fig_desc.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig_desc, width="stretch")
                else:
                    st.info("No text prompts found in the selected sessions.")

    else:
        st.warning("No valid chat requests found in the selected files.")
else:
    st.info("Please select one or more phases or upload a 'chatTemplate.json' file to begin.")
