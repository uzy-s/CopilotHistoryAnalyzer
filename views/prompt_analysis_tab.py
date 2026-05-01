import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import json

from views.shared import *

def render_prompt_analysis_tab(df_chat_analysis: pd.DataFrame) -> None:
    """Render the Prompt Analysis tab.

    Args:
        df_chat_analysis: Session-filtered chat dataframe for style analysis.

    Returns:
        None. Renders Streamlit UI elements directly.
    """
    st.subheader("Prompt Analysis & Styling")
    st.write("This tab takes in all the extracted user prompts and categorizes them based on their tone, length, and style descriptors.")

    if df_chat_analysis.empty:
        st.warning("Please select at least one session in 'Analysis Filters' to view prompt analysis.")
        return

    prompt_df = analyze_prompt_style(df_chat_analysis)
    if prompt_df.empty:
        st.info("No text prompts found in the selected sessions.")
        return

    # Top-level prompt dataset stats.
    p_col1, p_col2, p_col3 = st.columns(3)
    p_col1.metric("Total Prompts Analyzed", len(prompt_df))
    p_col2.metric("Average Word Count", f"{prompt_df['Word Count'].mean():.1f}")
    p_col3.metric("Average Complexity Score", f"{prompt_df['Complexity Score'].mean():.1f}")

    st.divider()

    st.write("### Prompt Repository")
    # Wide table view for row-level inspection and sorting/filtering in UI.
    st.dataframe(
        prompt_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Prompt": st.column_config.TextColumn(width="large"),
            "Complexity Score": st.column_config.ProgressColumn(format="%d", min_value=1, max_value=10),
        },
    )

    st.divider()
    st.write("### Tone & Style Breakdown")

    # Flatten descriptor strings into a frequency distribution.
    all_desc = []
    for desc in prompt_df["Style Descriptors"]:
        if desc:
            all_desc.extend([d.strip() for d in desc.split(",")])

    if all_desc:
        desc_counts = pd.Series(all_desc).value_counts().reset_index()
        desc_counts.columns = ["Descriptor", "Count"]

        fig_desc = px.bar(desc_counts, x="Count", y="Descriptor", orientation="h", title="Distribution of Prompt Tone/Style")
        fig_desc.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_desc, width="stretch")



