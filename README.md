# Copilot History Analyzer

This is a simple Streamlit web application to analyze your GitHub Copilot chat history.

## Features

- **Recreate Chat UI**: Visualize your chat sessions in a familiar chat interface.
- **Statistics**: View AI vs Human contribution metrics (tokens, code lines).
- **Development Timeline**: Correlate chat activity with your local Git repository history.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    (Or `pip install streamlit pandas plotly gitpython`)

2.  **Run the App**:
    ```bash
    streamlit run app.py
    ```

3.  **Usage**:
    - Upload your `chatTemplate.json` file (usually found in your project workspace or VS Code logs).
    - Optionally, enter the full path to your local Git repository to see commit statistics overlay.
