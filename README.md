# Copilot History Analyzer

Streamlit app for analyzing GitHub Copilot chat exports and comparing AI interaction activity with local Git history.

## What this app does

- Recreates chat sessions in a chat-style view.
- Computes usage and quality metrics from prompts/responses.
- Builds activity timelines from chat events and git commits.
- Compares behavior across phases (for example, Phase1 vs Phase 2).
- Analyzes prompt style (tone, complexity, troubleshooting patterns).

## Requirements

- Python 3.10+ (3.11 recommended)
- pip
- A Copilot chat export JSON with a requests array

## Quick start (Windows PowerShell)

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Configure the app password (required):

```powershell
$env:APP_PASSWORD = "your-password-here"
```

4. Run the app:

```powershell
streamlit run app.py
```

5. Open the local URL printed by Streamlit (usually http://localhost:8501) and sign in with the password from step 3.

## Password configuration options

The app checks credentials in this order:

1. Streamlit secrets: APP_PASSWORD
2. Environment variable: APP_PASSWORD

You can use either option:

- Environment variable (good for local dev).
- Streamlit secrets (good for shared environments):

Create .streamlit/secrets.toml:

```toml
APP_PASSWORD = "your-password-here"
```

## Input data options

You can load data in two ways:

1. Upload JSON files in the sidebar.
2. Select phase folders discovered under data/.

The app recursively scans selected phases for JSON files.

Expected chat export shape:

- Top-level requests list.
- Each request may include timestamp, message, response, result, and variableData.

## Optional git correlation

To include human commit stats, provide a local repository path in the sidebar.

The app extracts:

- Commit timestamp
- Author
- Commit message
- Insertions/deletions

## Project structure

Top-level runtime entry:

- app.py: Streamlit entrypoint and high-level orchestration.

Service layer:

- services/auth.py: Password gate and sign-in flow.
- services/data_processing.py: Chat/git parsing, file discovery, normalization.
- services/analytics.py: Reusable metrics and prompt-style analysis helpers.

View layer:

- views/tabs.py: One renderer function per Streamlit tab.

Data folder:

- data/: Local chat datasets organized by phase/user/session.

## App flow

1. App starts and enforces password authentication.
2. Sidebar gathers phase selection, uploaded files, and optional git path.
3. Chat JSON files are parsed into a normalized dataframe.
4. Session filters are applied for analysis scope.
5. Tab renderers consume prepared dataframes and display charts/tables.

## Troubleshooting

- "App password is not configured": Set APP_PASSWORD in env vars or .streamlit/secrets.toml.
- "No valid chat requests found": Confirm your JSON contains a requests array.
- Empty git stats: Confirm the provided path is a valid git repository root.
- PowerShell activation blocked: Run once as needed:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Development notes

- Keep parsing and analytics logic in services/.
- Keep UI-only logic in views/.
- Add new tabs by implementing a renderer in views/tabs.py and wiring it in app.py.
