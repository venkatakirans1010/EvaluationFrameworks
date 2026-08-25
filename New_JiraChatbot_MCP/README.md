# Jira Metrics Chatbot (Streamlit + FastAPI)

This is a basic full-stack application that lets you query Jira metrics in natural language (e.g., "How many bugs are there in the sprint?") using Atlassian MCP APIs. The backend is built with FastAPI, and the frontend uses Streamlit.

## Project Structure

- `backend/` — FastAPI backend (API server)
- `frontend/` — Streamlit frontend (UI)
- `requirements.txt` — Python dependencies

## Prerequisites

- Python 3.8+
- (Recommended) [Create a virtual environment](https://docs.python.org/3/library/venv.html)

## Setup & Run

1. **Install dependencies**

   ```sh
   pip install -r requirements.txt
   ```

2. **Start the FastAPI backend**

   ```sh
   uvicorn backend.main:app --reload
   ```
   - The API will be available at [http://localhost:8000/docs](http://localhost:8000/docs)

3. **Start the Streamlit frontend** (in a new terminal)

   ```sh
   streamlit run frontend/app.py
   ```
   - The UI will be available at [http://localhost:8501](http://localhost:8501)

4. **Usage**
   - Enter your natural language Jira query in the Streamlit UI.
   - The frontend sends the query to the FastAPI backend, which calls Atlassian MCP/Jira APIs and returns the result.

## Configuration

- You may need to set your Atlassian/Jira credentials and MCP server URL in `backend/config.py` or as environment variables (see backend code for details).

## Notes
- This is a minimal starter. You can expand the backend to support more complex queries, authentication, etc.
- For production, consider using Docker and securing credentials properly.

## License
MIT 