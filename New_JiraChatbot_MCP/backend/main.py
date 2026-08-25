from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import httpx
import backend.config as config
import base64

app = FastAPI()

# Allow CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

# Helper to get Jira auth header
def get_jira_auth_headers():
    user_pass = f"{config.JIRA_EMAIL}:{config.JIRA_API_TOKEN}"
    b64 = base64.b64encode(user_pass.encode()).decode()
    return {
        "Authorization": f"Basic {b64}",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

async def get_bug_count_in_current_sprint():
    headers = get_jira_auth_headers()
    async with httpx.AsyncClient() as client:
        # Get boards
        boards_url = f"{config.JIRA_CLOUD_ID.rstrip('/')}/rest/agile/1.0/board"
        boards_resp = await client.get(boards_url, headers=headers)
        if boards_resp.status_code != 200:
            return f"Error fetching boards: {boards_resp.status_code} {boards_resp.text}"
        try:
            boards = boards_resp.json().get("values", [])
        except Exception:
            return f"Error parsing boards response: {boards_resp.text}"
        if not boards:
            return "No boards found."
        board_id = boards[0]["id"]
        # Get sprints for the board
        sprints_url = f"{config.JIRA_CLOUD_ID.rstrip('/')}/rest/agile/1.0/board/{board_id}/sprint?state=active"
        sprints_resp = await client.get(sprints_url, headers=headers)
        if sprints_resp.status_code != 200:
            return f"Error fetching sprints: {sprints_resp.status_code} {sprints_resp.text}"
        try:
            sprints = sprints_resp.json().get("values", [])
        except Exception:
            return f"Error parsing sprints response: {sprints_resp.text}"
        if not sprints:
            return "No active sprint found."
        sprint_id = sprints[0]["id"]
        # Get issues in the sprint
        issues_url = f"{config.JIRA_CLOUD_ID.rstrip('/')}/rest/agile/1.0/sprint/{sprint_id}/issue?jql=issuetype=Bug"
        issues_resp = await client.get(issues_url, headers=headers)
        if issues_resp.status_code != 200:
            return f"Error fetching issues: {issues_resp.status_code} {issues_resp.text}"
        try:
            issues = issues_resp.json().get("issues", [])
        except Exception:
            return f"Error parsing issues response: {issues_resp.text}"
        return f"There are {len(issues)} bugs in the current sprint."

async def run_jql_query(jql: str):
    headers = get_jira_auth_headers()
    search_url = f"{config.JIRA_CLOUD_ID.rstrip('/')}/rest/api/3/search"
    payload = {"jql": jql}
    async with httpx.AsyncClient() as client:
        resp = await client.post(search_url, headers=headers, json=payload)
        if resp.status_code != 200:
            return f"Error running JQL: {resp.status_code} {resp.text}"
        try:
            data = resp.json()
            total = data.get("total", 0)
            return f"JQL matched {total} issues."
        except Exception:
            return f"Error parsing JQL response: {resp.text}"

@app.post("/query", response_model=QueryResponse)
async def query_jira(request: QueryRequest):
    q = request.query.strip()
    # Heuristic: if looks like JQL, run as JQL
    if any(x in q for x in ["project =", "ORDER BY", "issuetype =", "status =", "resolution ="]):
        answer = await run_jql_query(q)
    elif "bug" in q.lower() and "sprint" in q.lower():
        answer = await get_bug_count_in_current_sprint()
    else:
        answer = f"Sorry, I can't answer: {request.query} (MCP integration needed)"
    return QueryResponse(answer=answer) 