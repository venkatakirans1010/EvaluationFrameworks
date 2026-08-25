# POC: UI test case generator from Jira ID + uploaded docs (RAG); uses Gemini 2.5 Flash.

## Project Structure

```
ai_ui_testcase_creator/
├── backend/          # FastAPI backend
├── frontend/         # React frontend (Vite)
└── README.md
```

## Setup Instructions

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the `backend/` directory with your credentials:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   JIRA_URL=https://your-jira-instance.atlassian.net
   JIRA_USER=your_email@example.com
   JIRA_API_TOKEN=your_jira_api_token
   ```

5. Run the backend server:
   ```bash
   uvicorn main:app --reload
   ```

The backend will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Run the development server:
   ```bash
   npm run dev
   ```

The frontend will be available at `http://localhost:3000`

## Usage

1. Start both backend and frontend servers
2. Open `http://localhost:3000` in your browser
3. Enter a Jira issue ID (e.g., `PROJ-123`)
4. Optionally upload supporting documents (PDF or TXT files)
5. Click "Generate Test Cases"
6. View the generated test cases and download the Excel file

## API Endpoints

- `POST /generate_test_cases` - Generate test cases from Jira ID and files
- `GET /health` - Health check endpoint
- `GET /download/{filename}` - Download generated Excel files

