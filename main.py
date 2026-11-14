import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
from graph import JobAnalyzerWorkflow

# Load environment variables from .env
load_dotenv()

app = FastAPI(title="AI Job Description Analyzer")
templates = Jinja2Templates(directory="templates")

class AnalysisRequest(BaseModel):
    job_description: str
    api_key: Optional[str] = None

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Render the main light glassmorphism UI."""
    return templates.TemplateResponse(request, "index.html", {"request": request})

@app.post("/api/analyze")
async def analyze_job(payload: AnalysisRequest):
    """API endpoint triggered via JS fetch to run the LangGraph pipeline."""
    if not payload.job_description.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty.")
    
    try:
        api_key = payload.api_key or os.getenv("GEMINI_API_KEY")
        workflow = JobAnalyzerWorkflow(api_key=api_key)
        result = workflow.run(payload.job_description)
        
        if result.get("error"):
            raise HTTPException(status_code=400, detail=result["error"])
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)