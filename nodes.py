import os
import json
import time
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List, Optional
from state import JobState
from dotenv import load_dotenv

# .env file ko load karein
load_dotenv()

# Environment variables access karein
api_key = os.getenv("GEMINI_API_KEY")
db_url = os.getenv("DATABASE_URL")

class ExtractedInfo(BaseModel):
    job_title: str = Field(description="Title of the job position")
    company_name: Optional[str] = Field(description="Company offering the job, if available")
    required_experience: str = Field(description="Required experience years or level")
    required_skills: List[str] = Field(description="List of required skills mentioned")

class ClassificationInfo(BaseModel):
    job_category: str = Field(description="Job category e.g., AI, Web Development, Data Science, etc.")
    employment_type: str = Field(description="Employment type e.g., Full-time, Part-time, Internship, Remote, Hybrid, On-site")

class SkillAnalysisInfo(BaseModel):
    technical_skills: List[str] = Field(description="Categorized technical skills")
    soft_skills: List[str] = Field(description="Categorized soft skills")
    top_3_skills: List[str] = Field(description="Top 3 most important skills for the role")

class JobAnalyzerNodes:
    """Object-Oriented encapsulation for LangGraph processing nodes."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key)

    def extract_info(self, state: JobState) -> dict:
        """Node 1: Information Extraction with 503 Retry Handling"""
        if not state.job_description or not state.job_description.strip():
            return {"error": "Job description cannot be empty."}
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                prompt = f"""Extract the following information from the job description:
1. Job Title
2. Company Name (if available)
3. Required Experience
4. Required Skills

Job Description:
{state.job_description}
"""
                response = self.client.models.generate_content(
                    model='gemini-3.5-flash',
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ExtractedInfo,
                        temperature=0.1
                    ),
                )
                
                response_text = response.text if response.text else "{}"
                data = json.loads(response_text)
                
                return {
                    "job_title": data.get("job_title"),
                    "company_name": data.get("company_name"),
                    "required_experience": data.get("required_experience"),
                    "required_skills": data.get("required_skills", []),
                    "error": None
                }
            except Exception as e:
                # Agar 503 unavailable ya high demand ho toh retry karein
                error_str = str(e)
                if ("503" in error_str or "UNAVAILABLE" in error_str) and attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))  # 2s, phir 4s wait karega
                    continue
                return {"error": f"Extraction Error: {error_str}"}

    def classify_job(self, state: JobState) -> dict:
        """Node 2: Job Classification with 503 Retry Handling"""
        if state.error:
            return {}
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                prompt = f"""Classify the following job description:
1. Job Category (e.g., AI, Web Development, Data Science, DevOps, etc.)
2. Employment Type (e.g., Full-time, Part-time, Internship, Remote, Hybrid, On-site)

Job Title: {state.job_title}
Job Description:
{state.job_description}
"""
                response = self.client.models.generate_content(
                    model='gemini-3.5-flash',
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ClassificationInfo,
                        temperature=0.1
                    ),
                )
                
                response_text = response.text if response.text else "{}"
                data = json.loads(response_text)
                
                return {
                    "job_category": data.get("job_category"),
                    "employment_type": data.get("employment_type")
                }
            except Exception as e:
                error_str = str(e)
                if ("503" in error_str or "UNAVAILABLE" in error_str) and attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                    continue
                return {"error": f"Classification Error: {error_str}"}

    def analyze_skills(self, state: JobState) -> dict:
        """Node 3: Skill Analysis with 503 Retry Handling"""
        if state.error:
            return {}
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                prompt = f"""Analyze the skills for this job and generate:
1. Technical Skills
2. Soft Skills
3. Top 3 Most Important Skills

Job Title: {state.job_title}
Required Skills: {state.required_skills}
Job Description:
{state.job_description}
"""
                response = self.client.models.generate_content(
                    model='gemini-3.5-flash',
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=SkillAnalysisInfo,
                        temperature=0.1
                    ),
                )
                
                response_text = response.text if response.text else "{}"
                data = json.loads(response_text)
                
                return {
                    "technical_skills": data.get("technical_skills", []),
                    "soft_skills": data.get("soft_skills", []),
                    "top_3_skills": data.get("top_3_skills", [])
                }
            except Exception as e:
                error_str = str(e)
                if ("503" in error_str or "UNAVAILABLE" in error_str) and attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                    continue
                return {"error": f"Skill Analysis Error: {error_str}"}

    def generate_report(self, state: JobState) -> dict:
        """Node 4: Hiring Report Generation (Handled natively in Python)"""
        if state.error:
            return {}
        
        try:
            tech_str = ", ".join(state.technical_skills) if state.technical_skills else "N/A"
            soft_str = ", ".join(state.soft_skills) if state.soft_skills else "N/A"
            
            top_skills_lines = ""
            if state.top_3_skills:
                for idx, skill in enumerate(state.top_3_skills, 1):
                    top_skills_lines += f"{idx}. {skill}\n"
            else:
                top_skills_lines = "N/A\n"

            report = f"""========================================
           HIRING REPORT
========================================
Job Title: {state.job_title or 'N/A'}
Company: {state.company_name or 'N/A'}
Job Category: {state.job_category or 'N/A'}
Employment Type: {state.employment_type or 'N/A'}
Required Experience: {state.required_experience or 'N/A'}

--- SKILL BREAKDOWN ---
Technical Skills: {tech_str}
Soft Skills: {soft_str}

Top 3 Most Important Skills:
{top_skills_lines.strip()}
========================================"""
            return {"hiring_report": report}
        except Exception as e:
            return {"error": f"Report Generation Error: {str(e)}"}