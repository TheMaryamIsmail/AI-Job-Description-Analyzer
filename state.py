from typing import List, Optional
from pydantic import BaseModel, Field

class JobState(BaseModel):
    job_description: str = ""
    job_title: Optional[str] = None
    company_name: Optional[str] = None
    required_experience: Optional[str] = None
    required_skills: List[str] = Field(default_factory=list)
    job_category: Optional[str] = None
    employment_type: Optional[str] = None
    technical_skills: List[str] = Field(default_factory=list)
    soft_skills: List[str] = Field(default_factory=list)
    top_3_skills: List[str] = Field(default_factory=list)
    hiring_report: Optional[str] = None
    error: Optional[str] = None