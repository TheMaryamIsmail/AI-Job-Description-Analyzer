from typing import Optional
from langgraph.graph import StateGraph, END
from state import JobState
from nodes import JobAnalyzerNodes

class JobAnalyzerWorkflow:
    """Object-Oriented wrapper to compile and execute the LangGraph pipeline."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.nodes = JobAnalyzerNodes(api_key=api_key)
        self.app = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(JobState)

        # Register nodes
        workflow.add_node("extraction", self.nodes.extract_info)
        workflow.add_node("classification", self.nodes.classify_job)
        workflow.add_node("skill_analysis", self.nodes.analyze_skills)
        workflow.add_node("hiring_report", self.nodes.generate_report)

        # Set entry point and conditional/sequential edges
        workflow.set_entry_point("extraction")

        def check_error(state: JobState):
            return "end" if state.error else "continue"

        workflow.add_conditional_edges(
            "extraction",
            check_error,
            {"end": END, "continue": "classification"}
        )
        workflow.add_edge("classification", "skill_analysis")
        workflow.add_edge("skill_analysis", "hiring_report")
        workflow.add_edge("hiring_report", END)

        return workflow.compile()

    def run(self, job_description: str) -> dict:
        initial_state = JobState(job_description=job_description)
        result = self.app.invoke(initial_state)
        return result if isinstance(result, dict) else result.__dict__