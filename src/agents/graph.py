from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import vision_agent, risk_agent, diagnostic_agent, report_agent

def build_agent_graph():
    """
    Constructs the StateGraph for Glaucoma Screening.
    Structure:
       Start -> [Vision, Risk] --(parallel)--> Diagnostic -> Report -> End
    """
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("vision_agent", vision_agent)
    workflow.add_node("risk_agent", risk_agent)
    workflow.add_node("diagnostic_agent", diagnostic_agent)
    workflow.add_node("report_agent", report_agent)
    
    # Define edges
    # Parallel execution of Vision and Risk
    workflow.set_entry_point("vision_agent") 
    # Actually LangGraph usually supports parallel branches by branching from start.
    # We can add a dummy start or just set entry point. 
    # Since we want Vision and Risk to run effectively in parallel or sequence, 
    # let's modify graph to: Start -> Vision -> Risk -> Diagnostic
    # Or strict parallel: Start -> Split -> ...
    # Simple linear for now: Vision -> Risk -> Diagnostic -> Report
    # Note: 'Risk' doesn't depend on 'Vision', so order doesn't matter between them.
    
    workflow.add_edge("vision_agent", "risk_agent")
    workflow.add_edge("risk_agent", "diagnostic_agent")
    workflow.add_edge("diagnostic_agent", "report_agent")
    workflow.add_edge("report_agent", END)
    
    return workflow.compile()
