from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import vision_agent, risk_agent, hvf_agent, diagnostic_agent, report_agent

def build_agent_graph():
    """
    Constructs the StateGraph for Glaucoma Screening.
    Structure:
       Start -> Vision -> Risk -> HVF -> Diagnostic -> Report -> End
    """
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("vision_agent", vision_agent)
    workflow.add_node("risk_agent", risk_agent)
    workflow.add_node("hvf_agent", hvf_agent)
    workflow.add_node("diagnostic_agent", diagnostic_agent)
    workflow.add_node("report_agent", report_agent)
    
    workflow.set_entry_point("vision_agent") 
    
    workflow.add_edge("vision_agent", "risk_agent")
    workflow.add_edge("risk_agent", "hvf_agent")
    workflow.add_edge("hvf_agent", "diagnostic_agent")
    workflow.add_edge("diagnostic_agent", "report_agent")
    workflow.add_edge("report_agent", END)
    
    return workflow.compile()
