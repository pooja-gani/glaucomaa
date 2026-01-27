from typing import TypedDict, Optional, Dict, List

class AgentState(TypedDict):
    """
    Shared state for the Glaucoma Screening Agent graph.
    """
    # Inputs
    image_path: str
    patient_metadata: Dict[str, float | str] # Age, IOP, cleanup, etc.
    
    # Model Outputs
    segmentation_metrics: Dict[str, float] # vCDR, areas
    glaucoma_probability: float
    
    # Agent Reasoning Outputs
    vision_analysis: Optional[str]
    risk_analysis: Optional[str]
    diagnostic_reasoning: Optional[str]
    final_report: Optional[str]
    
    # Flags
    needs_review: bool
    segmentation_available: bool
