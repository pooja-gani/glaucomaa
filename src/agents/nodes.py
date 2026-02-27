import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .state import AgentState

def get_llm():
    """
    Lazy initialization of LLM to allow API Key injection at runtime.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")
        
    model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        
    return ChatGroq(
        model=model_name, 
        temperature=0.2,
        api_key=api_key
    )

def vision_agent(state: AgentState) -> dict:
    seg_avail = state.get("segmentation_available", True)
    
    if not seg_avail:
        return {"vision_analysis": "Structural analysis unavailable (Segmentation Model untrained/missing)."}
        
    metrics = state.get("segmentation_metrics", {})
    vcdr = metrics.get("vCDR", 0.0)
    disc_area = metrics.get("Disc_Area_Px", 0)
    
    # Check if segmentation actually worked (non-zero area)
    if not seg_avail or (vcdr == 0.0 and disc_area <= 1):
        return {"vision_analysis": "Structural metrics pending further analysis. (Segmentation metrics unavailable)."}
    
    # Clinical rule: vCDR > 0.6 is suspicious.
    # LLM helps contextualize this.
    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert Vision Analysis Agent for Glaucoma Screening.
        
        Analyze the following structural metrics derived from retinal fundus images:
        - Vertical Cup-to-Disc Ratio (vCDR): {vcdr:.2f}
        - Optic Disc Area (px): {disc_area}
        - Optic Cup Area (px): {cup_area}
        
        Rule: A vCDR > 0.6 is generally considered suspicious for glaucoma.
        
        Provide a concise analysis of the structural risk based *only* on these metrics.
        """
    )
    
    llm = get_llm()
    chain = prompt | llm | StrOutputParser()
    analysis = chain.invoke({
        "vcdr": vcdr,
        "disc_area": disc_area,
        "cup_area": metrics.get("Cup_Area_Px", 0)
    })
    
    return {"vision_analysis": analysis}

def risk_agent(state: AgentState) -> dict:
    """
    Risk Agent: Evaluates IOP and CCT.
    """
    meta = state.get("patient_metadata", {})
    
    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert clinical agent assessing glaucoma risk from IOP and Central Corneal Thickness (CCT).
        
        - IOP (Intraocular Pressure): {iop} mmHg
        - CCT (Central Corneal Thickness): {cct} μm
        
        Clinical context:
        - Normal IOP: 10–21 mmHg. Elevated: >21. Critically high: >30.
        - Normal CCT: ~520–540 μm. Thin CCT (<520 μm) is an independent risk factor — it means true IOP may be underestimated and the optic nerve is more vulnerable.
        - Thick CCT (>560 μm) may cause IOP to be overestimated (lower actual risk).
        
        Assess both values together. Note how CCT modifies interpretation of IOP. Do not reference age, family history, or any other factor.
        """
    )
    
    llm = get_llm()
    chain = prompt | llm | StrOutputParser()
    analysis = chain.invoke({
        "iop": meta.get("IOP", "Unknown"),
        "cct": meta.get("CCT", "Unknown"),
    })
    
    return {"risk_analysis": analysis}

def diagnostic_agent(state: AgentState) -> dict:
    """
    Diagnostic Agent: Synthesizes model probability with agent analyses.
    """
    prob = state.get("glaucoma_probability", 0.0)
    risk = state.get("risk_analysis", "")
    
    prompt = ChatPromptTemplate.from_template(
        """
        You are the Chief Diagnostic Agent. Synthesize the evidence using ONLY the two inputs below.
        
        STEP 1: FUNDUS IMAGE ASSESSMENT
        - Deep Learning Probability of glaucoma: {prob:.2%}
        *Formulate your primary clinical impression based solely on this probability.*
        
        STEP 2: IOP REFINEMENT
        - IOP Assessment: {risk}
        *Adjust your impression based on IOP. Do NOT factor in any other variable.*
        
        Determine the final screening category:
        1. Normal / Low Risk
        2. Glaucoma Suspect (requires monitoring)
        3. High Risk / Glaucoma Likely (requires referral)
        
        Provide your reasoning trace (Step 1 → Step 2) and final category.
        """
    )
    
    llm = get_llm()
    chain = prompt | llm | StrOutputParser()
    reasoning = chain.invoke({
        "prob": prob,
        "risk": risk
    })
    
    return {"diagnostic_reasoning": reasoning}

def report_agent(state: AgentState) -> dict:
    """
    Report Agent: Generates a patient-specific clinical report based on AI probability, IOP, and CCT only.
    """
    reasoning = state.get("diagnostic_reasoning", "")
    meta = state.get("patient_metadata", {})
    iop = meta.get("IOP", "Unknown")
    cct = meta.get("CCT", "Unknown")
    prob = state.get("glaucoma_probability", 0.0)

    prompt = ChatPromptTemplate.from_template(
        """You are a Clinical Documentation Agent generating a glaucoma screening report.

STRICT RULES — violating any of these is unacceptable:
- Do NOT mention vCDR, optic disc, cup-to-disc ratio, or any structural image metric.
- Do NOT mention age, family history, diabetes, or any factor not listed below.
- Do NOT say information is "missing" or "not provided".
- Do NOT use any pre-written drug list, fixed protocol, or template workup.
- Every sentence must be derived from the three values below. Generic filler is forbidden.

PATIENT VALUES FOR THIS REPORT:
- IOP (Intraocular Pressure): {iop} mmHg  [normal 10-21 | elevated 21-30 | critical >30]
- CCT (Central Corneal Thickness): {cct} μm  [thin <520 = true IOP likely higher | normal 520-540 | thick >560 = IOP may be overestimated]
- AI Glaucoma Probability: {prob:.1%}  [low <30% | moderate 30-70% | high >70%]

AGENT REASONING:
{reasoning}

Write exactly these 4 sections. No extra sections, no boilerplate:

**IOP & CCT ASSESSMENT**
IOP is {iop} mmHg — state whether it is normal, elevated, or critical. CCT is {cct} μm — does it suggest the true IOP is higher or lower than {iop}? State the adjusted clinical risk clearly.

**AI MODEL PREDICTION**
The AI probability is {prob:.1%} — classify as low, moderate, or high concern and explain what this means clinically for this patient.

**CLINICAL IMPRESSION**
Given IOP={iop} mmHg, CCT={cct} μm, and AI probability={prob:.1%}, assign a risk category: Normal / Glaucoma Suspect / High Risk. State which specific value is the primary driver.

**RECOMMENDED NEXT STEPS**
Write 2-4 specific actions based only on the above values. If IOP={iop} is elevated, calculate and state the target IOP (20-30% reduction from {iop}). Do not list generic tests — tie every recommendation to the actual numbers.

*This is an AI screening tool. Final diagnosis requires a qualified ophthalmologist.*"""
    )

    llm = ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), temperature=0.5, api_key=os.getenv("GROQ_API_KEY"))
    chain = prompt | llm | StrOutputParser()
    report = chain.invoke({
        "reasoning": reasoning,
        "iop": iop,
        "cct": cct,
        "prob": prob,
    })

    return {"final_report": report}

