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
    Risk Agent: Evaluates patient metadata/epidemiology.
    """
    meta = state.get("patient_metadata", {})
    
    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert Epidemiology Risk Agent for Glaucoma.
        
        Analyze the patient risk profile:
        - Age: {age}
        - IOP (Intraocular Pressure): {iop} mmHg
        - Family History: {family_history}
        - Diabetes: {diabetes}
        
        High risk factors include: Age > 60, IOP > 21 mmHg, positive family history.
        
        Provide a concise risk assessment based *only* on these factors.
        """
    )
    
    llm = get_llm()
    chain = prompt | llm | StrOutputParser()
    analysis = chain.invoke({
        "age": meta.get("Age", "Unknown"),
        "iop": meta.get("IOP", "Unknown"),
        "family_history": meta.get("Family_History", "Unknown"),
        "diabetes": meta.get("Diabetes", "Unknown")
    })
    
    return {"risk_analysis": analysis}

def diagnostic_agent(state: AgentState) -> dict:
    """
    Diagnostic Agent: Synthesizes model probability with agent analyses.
    """
    prob = state.get("glaucoma_probability", 0.0)
    vision = state.get("vision_analysis", "")
    risk = state.get("risk_analysis", "")
    
    prompt = ChatPromptTemplate.from_template(
        """
        You are the Chief Diagnostic Agent. synthesize the evidence.
        
        Follow this strict TWO-STEP REASONING PROCESS:
        
        STEP 1: PRIMARY ASSESSMENT (IMAGE & VISION DATA)
        - Deep Learning Probability: {prob:.2%}
        - Vision Analysis: {vision}
        *Formulate your initial clinical impression based ONLY on the above.*
        
        STEP 2: SECONDARY REFINEMENT (PATIENT CONTEXT)
        - Risk Factors: {risk}
        *Adjust your suspicion level based on the patient's risk profile (Age, IOP, Family History).*
        
        Determine the final screening category:
        1. Normal / Low Risk
        2. Glaucoma Suspect (requires monitoring)
        3. High Risk / Glaucoma Likely (requires referral)
        
        Provide your reasoning trace (Step 1 -> Step 2) and final category.
        """
    )
    
    llm = get_llm()
    chain = prompt | llm | StrOutputParser()
    reasoning = chain.invoke({
        "prob": prob,
        "vision": vision,
        "risk": risk
    })
    
    return {"diagnostic_reasoning": reasoning}

def report_agent(state: AgentState) -> dict:
    """
    Report Agent: Generates final clinical report.
    """
    reasoning = state.get("diagnostic_reasoning", "")
    
    prompt = ChatPromptTemplate.from_template(
        """
        You are a Clinical Documentation Agent. Generate a patient screening report.
        
        DIAGNOSTIC REASONING:
        {reasoning}
        
        Format the report clearly for a clinician.
        
        Structure:
        - Patient Summary
        - Structural Findings (Vision) -> OMIT this section if vision analysis states "metrics could not be calculated" or "unavailable".
        - Risk Assessment (Metadata)
        - AI Model Prediction
        - CLINICAL IMPRESSION
        
        - DETAILED FUTURE DIAGNOSTIC WORKUP (Must Include):
          1. **Imaging**: Optical Coherence Tomography (OCT) of the Retinal Nerve Fiber Layer (RNFL) and Ganglion Cell Complex (GCC) to confirm structural damage.
          2. **Functional Testing**: Standard Automated Perimetry (HVF 24-2) to assess visual field loss. Consider 10-2 if advanced.
          3. **Clinical Assessment**: 
             - Gonioscopy to rule out angle-closure.
             - Pachymetry to measure Central Corneal Thickness (CCT).
             - Phasing IOP (diurnal measurement) if progression is suspected despite normal office IOP.

        - CLINICAL MANAGEMENT & MEDICATION PLAN:
          *If High Risk / Glaucoma Suspect:*
          1. **First-Line Therapy**: Consider initiating Prostaglandin Analogs (e.g., **Latanoprost 0.005%** q.h.s.) or Beta-Blockers (e.g., **Timolol 0.5%** b.i.d.) if contraindications permit.
          2. **Target IOP**: Aim for a 20-30% reduction from baseline.
          3. **Monitoring Schedule**:
             - Repeat IOP check in **4-6 weeks** after starting meds.
             - Repeat Visual Field & OCT every **4-6 months** to monitor stability.
          
          *If Normal / Low Risk:*
          1. Routine annual comprehensive eye exam (dilated fundus exam + IOP).
          2. Patient education on risk factors.
        
        Disclaimer: This is an AI screening tool, not a definitive diagnosis.
        """
    )
    
    llm = get_llm()
    chain = prompt | llm | StrOutputParser()
    report = chain.invoke({"reasoning": reasoning})
    
    return {"final_report": report}

