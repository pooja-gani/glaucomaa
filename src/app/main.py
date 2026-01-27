import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import os
import sys
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add project root to path
sys.path.append(os.getcwd())

from src.models.segmentation import GlaucomaSegmentationModel
from src.models.classification import GlaucomaClassifier
from src.utils.metrics import GlaucomaMetrics
from src.utils.roi import extract_disc_roi
from src.agents.graph import build_agent_graph

# Page config
st.set_page_config(
    page_title="Glaucoma Screening Agent",
    page_icon="👁️",
    layout="wide"
)

@st.cache_resource
def load_models():
    """Load models with error handling for missing weights."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Segmentation (SegFormer - Pretrained)
    seg_model = GlaucomaSegmentationModel()
    seg_ready = False
    
    # Check for fine-tuned weights, else use pretrained
    if os.path.exists("checkpoints/segmentation_best.pth"):
        seg_model.load_state_dict(torch.load("checkpoints/segmentation_best.pth", map_location=device))
        seg_ready = True
    else:
        # Warn user we are using base pretrained model
        # st.sidebar.warning("⚠️ Using base Pretrained SegFormer (not fine-tuned).")
        seg_ready = True
        
    seg_model.to(device)
    seg_model.eval()
    
    # Classification (SwinV2 - Fine-tuned)
    cls_model = GlaucomaClassifier(use_features=False)
    if os.path.exists("checkpoints/classification_best.pth"):
        cls_model.load_state_dict(torch.load("checkpoints/classification_best.pth", map_location=device))
        st.sidebar.success("✅ Classification Model Loaded (Fine-tuned)")
    else:
        st.warning("⚠️ Classification weights not found. Using initialized SwinV2 base.")
    
    cls_model.to(device)
    cls_model.eval()
    
    return seg_model, cls_model, device, seg_ready

def get_transforms():
    """Returns dictionary of transforms for different models."""
    norm = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    
    return {
        # SegFormer expects 512x512
        "seg": A.Compose([
            A.Resize(512, 512),
            norm,
            ToTensorV2()
        ]),
        # SwinV2 Tiny expects 256x256
        "cls": A.Compose([
            A.Resize(256, 256),
            norm,
            ToTensorV2()
        ])
    }

def main():
    st.title("👁️ Automated Agentic AI for Glaucoma Screening")
    st.markdown("---")
    
    # Sidebar: Setup & Metadata
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Groq API Key Input
        api_key_input = st.text_input("Groq API Key (Optional)", type="password", help="Enter your Groq API key here to enable Agentic Reasoning.")
        if api_key_input:
            os.environ["GROQ_API_KEY"] = api_key_input
            
        # Model Selection
        model_options = [
            "llama-3.3-70b-versatile",
            "deepseek-r1-distill-llama-70b",
            "mixtral-8x7b-32768",
            "llama-3.1-8b-instant",
            "gemma2-9b-it"
        ]
        selected_model = st.selectbox("Select Reasoning Model", model_options, index=0)
        os.environ["GROQ_MODEL"] = selected_model
            
        st.divider()
        
        st.header("Patient Data")
        age = st.number_input("Age", min_value=0, max_value=120, value=65)
        iop = st.number_input("Intraocular Pressure (mmHg)", 0, 60, 15)
        family_history = st.selectbox("Family History of Glaucoma", ["No", "Yes"])
        diabetes = st.selectbox("Diabetes Status", ["No", "Yes"])
        
    # Main Area: Image Upload
    uploaded_file = st.file_uploader("Upload Retinal Fundus Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Load models
        seg_model, cls_model, device, seg_ready = load_models()
        transforms = get_transforms()
        
        # Display Image
        image = Image.open(uploaded_file).convert('RGB')
        img_np = np.array(image)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Fundus Image", width="stretch")
            
        with st.spinner("Analyzing..."):
            
            # --- Segmentation Inference ---
            vcdr = 0.0
            metrics = {}
            seg_map_crop = None
            
            if seg_ready:
                # 1. Extract ROI (512x512 around brightest spot)
                roi_img, (start_x, start_y) = extract_disc_roi(img_np, crop_size=512)
                
                # 2. Preprocess ROI
                aug_seg = transforms["seg"](image=roi_img)
                input_seg = aug_seg['image'].unsqueeze(0).to(device)
                
                with torch.no_grad():
                    preds, probs = seg_model.predict_step(input_seg)
                    seg_map_crop = preds[0].cpu().numpy() # (512, 512)
                
                # 3. Metrics (calculated on ROI is fine for ratios)
                metrics = GlaucomaMetrics.process_segmentation_output(seg_map_crop)
                vcdr = metrics["vCDR"]
                
                # 4. Visualize
                # Overlay on ROI
                vis_roi = cv2.resize(roi_img, (512, 512))
                color_mask = np.zeros_like(vis_roi)
                color_mask[seg_map_crop == 1] = [0, 255, 0] # Disc
                color_mask[seg_map_crop == 2] = [255, 0, 0] # Cup
                
                overlay_roi = cv2.addWeighted(vis_roi, 0.7, color_mask, 0.3, 0)
                
                # Optionally: Paste back to original image? 
                # For UI, showing the zoomed ROI is actually better for inspection.
                
                with col2:
                    st.image(overlay_roi, caption=f"Optic Disc ROI (vCDR: {vcdr:.2f})", width="stretch")
                    if vcdr == 0.0:
                        st.info("Still inconclusive? The ROI detection might have missed the disc.")
            else:
                 with col2:
                     st.warning("Segmentation unavailable.")
            
            # --- Classification Inference ---
            # Prepare Input: 256x256
            aug_cls = transforms["cls"](image=img_np)
            input_cls = aug_cls['image'].unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = cls_model(input_cls) # (1, 2)
                probs = torch.softmax(logits, dim=1)
                prob_g = probs[0, 1].item()
            
            # --- Agentic Reasoning ---
            st.subheader("🤖 Agentic Diagnosis")
            
            state_input = {
                "image_path": uploaded_file.name,
                "patient_metadata": {
                    "Age": age,
                    "IOP": iop,
                    "Family_History": family_history,
                    "Diabetes": diabetes
                },
                "segmentation_metrics": metrics,
                "glaucoma_probability": prob_g,
                "segmentation_available": seg_ready,
                "needs_review": False
            }
            
            if "GROQ_API_KEY" not in os.environ or not os.environ["GROQ_API_KEY"]:
                 st.error("GROQ_API_KEY not found. Please enter it in the sidebar to enable Agentic Reasoning.")
                 st.info(f"Model Probability for Glaucoma: {prob_g:.2%}")
            else:
                 try:
                     agent_graph = build_agent_graph() 
                     result = agent_graph.invoke(state_input)
                     
                     st.markdown("### 📋 Final Clinical Report")
                     st.success(result.get("final_report"))
                     
                     with st.expander("See Reasoning Trace"):
                         st.markdown(f"**Vision Agent:** {result.get('vision_analysis')}")
                         st.markdown(f"**Risk Agent:** {result.get('risk_analysis')}")
                         st.markdown(f"**Diagnostic Logic:** {result.get('diagnostic_reasoning')}")
                 except Exception as e:
                     st.error(f"Agent execution failed: {e}")

if __name__ == "__main__":
    main()
