# Automated Agentic AI Framework for Glaucoma Screening

This project implements an autonomous AI system for glaucoma screening using retinal fundus images, adhering to strict clinical and engineering requirements.

## Architecture

1. **Data Layer**:
   - Automated Kaggle download.
   - Clinical preprocessing (CLAHE, Resize 512x512).

2. **Model Layer**:
   - **Segmentation**: U-Net++ (ResNet-34) for Optic Disc/Cup.
   - **Classification**: DenseNet-121 for Glaucoma detection.
   - **Biomarkers**: Automatic CDR computation.

3. **Agentic Layer (LangGraph)**:
   - **Vision Agent**: Analyzes structural metrics.
   - **Risk Agent**: Evaluates patient metadata.
   - **Diagnostic Agent**: Synthesizes evidence.
   - **Report Agent**: Generates clinical reports.

4. **Interface**:
   - Streamlit web application.

## Usage

See `walkthrough.md` for detailed instructions on how to build and run via Docker.

## Project Structure
```
.
├── Dockerfile
├── requirements.txt
├── src
│   ├── agents   # LangGraph agents
│   ├── app      # Streamlit UI
│   ├── data     # Data processing
│   ├── models   # PyTorch models
│   └── utils    # Metrics
└── data         # Dataset directory (created at runtime)
```
