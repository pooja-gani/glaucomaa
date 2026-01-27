# Requirements Specification – Automated Agentic AI Framework for Glaucoma Screening

## 1. Project Overview

This document defines the complete functional, non-functional, system, and research requirements for implementing the **Automated Agentic AI Framework for Glaucoma Screening Using Retinal Fundus Images**, as described in the REU project report.

The system aims to provide **end-to-end, interpretable, and clinically aligned glaucoma screening** by integrating:

* Deep learning–based retinal image analysis
* Clinically grounded biomarker extraction
* Multi-agent AI reasoning using LLMs
* Explainable diagnostic reporting

---

## 2. Objectives

* Automated detection of glaucoma from retinal fundus images
* Accurate optic disc and optic cup segmentation
* Extraction of clinically relevant biomarkers (e.g., CDR)
* Robust glaucoma risk classification
* Human-aligned, explainable diagnostic reasoning using agentic AI
* Generation of transparent, clinician-friendly diagnostic reports
* Scalability for large-scale screening in resource-constrained settings

---

## 3. System Architecture Requirements

> **Execution Environment Constraint**

* The system **must be compatible with the Antigravity platform** (sandboxed execution, containerized workloads, reproducible environments).
* All components must run without persistent local state unless explicitly stored via Antigravity-supported volumes or object storage.
* External API calls must be explicitly configurable via environment variables.

### 3.1 High-Level Architecture

The system shall follow a **modular pipeline architecture** consisting of:

1. Data ingestion and preprocessing
2. Optic disc & optic cup segmentation module
3. Biomarker computation module
4. Glaucoma classification module
5. Agentic AI reasoning layer
6. Diagnostic report generation module
7. User interface layer

Each module must be independently upgradable without breaking downstream components.

---

## 4. Input Requirements

### 4.0 Dataset Source (Mandatory)

**Dataset:** `Glaucoma Detection Dataset` from Kaggle

* Source: Kaggle public dataset containing retinal fundus images labeled as glaucomatous / non-glaucomatous
* Modalities:

  * Color fundus photographs (RGB)
* Labels:

  * Binary classification: `Glaucoma`, `Normal`
  * Some subsets include optic disc annotations (used when available)

### 4.1 Dataset Access Requirements

* Dataset **must be downloaded programmatically using Kaggle API**
* Kaggle credentials (`kaggle.json`) must be injected via Antigravity environment variables
* No manual uploads or local file dependencies are permitted

### 4.2 Dataset Directory Structure (Post-Download)

```
data/
├── raw/
│   ├── glaucoma/
│   └── normal/
├── processed/
│   ├── images/
│   ├── masks_disc/
│   └── masks_cup/
├── splits/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
```

Dataset splitting must be deterministic and seed-controlled.

### 4.0 Dataset Source Requirement

* The primary dataset **must be downloaded programmatically from Kaggle**.
* Dataset: *Glaucoma Detection Dataset* (Kaggle)
* Access shall use Kaggle API credentials (`kaggle.json`) injected securely via environment variables (no hardcoded secrets).
* Dataset download must be reproducible inside Antigravity containers.

### 4.1 Image Data

* RGB retinal fundus images
* Supported resolution: variable (up to 2000×2000), resized internally to 512×512
* Image format: JPG / PNG
* Single-eye image per inference

### 4.2 Patient Metadata (Structured Input)

The system shall accept the following optional but recommended metadata:

* Age
* Eye laterality (left / right)
* Intraocular pressure (IOP)
* Central corneal thickness
* Family history of glaucoma (binary)
* Diabetes status (binary)

Metadata must be validated and normalized before reasoning.

---

## 5. Preprocessing Requirements

### 5.1 Image Standardization Pipeline

Each fundus image must undergo the following deterministic preprocessing steps:

1. Resize to `512 × 512`
2. Center-crop around optic nerve head (ONH)
3. Illumination normalization using CLAHE
4. Gamma correction (γ ∈ [0.8, 1.2])
5. Pixel normalization to ImageNet mean/std

### 5.2 Augmentation (Training Only)

* Random rotation (±15°)
* Horizontal flip
* Brightness & contrast jitter
* Gaussian noise (low variance)

Augmentations must be disabled during validation and inference.

* Gamma correction
* Contrast Limited Adaptive Histogram Equalization (CLAHE)
* Illumination normalization
* Optic nerve head localization
* Spatial resizing to 512×512
* Data augmentation (rotation, flip, brightness)

The preprocessing pipeline must be reproducible and configurable.

---

## 6. Segmentation Module Requirements

### 6.1 Model Architecture

* Base architecture: **U-Net++**
* Encoder backbone: ResNet-34 (pretrained on ImageNet)
* Output channels: 3 (background, optic disc, optic cup)

### 6.2 Training Requirements

* Loss function:

  * Dice Loss + Cross-Entropy (weighted)
* Optimizer: AdamW
* Learning rate: 1e-4 with cosine decay
* Batch size: 8–16 (GPU dependent)

### 6.3 Inference Output

* Binary optic disc mask
* Binary optic cup mask
* Soft probability maps (for uncertainty estimation)

### 6.1 Model Architecture

* U-Net or U-Net variant
* Encoder–decoder structure with skip connections
* Multi-class segmentation (optic disc, optic cup, background)

### 6.2 Functional Requirements

* Generate binary masks for optic disc and optic cup
* Preserve anatomical boundaries
* Handle illumination and anatomical variability

### 6.3 Performance Requirements

* Dice coefficient ≥ 0.85 (optic disc)
* Dice coefficient ≥ 0.80 (optic cup)

---

## 7. Biomarker Extraction Requirements

### 7.1 Computed Clinical Biomarkers

From segmentation masks:

* Vertical Cup-to-Disc Ratio (vCDR)
* Area-based CDR
* Rim-to-disc ratio approximation

### 7.2 Computation Details

* Masks must be morphologically cleaned
* Largest connected component retained
* Vertical diameters computed via bounding-box projection

All biomarker values must be logged per image.

The system shall compute the following biomarkers:

* Cup-to-Disc Ratio (CDR)
* Optic cup area
* Optic disc area
* Neuroretinal rim approximation

CDR must be computed as:

CDR = Area(Optic Cup) / Area(Optic Disc)

Biomarkers must be logged for downstream reasoning and auditability.

---

## 8. Classification Module Requirements

### 8.1 Model Architecture

* Backbone: **DenseNet-121**
* Input: Preprocessed fundus image + optional CDR scalar (late fusion)
* Fusion: Fully connected layer concatenating image embedding and biomarker vector

### 8.2 Training Requirements

* Loss: Binary Cross-Entropy with focal weighting
* Class imbalance handling: Weighted sampling
* Calibration: Temperature scaling post-training

### 8.3 Output

* Glaucoma probability `P(G)`
* Confidence interval (via Monte Carlo dropout)

### 8.1 Model Architecture

* DenseNet-121 backbone
* Transfer learning with ImageNet weights
* Sigmoid output for binary classification

### 8.2 Functional Requirements

* Output glaucoma probability score (0–1)
* Support confidence calibration

### 8.3 Performance Requirements

* Accuracy ≥ 85%
* AUC-ROC ≥ 0.90
* High recall priority (minimize false negatives)

---

## 9. Agentic AI Reasoning Layer Requirements

### 9.1 Agent Graph (LangGraph-Compatible)

The agentic layer shall be implemented as a **directed acyclic graph**:

```
Vision Agent ─┐
              ├──► Diagnostic Agent ───► Report Agent
Risk Agent ───┘
```

### 9.2 Agent Specifications

#### Vision Agent

* Inputs: vCDR, segmentation quality metrics
* Logic: Compare against clinical thresholds (vCDR > 0.6)

#### Risk Agent

* Inputs: Age, IOP, family history
* Logic: Rule-based + prompt-guided reasoning

#### Diagnostic Agent

* Inputs: P(G), Vision Agent output, Risk Agent output
* Logic: Weighted evidence fusion

#### Report Agent

* Inputs: All agent outputs
* Output: Natural-language clinical screening report

### 9.3 LLM Execution

* Provider: **Groq API**
* Model: `llama-3.1-70b-versatile`
* Temperature ≤ 0.2
* Deterministic prompts only

### 9.0 LLM Provider Requirement

* The agentic reasoning layer **must use the Groq API** as the primary LLM inference backend.
* Groq API keys must be loaded via environment variables (`GROQ_API_KEY`).
* The system must support low-latency, stateless inference suitable for Antigravity execution.

### 9.1 Architecture

The reasoning layer shall consist of **cooperative autonomous agents** sharing a common memory.

### 9.2 Required Agents

#### Vision Agent

* Input: segmentation masks, CDR
* Role: assess structural glaucomatous patterns

#### Risk Agent

* Input: patient metadata
* Role: evaluate epidemiological risk factors

#### Diagnostic Agent

* Input: classifier probability
* Role: interpret prediction confidence and uncertainty

#### Report Agent

* Input: all agent outputs
* Role: generate interpretable clinical summary

### 9.3 Shared Memory

Shared memory must contain:

* CDR
* Glaucoma probability
* Patient metadata

### 9.4 Decision Fusion

* Weighted multi-agent fusion strategy
* Configurable agent weights
* Threshold-based final decision categories:

  * Non-glaucomatous
  * Glaucoma suspect
  * High-risk / referral recommended

---

## 10. Explainability & Interpretability Requirements

* Each decision must include:

  * Structural evidence (e.g., elevated CDR)
  * Risk factor contribution
  * Model confidence
* Reasoning trace must be human-readable
* Black-box predictions without explanation are not permitted

---

## 11. LLM & Knowledge Integration Requirements

* Primary LLM provider: **Groq API (LPU-based inference)**
* Supported Groq models (configurable):

  * `llama-3.1-70b-versatile`
  * `llama-3.1-8b-instant`
* All prompts must be:

  * Deterministic
  * Temperature-controlled
  * Clinically grounded
* No long-term memory persistence outside Antigravity-supported storage

### 11.1 Groq Usage Constraints

* Sub-100ms token latency expected

* Stateless request–response pattern

* Explicit retry and timeout handling

* Prompt + tool outputs must be logged for traceability

* Support for LLM-based reasoning (e.g., Groq / OpenAI / local LLM)

* Prompt templates grounded in ophthalmology guidelines

* Optional Retrieval-Augmented Generation (RAG) from medical knowledge base

* Hallucination mitigation through constrained prompts

---

## 12. User Interface Requirements

### 12.1 Functional UI Components

* Fundus image upload
* Patient data entry form
* Visualization of segmentation masks
* Display of CDR and risk scores
* Diagnostic summary and recommendations

### 12.2 UX Requirements

* Simple, clinician-friendly design
* Minimal required inputs
* Clear risk categorization

---

## 13. Evaluation & Validation Requirements

### 13.1 Segmentation Metrics

* Dice coefficient
* Precision
* Recall

### 13.2 Classification Metrics

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

### 13.3 Reasoning Evaluation

* Clinical plausibility of explanations
* Consistency across similar cases
* Ablation studies on agent contributions

---

## 14. Deployment Requirements

### 14.1 Antigravity Execution

* Single-command reproducible run
* No interactive authentication
* GPU optional but supported

### 14.2 Container Constraints

* Base image: CUDA-enabled Python image
* Explicit dependency pinning
* Stateless execution except mounted volumes

### 14.3 Runtime Configuration

* `GROQ_API_KEY`

* `KAGGLE_USERNAME`

* `KAGGLE_KEY`

* `RUN_MODE` (train / eval / inference)

* GPU support for training

* CPU-compatible inference option

* Modular Docker-based deployment (optional)

* Local or cloud execution

---

## 15. Ethical & Clinical Requirements

* Screening-only (not definitive diagnosis)
* Explicit disclaimer in UI and reports
* No autonomous treatment decisions
* Bias monitoring across demographics

---

## 16. Research & Extension Requirements

* Support future integration of:

  * Visual field data
  * OCT imaging
  * Longitudinal patient history
  * Federated learning
  * Continual learning

---

## 17. Documentation Requirements

* Model architecture documentation
* Training configuration logs
* Hyperparameter tables
* Experiment reproducibility notes
* System flow diagrams

---

## 18. Success Criteria

The system is considered successful if it:

* Matches reported performance benchmarks
* Produces clinically interpretable outputs
* Demonstrates agentic reasoning traces
* Can be explained end-to-end during project evaluation

---

*End of requirements.md*
