# Glaucoma Screening Assistant - Doctor's Guide

Welcome to the **Automated Agentic AI Glaucoma Screening System**. This guide will help you set up and run the screening tool on your laptop.

## 📋 Prerequisites
Before starting, ensure your laptop has the following installed:
- **Python 3.10 or higher**: [Download Here](https://www.python.org/downloads/)
- **Git**: [Download Here](https://git-scm.com/downloads)

## 🚀 Installation & Setup

### 1. Download the Application
Open your terminal (Command Prompt on Windows, Terminal on Mac/Linux) and run:

```bash
git clone https://github.com/pooja-gani/glaucomaa.git
cd glaucomaa
```

### 2. Set Up the Environment
Create a virtual environment to keep your system clean:

**Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies
Install the required software libraries:
```bash
pip install -r requirements.txt
```

## 🔑 AI Reasoning Setup (Groq API)
To enable the advanced "Agentic Reasoning" (where the AI explains its diagnosis and provides detailed reports), you need a free API key from Groq.

1.  Go to [console.groq.com](https://console.groq.com/keys).
2.  Login and click **"Create API Key"**.
3.  Copy the key (it starts with `gsk_...`).
4.  You will paste this key into the App sidebar when you run it.

## 🖥️ Running the Application
To start the screening tool, run the following command in your terminal:

```bash
./run.sh app
```

*Note: If `./run.sh` doesn't work on Windows, use `streamlit run src/app/main.py` instead.*

A browser window should automatically open with the application.

## 🩺 How to Use
1.  **Sidebar Configuration**:
    - Paste your **Groq API Key** in the "Settings" sidebar.
    - Enter Patient Metadata (Age, IOP, Family History, etc.).
2.  **Upload Image**:
    - Drag and drop a retinal fundus image (JPG/PNG) into the upload area.
3.  **Visual Analysis**:
    - The AI will inspect the optic disc and cup.
    - *Note: If segmentation is unavailable, it will rely on the classification model.*
4.  **Clinical Report**:
    - Scroll down to see the **Final Clinical Report**.
    - Review the **Suggested Management Plan**, which details specific medicines (e.g., Latanoprost) and follow-up schedules.

## ❓ Troubleshooting
- **"Command not found"**: Ensure you are inside the `glaucomaa` folder.
- **"Module not found"**: Ensure you activated the virtual environment (`source venv/bin/activate`) before running.
