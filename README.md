# RTGS AI Analyst — Real-Time Government Statistics AI Pipeline

📊 **A fully data-agnostic, AI-powered pipeline that transforms raw datasets into policy-ready insights and reports.**

This system automatically ingests, cleans, standardizes, and analyzes structured data to produce **executive summaries, technical analysis, and government-ready policy reports** — all in a single command.


<img width="2050" height="3840" alt="Image" src="https://github.com/user-attachments/assets/29020b75-34ad-4de6-99e7-a1ae404f06db" />


## ✨ Features

* **Data Agnostic**: Works with *any CSV/Excel dataset* without custom coding.
* **Domain-Aware**: Auto-detects relevant context (transport, health, urban, education, etc.).
* **Interactive Mode**: Optionally asks contextual questions (skippable).
* **LLM-Powered Agents**: Schema inference, narrative insights, polished reporting, memory, and observability.
* **Modular Architecture**: Built on **LangGraph** + **LangChain** for flexible, agentic orchestration.
* **Full Data Lifecycle**: Raw → Standardized → Cleaned → Transformed → Analyzed → Reported.
* **Rich Deliverables**: Markdown summaries, PDF reports, technical analysis dashboards, and policy recommendations.

---

## 🚀 Quickstart

### 1. Clone Repository

```bash
git clone https://github.com/Nithin8919/RTGS-Style_AI_Analyst.git
cd RTGS-Style_AI_Analyst
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Copy the environment template and update if needed:

```bash
cp env_template.txt .env
give your api key
```

### 4. Run Analysis

Basic run with your dataset:

```bash
python cli.py run --dataset path/to/your/dataset.csv
```

Interactive mode (asks contextual questions) optional to enter :

```bash
python cli.py run --dataset data/raw/sample.csv --interactive
```

Auto-detect domain:

```bash
python cli.py run --dataset data/raw/sample.csv --domain auto
```

---

## 🔄 Data Flow

```
Raw Data → Ingestion → Schema Inference → Standardization → Cleaning 
→ Transformation → Validation → Analysis → Insights → Report Assembly
```

Artifacts are saved systematically:

* `data/raw/` → Original files
* `data/standardized/` → Normalized schemas
* `data/cleaned/` → Clean datasets
* `data/transformed/` → Analysis-ready features
* `artifacts/reports/` → Markdown + PDF reports
* `artifacts/logs/` → Execution logs & traces
* `memory/store/` → Run history & reusable schema mappings

---

## 📂 Project Structure

```
├── artifacts
│   ├── docs/                # Documentation artifacts
│   ├── logs/                # Execution logs
│   └── reports/             # Generated reports (Markdown, PDF, dashboards)
├── backend_server.py         # Backend server entry point
├── cli.py                    # CLI runner for datasets
├── config.yaml               # Configurations
├── data
│   ├── raw/                 # Input datasets
│   ├── standardized/        # Schema-normalized datasets
│   ├── cleaned/             # Clean datasets
│   └── transformed/         # Analysis-ready datasets
├── docs/                     # Developer documentation
├── env_template.txt          # Environment template
├── Frontend/                 # Frontend (React + Tailwind + Vite)
│   ├── src/                  # Components, services, UI logic
│   └── vite.config.ts        # Build config
├── memory/
│   └── store/               # Memory DB + run history
├── README.md                 # Project overview
├── requirements.txt          # Python dependencies
├── src/
│   ├── agents/              # Core agents
│   │   ├── ingestion_agent.py
│   │   ├── schema_inference_agent.py
│   │   ├── standardization_agent.py
│   │   ├── cleaning_agent.py
│   │   ├── transformation_agent.py
│   │   ├── validator_agent.py
│   │   ├── analysis_agent.py
│   │   ├── insight_agent.py
│   │   ├── report_agent.py
│   │   ├── memory_agent.py
│   │   └── observability_agent.py
│   ├── orchestrator/        # Flow control + agent routing
│   └── utils/               # Helpers, logging, visualization
└── uploads/                  # User-uploaded datasets
```

---

## 🧩 Agent Architecture

### Data Preparation Agents

1. 📥 **Ingestion Agent** – fetches, validates, fingerprints raw data.
2. 🔍 **Schema Inference Agent (LLM)** – AI-powered column typing, naming, confidence scoring.
3. 🏗️ **Standardization Agent** – consistent formats, snake\_case, unit normalization.
4. 🧹 **Cleaning Agent** – handles missing data, duplicates, outliers.
5. ⚡ **Transformation Agent** – builds derived features, per-capita metrics, domain-specific KPIs.

### Analysis & Reporting Agents

6. ✅ **Validator Agent** – quality gates, data confidence scoring.
7. 📊 **Analysis Agent** – stats, trends, correlations, hypothesis tests.
8. 💡 **Insight Generation Agent (LLM)** – narratives, recommendations, impact statements.
9. 📋 **Report Assembly Agent (LLM)** – publication-ready Markdown + PDF.

### Infrastructure Agents

10. 🧠 **Memory Agent (LLM)** – schema reuse, fingerprinting, caching.
11. 👁️ **Observability Agent (LLM)** – traces, logs, performance monitoring.

---

## 🏗️ Under the Hood

* **LangGraph** – orchestrates multi-agent workflow as a state graph.
* **LangChain** – powers LLM interactions and tool execution.
* **Advanced Statistical Suite** – robust descriptive, inferential, and correlation analysis.
* **Memory Layer** – SQLite + JSON for schema mappings, run history, and LLM caching.

---

## 🎮 Usage Guide

### Basic Commands

#### **Full Analysis**
```bash
python cli.py run --dataset "path/to/your/dataset.csv"
```

#### **Interactive Mode** (Ask contextual questions)
```bash
python cli.py run --dataset "data/health_indicators.xlsx" --interactive
```

#### **Domain-Specific Analysis**
```bash
python cli.py run --dataset "data/transport_data.csv" --domain transport
```

#### **Custom Scope**
```bash
python cli.py run --dataset "data/education_survey.csv" --scope "Telangana 2020"
```

### CLI Options

- `--dataset`: Path to your dataset (CSV/Excel)
- `--domain`: Domain hint (transport, health, education, urban, auto)
- `--scope`: Geographic/temporal scope (e.g., "Telangana 2020", "Hyderabad district")
- `--interactive`: Ask domain-specific questions for context
- `--output-summary`: Generate only summary (no full report)
- `--export-pdf`: Generate PDF report

### Command Examples

```bash
# Run full pipeline
python cli.py run --dataset data/raw/education.csv --scope "Telangana 2020"

```



### Report Structure
Every run generates four key outputs inside `artifacts/reports/`:

* **Executive Summary** – One-page policy briefing.
* **Technical Analysis** – Statistical details, quality dashboards, correlation matrices.
* **Policy Report (PDF)** – Action items, interventions, budget estimates, success metrics.
* **Logs & Run History** – Traceable execution + schema memory.

**Example Output Structure:**
```
artifacts/reports/rtgs-education-20250905-001/
├── 📋 enhanced_policy_insights.md           # Run metadata & parameters
├── 📝 enhanced_techinal_analysis.md                  # Full markdown report  
├── 📄 techincal_anlysisr.pdf    
├── 📄 enhanced_policy_insights(Graphical DATA).pdf              # PDF version

├── 📁 data/                       # Processed datasets
│   ├── raw_sample.csv             # Original data sample
│   ├── standardized.csv           # Post-standardization  
│   ├── cleaned.csv                # Post-cleaning
│   └── transformed.csv            # Analysis-ready final dataset
└── 📋 logs/                       # Audit trail
    ├── transform_log.jsonl        # Every transform with justification

```

---



### **Frontend Integration**
The project includes a React frontend for non-technical users which is not completely ready yet it has some placeholder analysis and visualization:

```bash
cd Frontend
npm install
npm run dev
# Access at http://localhost:3000
```

---


## 🌍 Scope & Use Cases

Designed for **policy makers, analysts, and data teams** across:

* **Urban Development**: Infrastructure planning, service distribution analysis
* **Health & Education**: Resource allocation, performance monitoring  
* **Transportation**: Traffic analysis, infrastructure optimization
* **Economic Analysis**: Budget allocation, development indicators
* **Evidence-Based Governance**: Data-driven policy making

---

## 🚀 Why RTGS Stands Out

### **Hackathon Edge Factors**

1. **📊 Data Agnostic**: Works with ANY government dataset structure
2. **🤖 Multi-Agent Intelligence**: 11 specialized AI agents working together
3. **📋 Policy-First**: Every output is policy-actionable with confidence scores
4. **🔍 Full Transparency**: Complete audit trail of every transformation
5. **⚡ Speed**: Minutes vs weeks for analysis
6. **🏗️ Production Ready**: LangGraph orchestration, proper error handling
7. **🎯 Government Focus**: Built specifically for policy makers

### **Technical Depth**
- **LangGraph**: State-of-the-art agent orchestration
- **LangChain**: Modular LLM tool integration  
- **Statistical Rigor**: Hypothesis testing, confidence intervals, effect sizes
- **Memory System**: Learning from past runs for efficiency

### **Policy Impact**
- **Executive Summaries**: One-page policy briefs
- **Actionable Recommendations**: Prioritized by impact and confidence
- **Evidence-Based**: Every recommendation backed by statistical evidence
- **Risk Assessment**: Confidence scoring for decision support

---

## 🤝 Contributing

Contributions welcome! Please fork, branch, and open a PR. For major changes, discuss via issues first.



---

🔥 **Run once. Get policy-ready insights. Anywhere, any dataset.**
