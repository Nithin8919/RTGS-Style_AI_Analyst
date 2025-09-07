# RTGS AI Analyst — Real-Time Government Statistics AI Pipeline

**🎯 A multi-agent AI system that transforms messy government datasets into actionable policy insights in minutes, not months.**

[![Demo Video](https://img.shields.io/badge/📹_Demo_Video-Watch_Now-red)](YOUR_DEMO_VIDEO_LINK)
[![LangSmith Traces](https://img.shields.io/badge/🔍_LangSmith-View_Traces-blue)](YOUR_LANGSMITH_LINK)
[![Report Sample](https://img.shields.io/badge/📊_Sample_Report-View_PDF-green)](artifacts/reports/sample/report.pdf)

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/your-org/rtgs-ai-analyst
cd rtgs-ai-analyst
pip install -r requirements.txt

# Run analysis on any government dataset
python main.py run \
  --source "data/vehicle_registrations_hyderabad_2023.csv" \
  --domain-hint transport \
  --scope "district=Hyderabad,year=2023" \
  --auto-approve

# ✅ Get instant executive summary + full report in artifacts/
```

**💡 What you get:** Clean data + Interactive charts + Policy recommendations + Full audit trail — all generated automatically.

---

## 🎯 Problem We Solve

Government data analysts spend **80% of their time** cleaning data and only **20% analyzing**. RTGS AI Analyst flips this ratio:

**Before RTGS:**
- 3-4 weeks to clean a messy district dataset
- Manual schema mapping and outlier detection  
- Copy-paste analysis in Excel
- Static reports that become outdated

**After RTGS:**
- **3-4 minutes** end-to-end automated pipeline
- AI-powered schema inference and standardization
- Statistical analysis with hypothesis testing
- Interactive reports with confidence scores

---

## 🏗️ System Architecture

RTGS uses a **multi-agent architecture** orchestrated by LangGraph, with each agent handling a specific data transformation step:

```
Raw Dataset → [11 AI Agents] → Policy-Ready Insights

┌─────────────────────────────────────────────────────────┐
│                 LangGraph Orchestrator                  │
│   ┌─────────────────────────────────────────────────┐   │
│   │  Agent Flow (Linear Chain)                      │   │
│   │                                                 │   │
│   │  Ingest → Schema → Standard → Clean → Transform │   │
│   │     ↓        ↓        ↓        ↓        ↓       │   │
│   │  Validate → Analyze → Insights → Report → Memory │   │
│   │                                                 │   │
│   └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
        │                    │                    │
    ┌───▼───┐           ┌───▼───┐           ┌───▼───┐
    │LangChain│          │LlamaIndex│        │Observability│
    │(Tools)  │          │(Memory)  │        │(LangSmith)  │
    └─────────┘          └─────────┘        └─────────────┘
```

### Framework Roles

- **🎭 LangGraph**: Orchestrates the multi-agent pipeline, handles retries, branching logic
- **🔧 LangChain**: Powers individual agent tools (cleaning chains, analysis tools, report generation)  
- **🧠 LlamaIndex**: Long-term memory for schema mappings, transform decisions, and semantic Q&A
- **👁️ LangSmith**: Observability and tracing for debugging and audit trails

---

## 🤖 The 11 AI Agents

### **Data Processing Agents**

#### 1. 📥 **Ingestion Agent**
- **Role**: Fetches and validates raw data sources
- **Input**: URLs, file paths, domain hints
- **Output**: Raw data copy + basic manifest + sample rows
- **Smart Features**:
  - Auto-detects CSV/Excel encoding and separators
  - Handles multi-sheet Excel files
  - Streams large files (>100MB) efficiently
  - Generates dataset fingerprint for caching

#### 2. 🔍 **Schema Inference Agent** 
- **Role**: Understands what each column contains
- **Input**: Sample data rows + domain context
- **Output**: Inferred types + canonical names + confidence scores
- **Smart Features**:
  - Detects 8 data types: numeric, datetime, categorical, boolean, geo, ID, text
  - AI-powered column naming (e.g., "veh_reg_dt" → "vehicle_registration_date")
  - Confidence scoring for ambiguous columns
  - Domain-aware type detection (transport, health, education)

#### 3. 🏗️ **Standardization Agent**
- **Role**: Makes column names and formats consistent
- **Input**: Raw data + inferred schema
- **Output**: Standardized dataset with canonical column names
- **Smart Features**:
  - Snake_case column naming
  - Unit detection and normalization  
  - Boolean standardization (Yes/No → True/False)
  - Reusable alias mapping across datasets

#### 4. 🧹 **Cleaning Agent**
- **Role**: Handles missing data, duplicates, and outliers
- **Input**: Standardized data + cleaning policies
- **Output**: Clean dataset + detailed transform log
- **Smart Features**:
  - **Missing Data**: Median impute (<5% missing), group impute (<20%), drop column (>95%)
  - **Duplicates**: Exact + fuzzy duplicate detection
  - **Outliers**: IQR-based flagging with configurable thresholds
  - **Human-in-loop**: Requires approval for high-impact transforms (>5% rows affected)

#### 5. ⚡ **Transformation Agent**
- **Role**: Creates analysis-ready features
- **Input**: Clean data + domain knowledge
- **Output**: Transformed dataset with derived features
- **Smart Features**:
  - **Time Features**: Year/month extraction, rolling averages, percent change
  - **Per-Capita Metrics**: Population-normalized KPIs for geographic analysis
  - **Statistical Features**: Quantile buckets, correlation-ready variables
  - **Domain Templates**: Transport (vehicle density), Health (per-1000 rates), Education (dropout rates)

### **Analysis & Intelligence Agents**

#### 6. ✅ **Validator Agent**
- **Role**: Quality assurance and confidence scoring
- **Input**: Transformed data + transform history
- **Output**: Validation report + quality gates + confidence score (HIGH/MEDIUM/LOW)
- **Quality Gates**:
  - Completeness checks on key columns
  - Data type consistency validation
  - Range plausibility checks
  - Transform impact assessment

#### 7. 📊 **Analysis Agent**  
- **Role**: Computes statistical insights and KPIs
- **Input**: Clean dataset + KPI definitions
- **Output**: Numerical analysis + charts metadata + hypothesis tests
- **Statistical Methods**:
  - **KPIs**: Sum, mean, median, std, min, max, count for all numeric fields
  - **Time Trends**: Linear trend fitting + seasonality detection via autocorrelation
  - **Spatial Analysis**: Geographic inequality using Gini coefficient + per-capita rankings
  - **Hypothesis Testing**: T-tests and Wilcoxon tests with effect size (Cohen's d)
  - **Correlation Analysis**: Spearman correlation matrix with significance testing

#### 8. 💡 **Insight Generation Agent** (LLM-Powered)
- **Role**: Converts numbers into policy-relevant narratives  
- **Input**: Statistical outputs + domain context
- **Output**: Executive summary + prioritized recommendations + confidence badges
- **AI Features**:
  - Plain-language impact statements ("Vehicle registrations up 6% in Hyderabad")
  - Evidence-backed recommendations with statistical justification
  - Priority ranking by impact and confidence
  - **Privacy-Safe**: Only aggregated stats sent to LLM, never raw PII

#### 9. 📋 **Report Assembly Agent**
- **Role**: Creates publication-ready deliverables
- **Input**: All analysis artifacts + narratives + charts
- **Output**: CLI summary + Markdown report + PDF + interactive plots
- **Report Includes**:
  - 1-page executive summary with key findings
  - Data health snapshot (missing values, quality flags)
  - KPI tables sorted by policy importance
  - Interactive time series and geographic visualizations
  - Statistical test results with caveats
  - Actionable recommendations with confidence scores

### **Infrastructure Agents**

#### 10. 🧠 **Memory Agent**
- **Role**: Learns from past runs to speed up future analysis
- **Input**: Run artifacts + transform decisions + schemas
- **Output**: Reusable transform patterns + schema mappings
- **Smart Features**:
  - Dataset fingerprinting (schema + sample hash)
  - Transform rule reuse across similar datasets
  - LLM response caching to avoid repeated token usage
  - Schema similarity matching for new datasets

#### 11. 👁️ **Observability Agent**
- **Role**: Full pipeline traceability and debugging
- **Input**: All agent execution events
- **Output**: LangSmith traces + structured logs + audit trail
- **Tracking Features**:
  - Every transform logged with justification
  - LLM prompt/response tracing (redacted for privacy)
  - Performance metrics and bottleneck identification
  - Reproducible run snapshots

---

## 📊 Sample Output

### CLI Executive Summary
```
╔══════════════════════════════════════════════════════════════════════╗
║ RTGS Analysis: Hyderabad Vehicle Registrations 2023                 ║
║ RunID: rtgs-vehicles-20250905-001 | Confidence: HIGH ✅              ║
╠══════════════════════════════════════════════════════════════════════╣
║ 📈 KEY FINDINGS:                                                     ║
║ • Vehicle registrations +6.2% YoY (124K → 132K vehicles)            ║
║ • Peripheral mandals 40% below city average (intervention needed)   ║
║ • Electric vehicle adoption accelerating (+180% in Q4 2023)         ║
╠══════════════════════════════════════════════════════════════════════╣
║ 📊 TOP KPIS:          Value    │ Trend │ Confidence                  ║
║ • Total Registrations  132K     │  ↗ +6% │ HIGH                     ║
║ • Per-Capita Rate      8.3/1K   │  ↗ +3% │ MEDIUM                   ║
║ • Electric Vehicle %   12.4%    │ ↗ +180%│ HIGH                     ║
╠══════════════════════════════════════════════════════════════════════╣
║ 🎯 TOP RECOMMENDATIONS:                                              ║
║ 1. Deploy mobile registration units in low-performing mandals       ║
║ 2. Expand EV charging infrastructure to sustain growth              ║
║ 3. Investigate 40% urban-rural registration gap                     ║
╠══════════════════════════════════════════════════════════════════════╣
║ 📁 Full Report: artifacts/reports/rtgs-vehicles-20250905-001/       ║
║ 🔍 Trace: https://smith.langchain.com/run/abc123                    ║
╚══════════════════════════════════════════════════════════════════════╝
```

### Interactive Time Series (HTML Export)
```
Month    Registrations  Trend   EV %    Per-Capita
Jan 2023    10,245       ↗      8.2%     7.8/1K
Feb 2023    10,891       ↗      9.1%     8.1/1K  
Mar 2023    11,234       ↗     10.5%     8.3/1K
...
Dec 2023    12,789       ↗     18.7%     9.2/1K

📊 Interactive charts available at: artifacts/plots/rtgs-vehicles-20250905-001/
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9+
- 8GB+ RAM (16GB recommended for large datasets)
- API Keys: OpenAI/Anthropic (for LLM agents), LangSmith (for observability)

### 1. Clone Repository
```bash
git clone https://github.com/your-org/rtgs-ai-analyst
cd rtgs-ai-analyst
```

### 2. Install Dependencies
```bash
# Create virtual environment
python -m venv rtgs-env
source rtgs-env/bin/activate  # On Windows: rtgs-env\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 3. Configure Environment
```bash
# Copy sample config
cp config/config_sample.yaml config/config.yaml

# Set API keys
export OPENAI_API_KEY="your-openai-key"
export LANGSMITH_API_KEY="your-langsmith-key"  
export LANGSMITH_PROJECT="rtgs-analysis"

# Or create .env file
echo "OPENAI_API_KEY=your-key" >> .env
echo "LANGSMITH_API_KEY=your-key" >> .env
```

### 4. Verify Setup
```bash
# Test with sample data
python main.py run --source "examples/sample_transport_data.csv" --preview

# Should output schema inference and transforms preview
```

---

## 🎮 Usage Guide

### Basic Commands

#### **Full Analysis**
```bash
python main.py run \
  --source "path/to/your/dataset.csv" \
  --domain-hint transport \
  --scope "district=Hyderabad,year=2023" \
  --auto-approve
```

#### **Preview Mode** (See transforms before applying)
```bash
python main.py run \
  --source "data/health_indicators.xlsx" \
  --domain-hint health \
  --preview
```

#### **Dry Run** (Schema inference only)
```bash
python main.py run \
  --source "data/education_survey.csv" \
  --dry-run
```

### Advanced Options

#### **Custom Sampling**
```bash
python main.py run \
  --source "large_dataset.csv" \
  --sample-rows 1000 \
  --mode run
```

#### **Custom Output Directory**
```bash
python main.py run \
  --source "data.csv" \
  --output-dir "artifacts/my_analysis" \
  --report-format pdf
```

#### **Scope Filtering** 
```bash
# Geographic scope
python main.py run --source "state_data.csv" --scope "district=Warangal,mandal=Hanamkonda"

# Temporal scope  
python main.py run --source "monthly_data.csv" --scope "year=2023,quarter=Q4"

# Multiple filters
python main.py run --source "data.csv" --scope "district=Hyderabad,year=2023,category=urban"
```

### Domain Hints (Optimizes analysis for specific sectors)

- `transport`: Vehicle registrations, traffic data, road infrastructure
- `health`: Disease surveillance, hospital data, vaccination records
- `education`: School enrollment, dropout rates, exam results  
- `economics`: GDP, employment, budget allocation
- `agriculture`: Crop yields, farmer data, land records

---

## 📁 Project Structure

```
rtgs-ai-analyst/
├── 📁 src/
│   ├── 🎭 orchestrator/           # LangGraph pipeline orchestration
│   │   ├── flow_controller.py     # Main agent flow DAG
│   │   └── agent_router.py        # Output routing between agents
│   │
│   ├── 🤖 agents/                 # Individual agent implementations  
│   │   ├── ingestion_agent.py     # Data fetching & sampling
│   │   ├── schema_agent.py        # Type inference & canonicalization
│   │   ├── standardization_agent.py # Column naming & formatting
│   │   ├── cleaning_agent.py      # Missing data & outlier handling
│   │   ├── transformation_agent.py # Feature engineering
│   │   ├── validation_agent.py    # Quality gates & confidence scoring
│   │   ├── analysis_agent.py      # Statistical analysis & KPIs
│   │   ├── insight_agent.py       # LLM-powered narrative generation
│   │   ├── report_agent.py        # Report assembly & visualization
│   │   ├── memory_agent.py        # Learning & caching
│   │   └── observability_agent.py # Tracing & logging
│   │
│   └── 🔧 utils/                  # Helper functions
│       ├── data_utils.py          # Pandas/Polars helpers
│       ├── plot_utils.py          # Plotly chart generation
│       ├── config_manager.py      # Configuration handling
│       └── llm_utils.py           # LLM prompt templates & caching
│
├── 📊 artifacts/                  # Generated analysis outputs
│   ├── docs/                                      
│   ├── reports/                   # Markdown & PDF reports
│   └── logs/                      # Transform logs & run manifests
│
├── 🔧 config.yaml                   # Configuration files
│
│
├── 📖 examples/                  # Sample datasets & demos
│   ├── transport_demo/           # Vehicle registration sample
│   ├── health_demo/              # Disease surveillance sample
│   └── education_demo/           # School enrollment sample
│
├── 📋 main.py                    # CLI entry point
├── 🐍 requirements.txt           # Python dependencies
├── ⚙️ .env.sample               # Environment variables template
└── 📖 README.md                 # This file
```

---

## 🔧 Configuration

### **Main Config** (`config/config.yaml`)

```yaml
# Data Quality Thresholds (all configurable)
data_quality:
  sample_rows_default: 500
  drop_column_threshold: 0.95        # Drop if >95% null
  median_impute_threshold: 0.05      # Median impute if <5% null  
  group_impute_threshold: 0.20       # Group impute if <20% null
  high_missingness_threshold: 0.30   # Flag column as low-quality
  max_auto_drop_rows_percent: 0.01   # Require approval if >1% rows dropped
  outlier_iqr_multiplier: 1.5        # IQR outlier detection sensitivity

# Statistical Analysis
analysis:
  significance_alpha: 0.05           # P-value threshold
  effect_size_thresholds:
    small: 0.2                       # Cohen's d thresholds
    medium: 0.5
    large: 0.8
  correlation_threshold: 0.6         # Highlight correlations above this

# LLM Settings  
llm:
  model: "gpt-4o-mini"              # Fast model for most tasks
  model_advanced: "gpt-4o"          # Advanced model for complex insights
  max_tokens: 1000                   # Token limit per call
  cache_responses: true              # Cache by prompt fingerprint
  pii_redaction: true               # Never send raw PII to external LLMs

# Memory & Caching
memory:
  enable_schema_reuse: true         # Reuse transforms across similar datasets
  similarity_threshold: 0.7        # Schema similarity for reuse suggestions
  max_memory_entries: 1000         # Limit stored transform patterns

# Observability
observability:
  langsmith_enabled: true          # Send traces to LangSmith
  local_logs: true                 # Always keep local JSONL logs
  log_level: "INFO"               # DEBUG, INFO, WARNING, ERROR
```

### **Domain-Specific Rules** (`config/domain_mappings/`)

```yaml
# transport.yaml
canonical_columns:
  - pattern: ["vehicle.*reg.*", "veh.*reg.*", "registration.*"]
    canonical: "vehicle_registration_date"
    type: "datetime"
  - pattern: ["owner.*name", "proprietor", "holder"]  
    canonical: "owner_name"
    type: "text"
    pii: true

kpi_priorities:
  - "total_registrations"
  - "registrations_per_capita" 
  - "electric_vehicle_percentage"
  - "average_vehicle_age"

feature_templates:
  - name: "registrations_per_capita"
    formula: "total_registrations / population * 1000"
    requires: ["population"]
```

---

## 📊 Understanding the Output

### **Artifacts Directory Structure**
After each run, RTGS creates a timestamped folder with all outputs:

```
artifacts/reports/rtgs-vehicles-20250905-001/
├── 📋 run_manifest.json           # Run metadata & parameters
├── 📝 report.md                   # Full markdown report  
├── 📄 report.pdf                  # PDF version (if requested)
├── 📊 plots/                      # Interactive visualizations
│   ├── time_series.html           # Plotly time series charts
│   ├── spatial_analysis.html      # Geographic heatmaps  
│   ├── correlation_matrix.html    # Interactive correlation plot
│   └── kpi_dashboard.html         # Executive KPI dashboard
├── 📁 data/                       # Processed datasets
│   ├── raw_sample.csv             # Original data sample
│   ├── standardized.csv           # Post-standardization  
│   ├── cleaned.parquet            # Post-cleaning (Parquet for efficiency)
│   └── transformed.parquet        # Analysis-ready final dataset
└── 📋 logs/                       # Audit trail
    ├── transform_log.jsonl        # Every transform with justification
    ├── validation_report.json     # Quality gates & confidence scores
    └── agent_trace.jsonl          # Agent execution timeline
```

### **Key Files Explained**

#### **📋 run_manifest.json** - Run Metadata
```json
{
  "run_id": "rtgs-vehicles-20250905-001",
  "dataset_name": "hyderabad_vehicle_registrations",
  "source_url": "data/vehicles_2023.csv", 
  "scope": "district=Hyderabad,year=2023",
  "row_count_raw": 124234,
  "sample_rows_count": 500,
  "schema_hash": "abc123def456",
  "timestamp_utc": "2025-09-05T12:00:00Z",
  "confidence_overall": "HIGH",
  "agent_versions": {
    "ingestion": "v1.2",
    "cleaning": "v1.1", 
    "analysis": "v1.3"
  },
  "artifacts_paths": {
    "raw": "data/raw_sample.csv",
    "cleaned": "data/cleaned.parquet", 
    "report": "report.md",
    "plots": "plots/"
  },
  "notes": "High-quality dataset with minimal missing values"
}
```

#### **📋 transform_log.jsonl** - Audit Trail
```json
{"timestamp": "2025-09-05T12:01:15Z", "agent": "cleaning", "action": "median_impute", "column": "vehicle_age", "rows_affected": 1247, "rule_id": "num_median_impute_v1", "rationale": "null_frac=0.03 below threshold 0.05", "confidence": "high", "preview_before": "[null, null, 5]", "preview_after": "[8, 8, 5]"}
{"timestamp": "2025-09-05T12:01:22Z", "agent": "transformation", "action": "derive_feature", "column": "registrations_per_capita", "rows_affected": 124234, "rule_id": "per_capita_v1", "formula": "total_registrations / population * 1000", "confidence": "high"}
{"timestamp": "2025-09-05T12:02:45Z", "agent": "analysis", "action": "hypothesis_test", "test": "t_test", "groups": ["urban", "rural"], "p_value": 0.001, "effect_size": 0.8, "conclusion": "significant_difference"}
```

#### **📋 validation_report.json** - Quality Assessment
```json
{
  "gate_results": [
    {"gate": "completeness_check", "status": "PASS", "details": "All key columns <20% missing"},
    {"gate": "type_consistency", "status": "PASS", "details": "No type conflicts detected"},
    {"gate": "outlier_check", "status": "WARNING", "details": "12 extreme outliers flagged in vehicle_price"}
  ],
  "data_confidence_score": 0.85,
  "key_metrics": {
    "pct_columns_changed": 15.2,
    "pct_rows_changed": 8.7, 
    "num_rules_applied": 23
  },
  "recommended_actions": [
    "Review flagged outliers in vehicle_price column",
    "Consider additional validation for owner_age field"
  ]
}
```

---

## 🧪 Testing

### **Run Test Suite**
```bash
# All tests
python -m pytest tests/ -v

# Unit tests only  
python -m pytest tests/unit/ -v

# Integration tests (requires sample data)
python -m pytest tests/integration/ -v

# Test specific agent
python -m pytest tests/unit/test_cleaning_agent.py -v
```

### **Test Coverage**
```bash
pip install pytest-cov
python -m pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html  # View coverage report
```

### **Synthetic Test Data**
RTGS includes synthetic datasets for testing each domain:

```bash
# Test transport analysis
python main.py run --source "tests/fixtures/synthetic_transport.csv" --domain-hint transport

# Test health analysis  
python main.py run --source "tests/fixtures/synthetic_health.csv" --domain-hint health

# Test education analysis
python main.py run --source "tests/fixtures/synthetic_education.csv" --domain-hint education
```

---

## 🔍 Troubleshooting

### **Common Issues & Solutions**

#### **🚨 Error: "LangSmith API key not found"**
```bash
# Solution: Set environment variable
export LANGSMITH_API_KEY="your-key"
# Or disable LangSmith in config
echo "observability:\n  langsmith_enabled: false" >> config/config.yaml
```

#### **🚨 Error: "Memory usage too high"**
```bash
# Solution: Reduce sample size or use streaming
python main.py run --source "large_file.csv" --sample-rows 100 --streaming
```

#### **🚨 Error: "Schema inference failed"**
```bash
# Solution: Check file encoding and format
python main.py run --source "data.csv" --encoding "utf-8" --separator ";" --dry-run
```

#### **🚨 Error: "Too many missing values"**
```bash
# Solution: Adjust thresholds or use permissive mode
python main.py run --source "messy_data.csv" --permissive --high-missingness-threshold 0.5
```

### **Debug Mode**
```bash
# Enable verbose logging
python main.py run --source "data.csv" --log-level DEBUG

# Save intermediate outputs  
python main.py run --source "data.csv" --save-intermediates

# Skip problematic agents
python main.py run --source "data.csv" --skip-agents "insight,memory"
```

### **Performance Optimization**
```bash
# For large datasets (>1M rows)
python main.py run --source "big_data.csv" --chunk-size 50000 --parallel

# Use faster LLM for development
python main.py run --source "data.csv" --llm-model "gpt-4o-mini"

# Skip expensive operations
python main.py run --source "data.csv" --skip-hypothesis-tests --skip-correlations
```

---

## 🚀 Advanced Features

### **🧠 Semantic Q&A** (Optional)
Query your processed datasets in natural language:

```bash
# Enable semantic search
python main.py run --source "data.csv" --enable-semantic-qa

# Ask questions about your data
python main.py query --run-id "rtgs-vehicles-20250905-001" \
  --question "Which districts have the highest vehicle registration growth?"
```

### **📊 Run Comparison**
Compare analyses across time periods or regions:

```bash
# Compare two runs
python main.py compare \
  --run1 "rtgs-vehicles-20250805-001" \
  --run2 "rtgs-vehicles-20250905-001" \
  --output "comparison_report.md"
```

### **🎛️ What-If Analysis** (Experimental)
Simulate policy scenarios:

```bash
# Simulate 10% budget increase impact
python main.py simulate \
  --run-id "rtgs-health-20250905-001" \
  --scenario "budget_increase=10%" \
  --metric "hospital_beds_per_capita"
```

### **📱 Interactive Dashboard** (Optional)
Launch a web interface for non-technical users:

```bash
# Start dashboard server
python dashboard.py --port 8080

# Access at http://localhost:8080
# Upload datasets, view reports, ask questions
```

---

## 🏆 Hackathon Demo Script

### **2-Minute Demo Flow**

#### **Minute 1: Problem Setup** 
"Government analysts spend weeks cleaning messy datasets like this Telangana vehicle registration data. Watch RTGS do it in 30 seconds..."

```bash
# Live demo command
python main.py run \
  --source "demo/telangana_vehicles_messy.csv" \
  --domain-hint transport \
  --scope "district=Hyderabad,year=2023"
```

#### **Minute 2: Results Showcase**
- **CLI Summary**: "Instant executive summary with confidence scores"
- **Interactive Charts**: "Click through Plotly visualizations" 
- **Audit Trail**: "Every transform is logged and traceable"
- **Policy Recommendations**: "AI-generated insights with statistical backing"

### **Judge Q&A Prep**

**Q: "How does this compare to existing tools?"**
A: "Unlike static BI tools, RTGS adapts to any government dataset structure and provides AI-powered insights with confidence scoring and full audit trails."

**Q: "What about data privacy?"**  
A: "All PII is detected and never sent to external LLMs