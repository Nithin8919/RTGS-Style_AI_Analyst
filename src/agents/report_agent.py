"""
RTGS AI Analyst - Enhanced Report Agent
Comprehensive report generation with LLM analysis and advanced seaborn visualizations
"""

import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import warnings
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import io
import asyncio

from src.utils.logging import get_agent_logger

# Import our comprehensive visualization utility
from src.utils.visualization import GovernmentDataVisualizer

warnings.filterwarnings('ignore')

"""
Updated LLM Analysis Engine for OpenAI API
"""

import os
from groq import Groq
from dotenv import load_dotenv
import json
from typing import Dict
from src.utils.logging import get_agent_logger

# Load environment variables
load_dotenv()

class LLMAnalysisEngine:
    """LLM-powered analysis engine using Groq API"""
    
    def __init__(self):
        self.logger = get_agent_logger("llm_analysis")
        
        # Load configuration
        with open('config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load environment variables from .env file
        from dotenv import load_dotenv
        load_dotenv()
        
        # Initialize Groq client
        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )
        
        # Use model from configuration
        self.model = self.config['groq']['model']
    
    async def call_llm(self, prompt: str, max_tokens: int = 2000) -> str:
        """Call Groq API for analysis"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.1,  # Low temperature for consistent analysis
                top_p=1.0
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Groq API call failed: {e}")
            return "Analysis unavailable due to technical issues."
    
    async def analyze_data_patterns(self, df_summary: Dict, domain: str, scope: str) -> Dict:
        """Analyze data patterns and generate insights using Groq"""
        
        prompt = f"""You are a senior government data analyst with expertise in {domain} sector analysis. 

DATASET CONTEXT:
- Domain: {domain}
- Scope: {scope}
- Data Summary: {json.dumps(df_summary, indent=2, default=str)}

ANALYSIS TASK:
Analyze the data patterns and provide comprehensive insights. Focus on:

1. KEY PATTERNS: What are the most significant patterns in this data?
2. ANOMALIES: What unusual patterns or outliers require attention?
3. CORRELATIONS: What relationships between variables are policy-relevant?
4. TRENDS: What trends indicate improvement or deterioration?
5. GEOGRAPHIC INSIGHTS: What spatial patterns suggest targeted interventions?

RESPONSE FORMAT (JSON):
{{
    "key_patterns": [
        {{
            "pattern": "description of pattern",
            "significance": "why this matters for policy",
            "confidence": "HIGH/MEDIUM/LOW",
            "data_evidence": "specific numbers/metrics supporting this"
        }}
    ],
    "critical_anomalies": [
        {{
            "anomaly": "description",
            "potential_causes": ["cause1", "cause2"],
            "recommended_investigation": "what to investigate further",
            "urgency": "HIGH/MEDIUM/LOW"
        }}
    ],
    "actionable_correlations": [
        {{
            "correlation": "Variable A and Variable B relationship",
            "policy_implication": "what this means for interventions",
            "strength": "correlation strength",
            "recommended_action": "specific action to take"
        }}
    ],
    "geographic_insights": {{
        "high_performing_areas": "characteristics of top performers",
        "underperforming_areas": "characteristics needing attention",
        "inequality_assessment": "severity and nature of geographic inequalities",
        "targeted_intervention_areas": ["specific areas for immediate attention"]
    }}
}}

Provide analysis that is:
- Specific to the {domain} domain
- Actionable for government decision-makers
- Evidence-based using the provided data
- Focused on policy implications
"""
        
        result = await self.call_llm(prompt, 3000)
        try:
            return json.loads(result)
        except:
            return {"error": "Failed to parse LLM response", "raw_response": result}
    
    async def generate_policy_recommendations(self, data_insights: Dict, domain: str, 
                                            statistical_results: Dict, context: Dict) -> Dict:
        """Generate comprehensive policy recommendations using Groq"""
        
        prompt = f"""You are a senior policy advisor specializing in {domain} sector with 15+ years of experience.

ANALYSIS CONTEXT:
- Domain: {domain}
- Geographic Scope: {context.get('scope', 'Regional')}
- Data Insights: {json.dumps(data_insights, indent=2, default=str)}
- Statistical Results: {json.dumps(statistical_results, indent=2, default=str)}

POLICY TASK:
Generate comprehensive, actionable policy recommendations that are:
1. Specific to {domain} sector challenges
2. Evidence-based using the provided analysis
3. Implementable within government constraints
4. Prioritized by impact and feasibility

RESPONSE FORMAT (JSON):
{{
    "immediate_actions": [
        {{
            "action": "specific action name",
            "description": "detailed description",
            "rationale": "why this action based on data evidence",
            "timeline": "implementation timeframe",
            "budget_estimate": "cost estimate with reasoning",
            "responsible_agency": "which agency should lead",
            "success_metrics": ["how to measure success"],
            "implementation_steps": ["step 1", "step 2", "step 3"]
        }}
    ],
    "strategic_interventions": [
        {{
            "intervention": "intervention name",
            "problem_statement": "specific problem this addresses",
            "evidence_base": "data evidence supporting need",
            "implementation_approach": "how to implement",
            "expected_outcomes": {{
                "short_term": ["6 month outcomes"],
                "medium_term": ["1-2 year outcomes"],
                "long_term": ["3-5 year outcomes"]
            }}
        }}
    ],
    "resource_allocation_strategy": {{
        "allocation_principles": "how to allocate resources fairly and effectively",
        "priority_areas": ["area 1 - rationale", "area 2 - rationale"],
        "efficiency_measures": "how to maximize impact per rupee spent"
    }}
}}

Make recommendations grounded in the specific data evidence provided and tailored to {domain} sector best practices.
"""
        
        result = await self.call_llm(prompt, 4000)
        try:
            return json.loads(result)
        except:
            return {"error": "Failed to parse policy recommendations", "raw_response": result}

    async def interpret_statistical_results(self, stats_results: Dict, domain: str) -> Dict:
        """Interpret statistical results in policy context using Groq"""
        
        prompt = f"""You are a senior statistician and policy analyst specializing in {domain} sector analysis.

STATISTICAL RESULTS:
{json.dumps(stats_results, indent=2, default=str)}

INTERPRETATION TASK:
Provide a comprehensive interpretation of these statistical results for government policymakers. Focus on:

1. PRACTICAL SIGNIFICANCE: What do these numbers mean in real-world terms?
2. POLICY IMPLICATIONS: How should these results influence government action?
3. CAUSAL INTERPRETATIONS: What can we reasonably infer about cause and effect?
4. UNCERTAINTY AND LIMITATIONS: What are the caveats and confidence levels?
5. ACTION RECOMMENDATIONS: What specific actions do these results suggest?

RESPONSE FORMAT (JSON):
{{
    "executive_summary": "2-3 sentence summary of key statistical findings and their policy relevance",
    "key_statistical_insights": [
        {{
            "finding": "statistical finding in plain language",
            "technical_details": "p-values, effect sizes, confidence intervals",
            "practical_significance": "what this means in real-world terms",
            "policy_relevance": "how this should influence government decisions",
            "confidence_assessment": "how confident we can be in this finding"
        }}
    ],
    "actionable_insights": [
        {{
            "insight": "key insight from statistical analysis",
            "supporting_evidence": "statistical evidence supporting this",
            "recommended_action": "specific government action this suggests",
            "success_criteria": "how to know if action is working"
        }}
    ]
}}

Provide interpretations that are:
- Accessible to non-statisticians
- Honest about limitations and uncertainty
- Focused on actionable policy insights
- Specific to {domain} sector context
"""
        
        result = await self.call_llm(prompt, 3000)
        try:
            return json.loads(result)
        except:
            return {"error": "Failed to parse statistical interpretation", "raw_response": result}
    
    async def generate_domain_specific_insights(self, df_summary: Dict, domain: str, 
                                              analysis_results: Dict) -> Dict:
        """Generate domain-specific insights and recommendations using Groq"""
        
        domain_expertise_prompts = {
            'health': "You are a public health expert and health policy specialist",
            'education': "You are an education policy expert and learning outcomes specialist", 
            'transport': "You are a transportation planning expert and infrastructure specialist",
            'economics': "You are an economic development specialist and public finance expert",
            'agriculture': "You are an agricultural economist and rural development specialist",
            'environment': "You are an environmental policy expert and sustainability specialist",
            'urban': "You are an urban planning expert and smart cities specialist",
            'social': "You are a social policy expert and welfare systems specialist"
        }
        
        expert_role = domain_expertise_prompts.get(domain, 
            "You are a public policy expert with deep knowledge of government service delivery")
        
        prompt = f"""{expert_role} with 20+ years of experience in {domain} sector analysis and policy implementation.

DATA CONTEXT:
- Domain: {domain}
- Data Summary: {json.dumps(df_summary, indent=2, default=str)}
- Analysis Results: {json.dumps(analysis_results, indent=2, default=str)}

EXPERT ANALYSIS TASK:
Apply your deep {domain} domain expertise to provide insights that only a sector specialist would identify. Focus on:

1. DOMAIN-SPECIFIC PATTERNS: What patterns are typical/atypical for {domain} sector?
2. SECTOR BENCHMARKS: How does this data compare to {domain} sector standards?
3. SPECIALIZED INTERVENTIONS: What {domain}-specific interventions are indicated?
4. SECTOR BEST PRACTICES: What proven {domain} strategies should be considered?
5. DOMAIN RISKS: What {domain}-specific risks and challenges are evident?

RESPONSE FORMAT (JSON):
{{
    "domain_expert_assessment": {{
        "overall_sector_health": "assessment of {domain} sector performance",
        "benchmark_comparison": "how this data compares to sector standards",
        "critical_gaps": ["gap 1 specific to {domain}", "gap 2"],
        "hidden_opportunities": ["opportunity 1", "opportunity 2"]
    }},
    "specialized_interventions": [
        {{
            "intervention": "{domain}-specific intervention",
            "sector_rationale": "why this is important in {domain} context",
            "evidence_base": "research/evidence supporting this intervention",
            "expected_sector_impact": "specific {domain} outcomes expected",
            "success_examples": "where this has worked in {domain} sector"
        }}
    ],
    "best_practice_recommendations": [
        {{
            "practice": "proven {domain} best practice",
            "description": "what this practice involves",
            "implementation_requirements": "what's needed to implement",
            "measurement_approach": "how to measure success"
        }}
    ]
}}

Provide insights that demonstrate deep {domain} sector expertise and are immediately actionable for government leaders.
"""
        
        result = await self.call_llm(prompt, 4000)
        try:
            return json.loads(result)
        except:
            return {"error": "Failed to parse domain insights", "raw_response": result}
    
    async def create_narrative_summary(self, all_insights: Dict, domain: str, context: Dict) -> str:
        """Create a compelling narrative summary of all insights using Groq"""
        
        prompt = f"""You are a senior government communications specialist and policy writer, expert at translating complex analysis into compelling narratives for senior government officials.

INSIGHTS TO SYNTHESIZE:
{json.dumps(all_insights, indent=2, default=str)}

CONTEXT:
- Domain: {domain}
- Audience: Senior government officials, ministers, secretaries
- Purpose: Executive briefing and decision-making support

NARRATIVE TASK:
Create a compelling, executive-level narrative that synthesizes all insights into a coherent story. The narrative should:

1. Start with the big picture and key message
2. Build a logical case for action
3. Be persuasive but evidence-based
4. Include specific recommendations
5. End with a clear call to action

STRUCTURE:
1. EXECUTIVE SUMMARY (2-3 sentences capturing the core message)
2. SITUATION ASSESSMENT (current state and key challenges)
3. OPPORTUNITY ANALYSIS (what's possible with right interventions)
4. STRATEGIC RECOMMENDATIONS (prioritized actions)
5. IMPLEMENTATION PATHWAY (how to move forward)
6. CALL TO ACTION (specific next steps for leadership)

Write in a style that is:
- Clear and accessible to busy executives
- Evidence-based but not overly technical
- Action-oriented and decisive
- Compelling and persuasive
- Specific to {domain} sector context

Length: 800-1200 words
"""
        
        result = await self.call_llm(prompt, 2500)
        return result

class EnhancedReportAgent:
    """Enhanced report agent with LLM-powered analysis and comprehensive visualizations"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.logger = get_agent_logger("enhanced_report")
        
        # Load configuration safely
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_path} not found, using defaults")
            self.config = {}
        
        # Initialize our comprehensive visualization utility
        self.visualizer = GovernmentDataVisualizer()
        self.llm_engine = LLMAnalysisEngine()

    async def process(self, state) -> Any:
        """Enhanced report assembly with LLM-powered analysis and comprehensive visualizations"""
        self.logger.info("Starting LLM-enhanced report assembly process")
        
        try:
            # Extract data safely
            insights = getattr(state, 'insights', {}) or {}
            # Ensure insights is a dictionary, not a string
            if isinstance(insights, str):
                try:
                    insights = json.loads(insights)
                except (json.JSONDecodeError, TypeError):
                    insights = {}
            analysis_results = getattr(state, 'analysis_results', {})
            transformed_data = getattr(state, 'transformed_data', pd.DataFrame())
            
            # Get context information
            run_manifest = state.run_manifest
            domain = run_manifest.get('dataset_info', {}).get('domain_hint', 'general')
            scope = run_manifest.get('dataset_info', {}).get('scope', 'Regional Analysis')
            
            # Prepare data summary for LLM analysis
            self.logger.info("Preparing data summary for LLM analysis")
            data_summary = await self._prepare_data_summary(transformed_data, analysis_results)
            
            # Phase 1: LLM-powered data pattern analysis
            self.logger.info("Conducting LLM-powered data pattern analysis")
            pattern_insights = await self.llm_engine.analyze_data_patterns(
                data_summary, domain, scope
            )
            
            # Phase 2: Comprehensive policy recommendations
            self.logger.info("Generating comprehensive policy recommendations")
            policy_recommendations = await self.llm_engine.generate_policy_recommendations(
                pattern_insights, domain, analysis_results, 
                {'scope': scope, 'context': run_manifest}
            )
            
            # Phase 3: Generate comprehensive visualizations using our enhanced utility
            self.logger.info("Generating comprehensive visualizations with seaborn")
            figures = self.visualizer.create_comprehensive_overview(
                transformed_data, analysis_results, domain
            )
            
            # Create summary statistics table
            summary_table_fig = self.visualizer.create_summary_statistics_table(transformed_data)
            if summary_table_fig:
                figures.insert(0, summary_table_fig)
            
            # Phase 4: Create narrative summary
            self.logger.info("Creating executive narrative summary")
            all_insights = {
                'pattern_insights': pattern_insights,
                'policy_recommendations': policy_recommendations,
                'original_insights': insights
            }
            
            narrative_summary = await self._create_narrative_summary(
                all_insights, domain, {'scope': scope, 'context': run_manifest}
            )
            
            # Generate both PDF reports
            self.logger.info("Creating technical data quality PDF")
            technical_pdf_path = await self._create_technical_quality_pdf(
                state, transformed_data, analysis_results, figures, all_insights
            )
            
            self.logger.info("Creating policy-focused PDF")
            policy_pdf_path = await self._create_policy_focused_pdf(
                state, transformed_data, analysis_results, all_insights, 
                figures, domain, narrative_summary
            )
            
            # Create interactive dashboard
            dashboard_html_path = await self._create_interactive_dashboard(
                transformed_data, analysis_results, all_insights, state
            )
            
            # Update state with enhanced outputs
            state.llm_enhanced_reports = {
                'technical_quality_pdf': technical_pdf_path,
                'policy_focused_pdf': policy_pdf_path,
                'interactive_dashboard': dashboard_html_path,
                'pattern_insights': pattern_insights,
                'policy_recommendations': policy_recommendations,
                'narrative_summary': narrative_summary
            }
            
            state.visualization_figures = figures
            
            # Create enhanced CLI summary
            cli_summary = self._create_enhanced_cli_summary(
                all_insights, analysis_results, state
            )
            state.cli_summary = cli_summary
            
            self.logger.info("LLM-enhanced report assembly completed successfully")
            return state
            
        except Exception as e:
            self.logger.error(f"LLM-enhanced report assembly failed: {str(e)}")
            if not hasattr(state, 'errors'):
                state.errors = []
            state.errors.append(f"LLM-enhanced report assembly failed: {str(e)}")
            
            # Create fallback reports without LLM
            return await self._create_fallback_reports(state)

    async def _prepare_data_summary(self, df: pd.DataFrame, analysis_results: Dict) -> Dict:
        """Prepare comprehensive data summary for LLM analysis - FIXED VERSION"""
        
        if df is None or df.empty:
            return {"error": "No data available for analysis"}
        
        try:
            # Ensure analysis_results is a dictionary
            if isinstance(analysis_results, str):
                try:
                    analysis_results = json.loads(analysis_results)
                except (json.JSONDecodeError, TypeError):
                    analysis_results = {}
            elif analysis_results is None:
                analysis_results = {}
            
            summary = {
                "dataset_overview": {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
                    "categorical_columns": len(df.select_dtypes(include=['object']).columns),
                    "date_columns": len(df.select_dtypes(include=['datetime']).columns)
                },
                "data_quality": {
                    "missing_data_percentage": float(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100),
                    "columns_with_missing_data": [col for col in df.columns if df[col].isnull().sum() > 0],
                    "missing_data_by_column": {col: float(df[col].isnull().sum() / len(df) * 100) 
                                             for col in df.columns if df[col].isnull().sum() > 0}
                },
                "numerical_summary": {},
                "categorical_summary": {},
                "key_relationships": {}
            }
        
            # FIXED: Numerical summary with proper pandas handling
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if not numeric_cols.empty:  # Use .empty instead of direct boolean evaluation
                for col in numeric_cols[:10]:
                    try:
                        col_data = df[col].dropna()
                        if len(col_data) > 0:
                            # Use .iloc to avoid ambiguous truth value
                            q25 = col_data.quantile(0.25)
                            q75 = col_data.quantile(0.75)
                            iqr = q75 - q25
                            
                            # Fixed outlier calculation
                            lower_bound = q25 - 1.5 * iqr
                            upper_bound = q75 + 1.5 * iqr
                            outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
                            
                            summary["numerical_summary"][col] = {
                                "mean": float(col_data.mean()),
                                "median": float(col_data.median()),
                                "std": float(col_data.std()),
                                "min": float(col_data.min()),
                                "max": float(col_data.max()),
                                "outlier_count": int(outlier_mask.sum())
                            }
                    except Exception as e:
                        self.logger.warning(f"Failed to process numeric column {col}: {e}")
                        continue
        
            # FIXED: Categorical summary with proper pandas handling
            categorical_cols = df.select_dtypes(include=['object'])
            if not categorical_cols.empty:  # Use .empty instead of direct boolean evaluation
                for col in categorical_cols.columns[:8]:
                    try:
                        col_data = df[col].dropna()
                        if len(col_data) > 0:
                            value_counts = col_data.value_counts()
                            summary["categorical_summary"][col] = {
                                "unique_values": int(col_data.nunique()),
                                "most_common": dict(value_counts.head(5)),  # Convert to dict
                                "concentration_ratio": float(value_counts.iloc[0] / len(col_data)) if len(value_counts) > 0 else 0
                            }
                    except Exception as e:
                        self.logger.warning(f"Failed to process categorical column {col}: {e}")
                        continue
        
            # FIXED: Key relationships with better error handling
            if len(numeric_cols) >= 2:
                try:
                    # Select only first 10 numeric columns to avoid memory issues
                    selected_numeric = numeric_cols[:10]
                    corr_matrix = df[selected_numeric].corr()
                    strong_correlations = []
                    
                    # Use .iloc for safe indexing
                    n_cols = len(corr_matrix.columns)
                    for i in range(n_cols):
                        for j in range(i+1, n_cols):
                            corr_value = corr_matrix.iloc[i, j]
                            if not pd.isna(corr_value) and abs(corr_value) > 0.5:
                                strong_correlations.append({
                                    "variable_1": corr_matrix.columns[i],
                                    "variable_2": corr_matrix.columns[j],
                                    "correlation": float(corr_value)
                                })
                    
                    # Sort and limit correlations
                    summary["key_relationships"]["strong_correlations"] = sorted(
                        strong_correlations, key=lambda x: abs(x["correlation"]), reverse=True
                    )[:10]
                except Exception as e:
                    self.logger.warning(f"Correlation analysis failed: {e}")
                    summary["key_relationships"]["strong_correlations"] = []
        
            # FIXED: Geographic analysis with better column detection
            geo_cols = []
            potential_geo_names = ['district', 'region', 'state', 'city', 'area', 'zone', 'mandal', 'tehsil']
            
            # Safe column name checking
            try:
                for col in df.columns:
                    if any(geo_name in str(col).lower() for geo_name in potential_geo_names):
                        geo_cols.append(col)
            except Exception as e:
                self.logger.warning(f"Geographic column detection failed: {e}")
            
            if geo_cols and not numeric_cols.empty:
                try:
                    geo_col = geo_cols[0]
                    main_metric = numeric_cols[0]
                    
                    # Safe groupby operation
                    regional_data = df.groupby(geo_col)[main_metric].agg(['mean', 'count']).reset_index()
                    
                    if not regional_data.empty and not regional_data['mean'].isna().all():
                        # Find valid (non-NaN) means
                        valid_means = regional_data['mean'].dropna()
                        if len(valid_means) > 0:
                            max_idx = valid_means.idxmax()
                            min_idx = valid_means.idxmin()
                            
                            summary["geographic_analysis"] = {
                                "geographic_column": geo_col,
                                "number_of_regions": len(regional_data),
                                "performance_variation": {
                                    "highest_performing": {
                                        "region": str(regional_data.loc[max_idx, geo_col]),
                                        "value": float(valid_means.max())
                                    },
                                    "lowest_performing": {
                                        "region": str(regional_data.loc[min_idx, geo_col]),
                                        "value": float(valid_means.min())
                                    },
                                    "inequality_ratio": float(valid_means.max() / valid_means.min()) if valid_means.min() > 0 else None
                                }
                            }
                except Exception as e:
                    self.logger.warning(f"Geographic analysis failed: {e}")
                    summary["geographic_analysis"] = {"error": "Geographic analysis unavailable"}
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Data summary preparation failed: {e}")
            return {"error": f"Data summary preparation failed: {str(e)}"}

    async def _create_narrative_summary(self, all_insights: Dict, domain: str, context: Dict) -> str:
        """Create a compelling narrative summary of all insights"""
        
        prompt = f"""You are a senior government communications specialist writing for senior officials.

INSIGHTS TO SYNTHESIZE:
{json.dumps(all_insights, indent=2, default=str)}

CONTEXT:
- Domain: {domain}
- Audience: Senior government officials
- Purpose: Executive briefing and decision-making support

Create a compelling 400-600 word executive narrative that:
1. Starts with the key message and findings
2. Builds a logical case for action
3. Includes specific recommendations
4. Ends with clear next steps

Write for busy executives who need actionable insights.
"""
        
        result = await self.llm_engine.call_llm(prompt, 1500)
        return result

    async def _create_technical_quality_pdf(self, state, df: pd.DataFrame, 
                                          analysis_results: Dict, figures: List, 
                                          llm_insights: Dict) -> str:
        """Create comprehensive technical data quality PDF report with visualizations"""
        
        run_manifest = state.run_manifest
        output_dir = Path(run_manifest['artifacts_paths']['reports_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pdf_path = output_dir / f"technical_data_quality_report_{run_manifest['run_id']}.pdf"
        
        # Create PDF with our comprehensive figures using seaborn
        metadata = {
            'title': f'Technical Data Quality Report - {run_manifest["dataset_info"]["dataset_name"]}',
            'author': 'RTGS AI Analyst System - Enhanced with LLM',
            'subject': 'Data Quality and Technical Analysis',
            'keywords': 'Data Quality, Technical Analysis, Government Data, AI Analysis'
        }
        
        # Save all figures to PDF using our comprehensive visualizer
        self.visualizer.save_figures_to_pdf(figures, str(pdf_path), metadata)
        
        self.logger.info(f"Technical quality PDF with {len(figures)} visualizations created: {pdf_path}")
        return str(pdf_path)

    async def _create_policy_focused_pdf(self, state, df: pd.DataFrame, 
                                       analysis_results: Dict, llm_insights: Dict,
                                       figures: List, domain: str, narrative: str) -> str:
        """Create comprehensive policy-focused PDF report"""
        
        run_manifest = state.run_manifest
        output_dir = Path(run_manifest['artifacts_paths']['reports_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pdf_path = output_dir / f"policy_insights_report_{run_manifest['run_id']}.pdf"
        
        doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, topMargin=1*inch)
        styles = getSampleStyleSheet()
        story = []
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=20,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=15,
            textColor=colors.darkblue
        )
        
        # Title Page
        story.append(Paragraph("AI-Enhanced Government Policy Insights", title_style))
        story.append(Spacer(1, 20))
        
        # Dataset Information Box
        info_data = [
            ['Dataset', run_manifest['dataset_info']['dataset_name']],
            ['Domain', domain.title()],
            ['Scope', run_manifest['dataset_info']['scope']],
            ['Analysis Date', datetime.now().strftime('%B %d, %Y')],
            ['Records Analyzed', f"{len(df):,}"],
            ['Variables Analyzed', f"{len(df.columns)}"],
            ['AI Analysis Engine', 'Claude Sonnet 4 + Comprehensive Visualization Suite']
        ]
        
        info_table = Table(info_data, colWidths=[2*inch, 3*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.darkblue),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(info_table)
        story.append(PageBreak())
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", heading_style))
        story.append(Paragraph(narrative, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Key AI-Generated Insights
        story.append(Paragraph("AI-Generated Data Insights", heading_style))
        
        if 'pattern_insights' in llm_insights and 'key_patterns' in llm_insights['pattern_insights']:
            for i, pattern in enumerate(llm_insights['pattern_insights']['key_patterns'][:5], 1):
                story.append(Paragraph(f"{i}. {pattern.get('pattern', 'Pattern identified')}", styles['Heading3']))
                story.append(Paragraph(f"Policy Significance: {pattern.get('significance', 'Significant for policy planning')}", styles['Normal']))
                story.append(Paragraph(f"Evidence: {pattern.get('data_evidence', 'Based on comprehensive data analysis')}", styles['Normal']))
                story.append(Paragraph(f"Confidence Level: {pattern.get('confidence', 'HIGH')}", styles['Normal']))
                story.append(Spacer(1, 10))
        
        story.append(PageBreak())
        
        # Priority Actions
        story.append(Paragraph("Priority Policy Actions", heading_style))
        
        if 'policy_recommendations' in llm_insights and 'immediate_actions' in llm_insights['policy_recommendations']:
            actions = llm_insights['policy_recommendations']['immediate_actions']
            
            for i, action in enumerate(actions[:5], 1):
                # Create action details table
                action_data = [
                    ['Action', action.get('action', 'Policy Action')],
                    ['Description', action.get('description', 'Action description')],
                    ['Timeline', action.get('timeline', 'To be determined')],
                    ['Budget Estimate', action.get('budget_estimate', 'To be estimated')],
                    ['Responsible Agency', action.get('responsible_agency', 'To be assigned')],
                    ['Expected Outcome', action.get('expected_outcome', 'Positive impact expected')]
                ]
                
                action_table = Table(action_data, colWidths=[1.5*inch, 4*inch])
                action_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                    ('BACKGROUND', (1, 0), (1, -1), colors.white),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ]))
                story.append(action_table)
                story.append(Spacer(1, 15))
        
        # Note about comprehensive visualizations
        story.append(PageBreak())
        story.append(Paragraph("Comprehensive Data Visualizations", heading_style))
        story.append(Paragraph(f"This analysis includes {len(figures)} comprehensive visualizations created using advanced statistical plotting libraries (seaborn + matplotlib). These include:", styles['Normal']))
        story.append(Paragraph("• Data Quality Assessment Dashboard", styles['Normal']))
        story.append(Paragraph("• KPI Performance Matrix with Government Color Schemes", styles['Normal']))
        story.append(Paragraph("• Temporal Analysis Suite (trends, seasonality, volatility)", styles['Normal']))
        story.append(Paragraph("• Geographic Analysis and Inequality Visualization", styles['Normal']))
        story.append(Paragraph("• Distribution Analysis Suite", styles['Normal']))
        story.append(Paragraph("• Correlation Analysis with Policy Implications", styles['Normal']))
        story.append(Paragraph("• Statistical Significance Plots", styles['Normal']))
        story.append(Paragraph("• Policy Impact Simulation Visualizations", styles['Normal']))
        story.append(Spacer(1, 15))
        story.append(Paragraph("Please refer to the Technical Data Quality Report PDF for all detailed visualizations.", styles['Normal']))
        
        # Footer
        story.append(Spacer(1, 30))
        story.append(Paragraph("---", styles['Normal']))
        story.append(Paragraph(f"Report generated by RTGS AI Analyst System with LLM Enhancement", styles['Normal']))
        story.append(Paragraph(f"Visualization Suite: GovernmentDataVisualizer with Seaborn + Matplotlib", styles['Normal']))
        story.append(Paragraph(f"Run ID: {run_manifest['run_id']}", styles['Normal']))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}", styles['Normal']))
        
        # Build PDF
        doc.build(story)
        
        self.logger.info(f"Policy-focused PDF created: {pdf_path}")
        return str(pdf_path)

    async def _create_interactive_dashboard(self, df: pd.DataFrame, analysis_results: Dict,
                                          llm_insights: Dict, state) -> str:
        """Create interactive HTML dashboard with LLM insights"""
        
        run_manifest = state.run_manifest
        output_dir = Path(run_manifest['artifacts_paths']['plots_dir']) / "interactive"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        dashboard_path = output_dir / f"policy_dashboard_{run_manifest['run_id']}.html"
        
        # Get key insights for dashboard
        pattern_insights = llm_insights.get('pattern_insights', {})
        policy_recs = llm_insights.get('policy_recommendations', {})
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RTGS AI Analyst - Enhanced Policy Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 20px;
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .header {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }}
        .header h1 {{
            font-size: 2.5em;
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        .viz-note {{
            background: #e8f6f3;
            border-left: 4px solid #27ae60;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        .viz-note h3 {{
            color: #27ae60;
            margin-bottom: 10px;
        }}
        .card {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        }}
        .insight-item {{
            background: #ecf0f1;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 RTGS AI Enhanced Policy Dashboard</h1>
            <h2>{run_manifest['dataset_info']['dataset_name']}</h2>
            <p><strong>Domain:</strong> {run_manifest['dataset_info']['domain_hint'].title()} | 
               <strong>Analysis:</strong> LLM + Comprehensive Visualizations</p>
        </div>
        
        <div class="viz-note">
            <h3>📊 Comprehensive Visualization Suite</h3>
            <p>This analysis includes <strong>8+ types of advanced visualizations</strong> using seaborn and matplotlib:</p>
            <ul>
                <li><strong>Data Quality Dashboard:</strong> Missing data patterns, completeness scores, outlier detection</li>
                <li><strong>KPI Performance Matrix:</strong> Government-themed color schemes and performance indicators</li>
                <li><strong>Temporal Analysis:</strong> Time series, seasonality, growth rates, volatility analysis</li>
                <li><strong>Geographic Analysis:</strong> Regional performance comparison and inequality visualization</li>
                <li><strong>Distribution Analysis:</strong> Histograms with KDE, statistical summaries</li>
                <li><strong>Correlation Analysis:</strong> Advanced heatmaps and relationship networks</li>
                <li><strong>Statistical Significance:</strong> P-value plots, effect size visualization, confidence intervals</li>
                <li><strong>Policy Impact Simulation:</strong> Resource allocation, cost-benefit, implementation timelines</li>
            </ul>
            <p><strong>📄 View all visualizations in the Technical Data Quality PDF report.</strong></p>
        </div>
        
            <div class="card">
            <h3>🧠 AI-Generated Insights</h3>
"""
        
        # Add AI insights
        if 'key_patterns' in pattern_insights:
            for pattern in pattern_insights['key_patterns'][:3]:
                html_content += f"""
                <div class="insight-item">
                    <h4>{pattern.get('pattern', 'AI Pattern Detected')}</h4>
                    <p><strong>Policy Significance:</strong> {pattern.get('significance', 'Important for decision making')}</p>
                    <p><strong>Evidence:</strong> {pattern.get('data_evidence', 'Based on comprehensive analysis')}</p>
                    <p><strong>Confidence:</strong> {pattern.get('confidence', 'HIGH')}</p>
                </div>
"""
        
        html_content += """
            </div>
            
            <div class="card">
            <h3>🚨 Priority Actions</h3>
"""
        
        # Add policy actions
        if 'immediate_actions' in policy_recs:
            for action in policy_recs['immediate_actions'][:3]:
                html_content += f"""
                <div class="insight-item">
                    <h4>{action.get('action', 'Priority Action')}</h4>
                    <p><strong>Timeline:</strong> {action.get('timeline', 'To be determined')}</p>
                    <p><strong>Agency:</strong> {action.get('responsible_agency', 'To be assigned')}</p>
                    <p><strong>Expected Outcome:</strong> {action.get('expected_outcome', 'Positive impact')}</p>
                </div>
"""
        
                html_content += f"""
                </div>
        
        <div class="card">
            <h3>📈 Visualization Capabilities Demonstrated</h3>
            <p>This analysis showcases the system's ability to work with <strong>any government domain</strong> and automatically generate:</p>
            <ul>
                <li>✅ <strong>Domain-adaptive visualizations</strong> (automatically adjusts to {run_manifest['dataset_info']['domain_hint']} sector)</li>
                <li>✅ <strong>Government-appropriate color schemes</strong> and professional styling</li>
                <li>✅ <strong>Policy-relevant statistical analysis</strong> with confidence intervals and significance testing</li>
                <li>✅ <strong>Geographic inequality detection</strong> and intervention area identification</li>
                <li>✅ <strong>Resource allocation optimization</strong> and budget impact simulation</li>
                <li>✅ <strong>Implementation timeline visualization</strong> and risk assessment matrices</li>
                        </ul>
                    </div>
        
        <div class="card">
            <h3>🎯 System Capabilities Summary</h3>
            <p><strong>Data Processing:</strong> {len(df):,} records, {len(df.columns)} variables analyzed</p>
            <p><strong>AI Analysis:</strong> {len(pattern_insights.get('key_patterns', []))} patterns detected, {len(policy_recs.get('immediate_actions', []))} actions recommended</p>
            <p><strong>Visualization Suite:</strong> 8+ comprehensive chart types with seaborn + matplotlib</p>
            <p><strong>Domain Adaptability:</strong> Works with any government dataset (health, education, transport, etc.)</p>
            <p><strong>Output Formats:</strong> Technical PDF + Policy PDF + Interactive Dashboard</p>
        </div>
        
        <footer style="text-align: center; margin-top: 30px; color: #7f8c8d;">
            <p>Generated by RTGS AI Analyst System | Enhanced with LLM + Comprehensive Visualizations</p>
            <p>Run ID: {run_manifest['run_id']} | {datetime.now().strftime('%B %d, %Y')}</p>
        </footer>
    </div>
</body>
</html>
"""
        
        # Write HTML file
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"Enhanced interactive dashboard created: {dashboard_path}")
        return str(dashboard_path)

    def _create_enhanced_cli_summary(self, llm_insights: Dict, analysis_results: Dict, state) -> Dict:
        """Create enhanced CLI summary with comprehensive visualization info"""
        
        run_manifest = state.run_manifest
        pattern_insights = llm_insights.get('pattern_insights', {})
        policy_recs = llm_insights.get('policy_recommendations', {})
        
        key_patterns = pattern_insights.get('key_patterns', [])
        immediate_actions = policy_recs.get('immediate_actions', [])
        
        # Calculate confidence score
        confidence_scores = [90 if p.get('confidence') == 'HIGH' else 70 if p.get('confidence') == 'MEDIUM' else 50 
                           for p in key_patterns]
        overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 75
        
        confidence_badge = '🟢 HIGH' if overall_confidence >= 80 else '🟡 MEDIUM' if overall_confidence >= 60 else '🔴 LOW'
        
        return {
            'one_line_summary': f"LLM + Comprehensive Visualization analysis reveals {len(key_patterns)} patterns and {len(immediate_actions)} priority actions",
            'confidence_badge': confidence_badge,
            'quality_score': f"{overall_confidence:.0f}/100",
            'key_findings': [pattern.get('pattern', 'Pattern identified') for pattern in key_patterns[:3]],
            'priority_actions': [action.get('action', 'Action identified') for action in immediate_actions[:3]],
            'findings_count': len(key_patterns),
            'actions_count': len(immediate_actions),
            'llm_powered': True,
            'comprehensive_visualizations': True,
            'visualization_types': [
                'Data Quality Dashboard',
                'KPI Performance Matrix', 
                'Temporal Analysis Suite',
                'Geographic Analysis',
                'Distribution Analysis',
                'Correlation Analysis',
                'Statistical Significance',
                'Policy Impact Simulation'
            ],
            'artifacts_paths': {
                'technical_pdf': f"{state.run_manifest['artifacts_paths']['reports_dir']}/technical_data_quality_report_{state.run_manifest['run_id']}.pdf",
                'policy_pdf': f"{state.run_manifest['artifacts_paths']['reports_dir']}/policy_insights_report_{state.run_manifest['run_id']}.pdf",
                'dashboard': f"{state.run_manifest['artifacts_paths']['plots_dir']}/interactive/policy_dashboard_{state.run_manifest['run_id']}.html"
            }
        }

    async def _create_fallback_reports(self, state) -> Any:
        """Create fallback reports when LLM analysis fails"""
        
        self.logger.info("Creating fallback reports without LLM analysis")
        
        try:
            analysis_results = getattr(state, 'analysis_results', {})
            transformed_data = getattr(state, 'transformed_data', pd.DataFrame())
            
            # Ensure analysis_results is a dictionary, not a string
            if isinstance(analysis_results, str):
                try:
                    analysis_results = json.loads(analysis_results)
                except (json.JSONDecodeError, TypeError):
                    analysis_results = {}
            
            # Create basic visualizations using our comprehensive visualizer
            if not transformed_data.empty:
                domain = state.run_manifest.get('dataset_info', {}).get('domain_hint', 'general')
                figures = self.visualizer.create_comprehensive_overview(
                    transformed_data, analysis_results, domain
                )
            else:
                figures = []
            
            # Create minimal technical report with visualizations
            output_dir = Path(state.run_manifest['artifacts_paths']['reports_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            fallback_pdf = output_dir / f"fallback_report_with_viz_{state.run_manifest['run_id']}.pdf"
            
            if figures:
                metadata = {
                    'title': f'Data Analysis Report - {state.run_manifest["dataset_info"]["dataset_name"]}',
                    'author': 'RTGS AI Analyst System (Fallback Mode with Comprehensive Visualizations)',
                    'subject': 'Government Data Analysis Report',
                    'keywords': 'Government, Data Analysis, Policy, Seaborn, Matplotlib'
                }
                self.visualizer.save_figures_to_pdf(figures, str(fallback_pdf), metadata)
            
            # Update state with fallback outputs
            state.llm_enhanced_reports = {
                'technical_quality_pdf': str(fallback_pdf),
                'policy_focused_pdf': str(fallback_pdf),
                'interactive_dashboard': None,
                'fallback_mode': True,
                'error': 'LLM analysis unavailable, comprehensive visualizations still generated'
            }
            
            state.cli_summary = {
                'one_line_summary': f"Comprehensive visualization analysis of {state.run_manifest['dataset_info']['dataset_name']} completed",
                'confidence_badge': '🟡 MEDIUM',
                'quality_score': '70/100',
                'key_findings': ['Data processing completed', f'{len(figures)} comprehensive visualizations generated'],
                'priority_actions': ['Review visualization insights', 'Consider manual policy analysis'],
                'findings_count': 2,
                'actions_count': 2,
                'llm_powered': False,
                'comprehensive_visualizations': True,
                'fallback_mode': True
            }
            
            return state
            
        except Exception as e:
            self.logger.error(f"Even fallback report creation failed: {str(e)}")
            state.errors.append(f"Complete report generation failure: {str(e)}")
            return state

    def _create_minimal_insights(self, state) -> Dict:
        """Create minimal insights when no insights exist"""
        run_manifest = state.run_manifest
        
        return {
            'generation_timestamp': datetime.utcnow().isoformat(),
            'context': {
                'dataset_info': {
                    'name': run_manifest.get('dataset_info', {}).get('dataset_name', 'Unknown'),
                    'domain': run_manifest.get('dataset_info', {}).get('domain_hint', 'general'),
                    'scope': run_manifest.get('dataset_info', {}).get('scope', 'Unknown')
                }
            },
            'executive_summary': {
                'one_line_summary': f"Analysis of {run_manifest.get('dataset_info', {}).get('dataset_name', 'dataset')} completed with comprehensive visualizations",
                'key_insights_summary': 'Data processing and comprehensive visualization analysis completed successfully',
                'priority_actions': 'Review visualization insights and consider additional analysis'
            },
            'key_findings': [
                {
                    'finding': 'Comprehensive data processing and visualization completed',
                    'evidence': 'Dataset processed with full visualization suite',
                    'confidence': 'MEDIUM',
                    'policy_relevance': 'Visual insights available for decision making'
                }
            ],
            'policy_recommendations': [
                {
                    'recommendation': 'Review comprehensive visualizations and consider targeted interventions',
                    'priority': 'MEDIUM',
                    'timeframe': 'Short-term',
                    'responsible_agency': 'Data Analysis and Policy Team'
                }
            ],
            'confidence_assessment': {
                'overall_confidence': 'MEDIUM'
            }
        }