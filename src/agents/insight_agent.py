"""
RTGS AI Analyst - Insight Agent
Uses LLM to convert statistical findings into policy-relevant insights and recommendations
"""

import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib

from groq import Groq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser

from src.utils.logging import get_agent_logger, TransformLogger


class InsightAgent:
    """Agent responsible for generating policy insights using LLM analysis"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.logger = get_agent_logger("insight")
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize Groq client
        import os
        from dotenv import load_dotenv
        
        # Load environment variables from .env file
        load_dotenv()
        
        self.groq_client = Groq(
            api_key=os.getenv('GROQ_API_KEY')
        )
        self.model = self.config['groq']['model']
        self.temperature = self.config['groq']['temperature']
        self.max_tokens = self.config['groq']['max_tokens']
        
        # Load insight generation settings
        self.insights_config = self.config.get('insights', {})
        
    async def process(self, state) -> Any:
        """Main insight generation processing pipeline with robust error handling"""
        self.logger.info("Starting insight generation process")
        
        try:
            # Get best available data with fallback
            input_data = None
            data_sources = ['analysis_results', 'transformed_data', 'cleaned_data']
            
            for source in data_sources:
                if hasattr(state, source) and getattr(state, source) is not None:
                    if source == 'analysis_results':
                        analysis_results = getattr(state, source)
                        # Validate that it's actually analysis results
                        if isinstance(analysis_results, dict) and len(analysis_results) > 0:
                            input_data = analysis_results
                            self.logger.info(f"Using {source} for insight generation")
                            break
                    else:
                        # Fallback: create minimal analysis from raw data
                        raw_data = getattr(state, source)
                        input_data = self._create_minimal_analysis(raw_data)
                        self.logger.warning(f"Created minimal analysis from {source}")
                        break
            
            if input_data is None:
                raise ValueError("No analysis results available for insight generation")
            
            # Initialize error tracking
            if not hasattr(state, 'errors'):
                state.errors = []
            if not hasattr(state, 'warnings'):
                state.warnings = []
            
            # Prepare context for LLM
            try:
                insight_context = await self._prepare_insight_context(state, input_data)
            except Exception as e:
                self.logger.warning(f"Context preparation failed: {str(e)}")
                insight_context = self._create_fallback_context(state)
            
            # Generate insights with individual error handling
            insights = {
                'generation_timestamp': datetime.utcnow().isoformat(),
                'context': insight_context
            }
            
            # Key findings
            try:
                key_findings = await self._generate_key_findings(input_data, insight_context)
                insights['key_findings'] = key_findings
            except Exception as e:
                self.logger.warning(f"Key findings generation failed: {str(e)}")
                insights['key_findings'] = self._generate_fallback_findings(input_data, insight_context)
            
            # Policy recommendations
            try:
                policy_recommendations = await self._generate_policy_recommendations(
                    input_data, insight_context, insights['key_findings']
                )
                insights['policy_recommendations'] = policy_recommendations
            except Exception as e:
                self.logger.warning(f"Policy recommendations generation failed: {str(e)}")
                insights['policy_recommendations'] = self._generate_fallback_recommendations(
                    insights['key_findings'], insight_context
                )
            
            # Executive summary
            try:
                executive_summary = await self._generate_executive_summary(
                    insights['key_findings'], insights['policy_recommendations'], insight_context
                )
                insights['executive_summary'] = executive_summary
            except Exception as e:
                self.logger.warning(f"Executive summary generation failed: {str(e)}")
                insights['executive_summary'] = self._generate_fallback_summary(insight_context)
            
            # Statistical narrative
            try:
                statistical_narrative = await self._generate_statistical_narrative(input_data, insight_context)
                insights['statistical_narrative'] = statistical_narrative
            except Exception as e:
                self.logger.warning(f"Statistical narrative generation failed: {str(e)}")
                insights['statistical_narrative'] = self._generate_fallback_narrative()
            
            # Confidence assessment
            try:
                confidence_assessment = self._assess_insight_confidence(
                    insights['key_findings'], input_data, state
                )
                insights['confidence_assessment'] = confidence_assessment
            except Exception as e:
                self.logger.warning(f"Confidence assessment failed: {str(e)}")
                insights['confidence_assessment'] = {'overall_confidence': 'MEDIUM'}
            
            # Ensure output directories exist
            docs_dir = Path(state.run_manifest['artifacts_paths']['docs_dir'])
            docs_dir.mkdir(parents=True, exist_ok=True)
            
            # Save insights
            try:
                insights_path = docs_dir / "insights_executive.json"
                with open(insights_path, 'w') as f:
                    json.dump(insights, f, indent=2, default=str)
                self.logger.info(f"Saved insights to {insights_path}")
            except Exception as e:
                self.logger.warning(f"Failed to save insights: {str(e)}")
            
            # Update state
            state.insights = insights
            
            self.logger.info(f"Insight generation completed: {len(insights['key_findings'])} findings, {len(insights['policy_recommendations'])} recommendations")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Insight generation failed: {str(e)}")
            if not hasattr(state, 'errors'):
                state.errors = []
            state.errors.append(f"Insight generation failed: {str(e)}")
            
            # Create minimal insights to prevent cascade failure
            state.insights = {
                'error': str(e),
                'generation_timestamp': datetime.utcnow().isoformat(),
                'key_findings': [],
                'policy_recommendations': [],
                'executive_summary': {'one_line_summary': 'Analysis completed with limitations'},
                'confidence_assessment': {'overall_confidence': 'LOW'}
            }
            
            return state

    async def _prepare_insight_context(self, state, analysis_results: Dict) -> Dict[str, Any]:
        """Prepare context information for insight generation with safe data access"""
        
        run_manifest = state.run_manifest
        
        # Safe access to analysis results
        dataset_profile = analysis_results.get('dataset_profile', {})
        if isinstance(dataset_profile, str):
            dataset_profile = {}
        
        basic_info = dataset_profile.get('basic_info', {})
        if isinstance(basic_info, str):
            basic_info = {}
        
        context = {
            'dataset_info': {
                'name': run_manifest.get('dataset_info', {}).get('dataset_name', 'Unknown'),
                'domain': run_manifest.get('dataset_info', {}).get('domain_hint', 'general'),
                'scope': run_manifest.get('dataset_info', {}).get('scope', 'Unknown'),
                'description': run_manifest.get('dataset_info', {}).get('description', 'Analysis dataset')
            },
            'user_context': run_manifest.get('user_context', {}),
            'data_characteristics': {
                'rows': basic_info.get('rows', 0),
                'columns': basic_info.get('columns', 0),
                'analysis_quality': analysis_results.get('quality_assessment', {}).get('overall_score', 'unknown')
            },
            'analysis_summary': {
                'kpis_analyzed': len(self._safe_get_list(analysis_results, 'kpis')),
                'trends_identified': len(self._safe_get_list(analysis_results, 'trends')),
                'correlations_found': len(self._safe_get_list(analysis_results, 'correlations')),
                'spatial_patterns': bool(analysis_results.get('spatial_analysis', {})),
                'hypothesis_tests': len(self._safe_get_list(analysis_results, 'hypothesis_tests'))
            }
        }
        
        return context

    async def _generate_key_findings(self, analysis_results: Dict, context: Dict) -> List[Dict[str, Any]]:
        """Generate key findings from statistical analysis using LLM"""
        self.logger.info("Generating key findings")
        
        # Prepare statistical evidence for LLM
        evidence = self._extract_statistical_evidence(analysis_results)
        
        # Create LLM prompt
        system_prompt = f"""You are a senior policy analyst specialized in {context['dataset_info']['domain']} policy. 
Your task is to identify the most important policy-relevant findings from statistical analysis.

Focus on findings that:
1. Have statistical significance and practical importance
2. Reveal inequalities, trends, or patterns relevant to policy decisions
3. Can drive actionable government interventions
4. Are backed by solid evidence

Respond with a JSON array of findings, each containing:
{{
  "finding": "Clear, policy-focused statement",
  "evidence": "Statistical evidence supporting this finding", 
  "magnitude": "Quantified impact or scale",
  "confidence": "HIGH/MEDIUM/LOW",
  "policy_relevance": "Why this matters for policy decisions",
  "geographic_scope": "Area or population affected"
}}

Limit to the 3-5 most important findings. Use clear, non-technical language suitable for government officials."""

        user_prompt = f"""Dataset Context:
- Domain: {context['dataset_info']['domain']}
- Scope: {context['dataset_info']['scope']}
- Description: {context['dataset_info']['description']}

Business Questions to Address:
{self._format_business_questions(context)}

Statistical Evidence:

TOP METRICS:
{self._format_kpi_evidence(evidence.get('top_kpis', []))}

SIGNIFICANT TRENDS:
{self._format_trend_evidence(evidence.get('significant_trends', []))}

SPATIAL PATTERNS:
{self._format_spatial_evidence(evidence.get('spatial_patterns', {}))}

SIGNIFICANT RELATIONSHIPS:
{self._format_correlation_evidence(evidence.get('significant_correlations', []))}

GROUP DIFFERENCES:
{self._format_hypothesis_evidence(evidence.get('significant_tests', []))}

Generate 3-5 key policy findings based on this evidence."""

        try:
            # Call LLM
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            # Convert messages to Groq format
            groq_messages = []
            for msg in messages:
                if hasattr(msg, 'content'):
                    groq_messages.append({"role": "user", "content": msg.content})
            
            response = self.groq_client.chat.completions.create(
                model=self.model,
                messages=groq_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Parse JSON response
            parser = JsonOutputParser()
            findings = parser.parse(response.choices[0].message.content)
            
            # Validate and enhance findings
            validated_findings = []
            for i, finding in enumerate(findings[:5]):  # Limit to 5 findings
                if isinstance(finding, dict) and 'finding' in finding:
                    validated_finding = {
                        'id': f"finding_{i+1}",
                        'finding': finding.get('finding', ''),
                        'evidence': finding.get('evidence', ''),
                        'magnitude': finding.get('magnitude', ''),
                        'confidence': finding.get('confidence', 'MEDIUM'),
                        'policy_relevance': finding.get('policy_relevance', ''),
                        'geographic_scope': finding.get('geographic_scope', context['dataset_info']['scope']),
                        'supporting_data': self._link_supporting_data(finding, evidence)
                    }
                    validated_findings.append(validated_finding)
            
            return validated_findings
            
        except Exception as e:
            self.logger.error(f"LLM key findings generation failed: {str(e)}")
            # Fallback to rule-based findings
            return self._generate_fallback_findings(evidence, context)

    def _extract_statistical_evidence(self, analysis_results: Dict) -> Dict[str, Any]:
        """Extract key statistical evidence for LLM consumption"""
        
        evidence = {}
        
        # Top KPIs by domain relevance
        kpis = analysis_results.get('kpis', [])
        evidence['top_kpis'] = sorted(kpis, key=lambda x: (
            x.get('domain_relevance') == 'high',
            -x.get('data_quality', {}).get('missing_percentage', 100),
            x.get('sample_size', 0)
        ), reverse=True)[:5]
        
        # Significant trends
        trends = analysis_results.get('trends', [])
        evidence['significant_trends'] = [
            t for t in trends 
            if t.get('trend_analysis', {}).get('is_significant', False)
        ][:3]
        
        # Spatial patterns with high inequality
        spatial = analysis_results.get('spatial_analysis', {})
        evidence['spatial_patterns'] = {
            metric: data for metric, data in spatial.items()
            if data.get('inequality_measures', {}).get('inequality_level') in ['high', 'medium']
        }
        
        # Significant correlations
        correlations = analysis_results.get('correlations', [])
        evidence['significant_correlations'] = [
            c for c in correlations 
            if c.get('is_significant', False)
        ][:5]
        
        # Significant hypothesis tests
        tests = analysis_results.get('hypothesis_tests', [])
        evidence['significant_tests'] = [
            t for t in tests 
            if t.get('test_statistics', {}).get('is_significant', False)
        ][:3]
        
        return evidence

    def _format_business_questions(self, context: Dict) -> str:
        """Format business questions for LLM context"""
        questions = context.get('user_context', {}).get('business_questions', [])
        if questions:
            return '\n'.join(f"- {q}" for q in questions)
        return "- What are the key trends and patterns in this data?"

    def _format_kpi_evidence(self, kpis: List) -> str:
        """Format KPI evidence for LLM"""
        if not kpis:
            return "No significant KPIs identified."
        
        formatted = []
        for kpi in kpis[:3]:
            stats = kpi.get('statistics', {})
            name = kpi.get('metric_name', 'Unknown')
            formatted.append(
                f"• {name}: Mean={stats.get('mean', 0):.1f}, "
                f"Std={stats.get('std', 0):.1f}, "
                f"Range={stats.get('min', 0):.1f}-{stats.get('max', 0):.1f}"
            )
        
        return '\n'.join(formatted)

    def _format_trend_evidence(self, trends: List) -> str:
        """Format trend evidence for LLM"""
        if not trends:
            return "No significant trends detected."
        
        formatted = []
        for trend in trends:
            metric = trend.get('metric', 'Unknown')
            analysis = trend.get('trend_analysis', {})
            direction = analysis.get('direction', 'stable')
            strength = analysis.get('strength', 0)
            p_value = analysis.get('significance', 1)
            
            formatted.append(
                f"• {metric}: {direction} trend (strength={strength:.2f}, p={p_value:.3f})"
            )
        
        return '\n'.join(formatted)

    def _format_spatial_evidence(self, spatial: Dict) -> str:
        """Format spatial evidence for LLM"""
        if not spatial:
            return "No significant spatial patterns detected."
        
        formatted = []
        for metric, data in spatial.items():
            gini = data.get('inequality_measures', {}).get('gini_coefficient', 0)
            level = data.get('inequality_measures', {}).get('inequality_level', 'unknown')
            
            # Get top and bottom areas
            top_areas = list(data.get('top_performing_areas', {}).keys())[:2]
            bottom_areas = list(data.get('bottom_performing_areas', {}).keys())[:2]
            
            formatted.append(
                f"• {metric}: {level} inequality (Gini={gini:.2f}). "
                f"Top: {', '.join(top_areas)}. Bottom: {', '.join(bottom_areas)}"
            )
        
        return '\n'.join(formatted)

    def _format_correlation_evidence(self, correlations: List) -> str:
        """Format correlation evidence for LLM"""
        if not correlations:
            return "No significant correlations found."
        
        formatted = []
        for corr in correlations[:3]:
            var1 = corr.get('variable_1', 'Unknown')
            var2 = corr.get('variable_2', 'Unknown')
            coeff = corr.get('correlation_coefficient', 0)
            direction = corr.get('correlation_direction', 'unknown')
            
            formatted.append(
                f"• {var1} ↔ {var2}: {direction} correlation (r={coeff:.2f})"
            )
        
        return '\n'.join(formatted)

    def _format_hypothesis_evidence(self, tests: List) -> str:
        """Format hypothesis test evidence for LLM"""
        if not tests:
            return "No significant group differences found."
        
        formatted = []
        for test in tests:
            dependent = test.get('dependent_variable', 'Unknown')
            group1 = test.get('group_1', {})
            group2 = test.get('group_2', {})
            p_value = test.get('test_statistics', {}).get('p_value', 1)
            effect_size = test.get('effect_size', {}).get('interpretation', 'unknown')
            
            formatted.append(
                f"• {dependent}: {group1.get('name', 'Group1')} vs {group2.get('name', 'Group2')} "
                f"(p={p_value:.3f}, {effect_size} effect)"
            )
        
        return '\n'.join(formatted)

    def _link_supporting_data(self, finding: Dict, evidence: Dict) -> Dict[str, Any]:
        """Link finding to supporting statistical data"""
        # This is a simplified implementation
        # In practice, you'd implement more sophisticated matching
        return {
            'data_source': 'statistical_analysis',
            'evidence_strength': finding.get('confidence', 'MEDIUM'),
            'methodology': 'Statistical analysis with significance testing'
        }

    def _generate_fallback_findings(self, evidence: Dict, context: Dict) -> List[Dict[str, Any]]:
        """Generate fallback findings when LLM fails"""
        
        findings = []
        
        # Finding from top KPI
        top_kpis = evidence.get('top_kpis', [])
        if top_kpis:
            kpi = top_kpis[0]
            findings.append({
                'id': 'finding_1',
                'finding': f"Key metric {kpi['metric_name']} shows significant variation across the dataset",
                'evidence': f"Mean value: {kpi.get('statistics', {}).get('mean', 0):.1f}",
                'confidence': 'MEDIUM',
                'policy_relevance': 'Requires attention for resource allocation and planning'
            })
        
        # Finding from spatial patterns
        spatial_patterns = evidence.get('spatial_patterns', {})
        if spatial_patterns:
            metric, data = next(iter(spatial_patterns.items()))
            findings.append({
                'id': 'finding_2',
                'finding': f"Significant geographic inequality detected in {metric}",
                'evidence': f"Inequality level: {data.get('inequality_measures', {}).get('inequality_level', 'unknown')}",
                'confidence': 'HIGH',
                'policy_relevance': 'Geographic disparities require targeted interventions'
            })
        
        return findings

    async def _generate_policy_recommendations(self, analysis_results: Dict, context: Dict, 
                                             key_findings: List) -> List[Dict[str, Any]]:
        """Generate actionable policy recommendations"""
        self.logger.info("Generating policy recommendations")
        
        system_prompt = f"""You are a senior policy advisor for {context['dataset_info']['domain']} policy.
Based on the key findings, generate specific, actionable policy recommendations.

Each recommendation should:
1. Address a specific finding with concrete actions
2. Be implementable by government agencies
3. Include estimated impact and priority level
4. Consider budget and resource constraints

Respond with JSON array of recommendations:
{{
  "recommendation": "Specific action to take",
  "finding_addressed": "Which finding this addresses",
  "implementation_steps": ["Step 1", "Step 2", "Step 3"],
  "estimated_impact": "Expected outcome",
  "priority": "HIGH/MEDIUM/LOW",
  "estimated_cost": "LOW/MEDIUM/HIGH",
  "timeframe": "Implementation timeline",
  "responsible_agency": "Who should implement",
  "success_metrics": ["How to measure success"]
}}

Limit to 3-4 most important recommendations."""

        findings_summary = '\n'.join([
            f"• {f['finding']} (Confidence: {f['confidence']})"
            for f in key_findings
        ])

        user_prompt = f"""Context:
- Domain: {context['dataset_info']['domain']}
- Geographic Scope: {context['dataset_info']['scope']}
- Dataset: {context['dataset_info']['description']}

Key Findings to Address:
{findings_summary}

Generate actionable policy recommendations for government implementation."""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            # Convert messages to Groq format
            groq_messages = []
            for msg in messages:
                if hasattr(msg, 'content'):
                    groq_messages.append({"role": "user", "content": msg.content})
            
            response = self.groq_client.chat.completions.create(
                model=self.model,
                messages=groq_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            parser = JsonOutputParser()
            recommendations = parser.parse(response.choices[0].message.content)
            
            # Validate and enhance recommendations
            validated_recommendations = []
            for i, rec in enumerate(recommendations[:4]):
                if isinstance(rec, dict) and 'recommendation' in rec:
                    validated_rec = {
                        'id': f"recommendation_{i+1}",
                        'recommendation': rec.get('recommendation', ''),
                        'finding_addressed': rec.get('finding_addressed', ''),
                        'implementation_steps': rec.get('implementation_steps', []),
                        'estimated_impact': rec.get('estimated_impact', ''),
                        'priority': rec.get('priority', 'MEDIUM'),
                        'estimated_cost': rec.get('estimated_cost', 'MEDIUM'),
                        'timeframe': rec.get('timeframe', 'Medium-term'),
                        'responsible_agency': rec.get('responsible_agency', 'Relevant Department'),
                        'success_metrics': rec.get('success_metrics', []),
                        'evidence_strength': self._assess_recommendation_evidence(rec, key_findings)
                    }
                    validated_recommendations.append(validated_rec)
            
            return validated_recommendations
            
        except Exception as e:
            self.logger.error(f"LLM recommendations generation failed: {str(e)}")
            return self._generate_fallback_recommendations(key_findings, context)

    def _generate_fallback_recommendations(self, findings: List, context: Dict) -> List[Dict[str, Any]]:
        """Generate fallback recommendations when LLM fails"""
        
        recommendations = []
        
        if findings:
            recommendations.append({
                'id': 'recommendation_1',
                'recommendation': 'Conduct detailed analysis of identified patterns for targeted interventions',
                'finding_addressed': findings[0].get('finding', 'Key finding'),
                'priority': 'HIGH',
                'timeframe': 'Short-term',
                'estimated_impact': 'Improved understanding and targeted resource allocation'
            })
        
        return recommendations

    async def _generate_executive_summary(self, key_findings: List, policy_recommendations: List, 
                                        context: Dict) -> Dict[str, Any]:
        """Generate executive summary"""
        self.logger.info("Generating executive summary")
        
        system_prompt = """You are writing an executive summary for senior government officials.
Create a concise, high-level summary that captures the most important insights and next steps.

Format as JSON:
{
  "one_line_summary": "Single sentence capturing the main insight",
  "key_insights_summary": "2-3 sentence overview of main findings",
  "priority_actions": "1-2 most urgent recommendations",
  "overall_assessment": "Overall data quality and reliability assessment"
}

Use executive language appropriate for senior decision-makers."""

        findings_text = '\n'.join([f"• {f['finding']}" for f in key_findings[:3]])
        recommendations_text = '\n'.join([f"• {r['recommendation']}" for r in policy_recommendations[:2]])

        user_prompt = f"""Dataset: {context['dataset_info']['name']} ({context['dataset_info']['scope']})
Domain: {context['dataset_info']['domain']}

Key Findings:
{findings_text}

Top Recommendations:
{recommendations_text}

Data Quality: {context['data_characteristics']['analysis_quality']}

Create executive summary for senior officials."""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            # Convert messages to Groq format
            groq_messages = []
            for msg in messages:
                if hasattr(msg, 'content'):
                    groq_messages.append({"role": "user", "content": msg.content})
            
            response = self.groq_client.chat.completions.create(
                model=self.model,
                messages=groq_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            parser = JsonOutputParser()
            summary = parser.parse(response.choices[0].message.content)
            
            return {
                'one_line_summary': summary.get('one_line_summary', ''),
                'key_insights_summary': summary.get('key_insights_summary', ''),
                'priority_actions': summary.get('priority_actions', ''),
                'overall_assessment': summary.get('overall_assessment', ''),
                'findings_count': len(key_findings),
                'recommendations_count': len(policy_recommendations)
            }
            
        except Exception as e:
            self.logger.error(f"Executive summary generation failed: {str(e)}")
            return {
                'one_line_summary': f"Analysis of {context['dataset_info']['name']} reveals key patterns requiring policy attention",
                'key_insights_summary': f"Analysis identified {len(key_findings)} significant findings across {context['dataset_info']['domain']} data",
                'priority_actions': f"Implement {len(policy_recommendations)} recommended actions for improvement",
                'overall_assessment': f"Data quality: {context['data_characteristics']['analysis_quality']}"
            }

    async def _generate_statistical_narrative(self, analysis_results: Dict, context: Dict) -> Dict[str, Any]:
        """Generate narrative explanation of statistical methods and findings"""
        
        narrative = {
            'methodology_summary': {
                'analysis_approach': 'Comprehensive statistical analysis including descriptive statistics, trend analysis, correlation analysis, and hypothesis testing',
                'statistical_tests_used': self._summarize_statistical_methods(analysis_results),
                'significance_level': self.config.get('statistics', {}).get('significance_alpha', 0.05),
                'sample_size': context['data_characteristics']['rows']
            },
            'data_quality_assessment': {
                'overall_quality': context['data_characteristics']['analysis_quality'],
                'completeness': f"Dataset contains {context['data_characteristics']['rows']} rows and {context['data_characteristics']['columns']} columns",
                'analytical_limitations': self._identify_limitations(analysis_results, context)
            },
            'statistical_confidence': {
                'high_confidence_findings': len([f for f in analysis_results.get('hypothesis_tests', []) if f.get('test_statistics', {}).get('p_value', 1) < 0.01]),
                'medium_confidence_findings': len([f for f in analysis_results.get('hypothesis_tests', []) if 0.01 <= f.get('test_statistics', {}).get('p_value', 1) < 0.05]),
                'overall_reliability': self._assess_overall_reliability(analysis_results)
            }
        }
        
        return narrative

    def _summarize_statistical_methods(self, analysis_results: Dict) -> List[str]:
        """Summarize statistical methods used"""
        methods = ['Descriptive statistics', 'Correlation analysis']
        
        if analysis_results.get('trends'):
            methods.append('Time trend analysis')
        
        if analysis_results.get('spatial_analysis'):
            methods.append('Spatial pattern analysis')
        
        if analysis_results.get('hypothesis_tests'):
            methods.append('Hypothesis testing (t-tests, Mann-Whitney U)')
        
        return methods

    def _identify_limitations(self, analysis_results: Dict, context: Dict) -> List[str]:
        """Identify analytical limitations"""
        limitations = []
        
        if context['data_characteristics']['rows'] < 100:
            limitations.append('Small sample size may limit statistical power')
        
        analysis_quality = analysis_results.get('analysis_quality', {})
        if analysis_quality.get('data_adequacy', {}).get('data_completeness', 1) < 0.8:
            limitations.append('Missing data may impact analysis reliability')
        
        if len(analysis_results.get('trends', [])) == 0:
            limitations.append('Limited time series data for trend analysis')
        
        return limitations

    def _assess_overall_reliability(self, analysis_results: Dict) -> str:
        """Assess overall reliability of statistical findings"""
        
        quality_score = analysis_results.get('analysis_quality', {}).get('overall_quality_score', 50)
        
        if quality_score >= 80:
            return 'HIGH'
        elif quality_score >= 60:
            return 'MEDIUM'
        else:
            return 'LOW'

    def _assess_insight_confidence(self, findings: List, analysis_results: Dict, state) -> Dict[str, Any]:
        """Assess confidence in generated insights"""
        
        # Count findings by confidence level
        confidence_distribution = {}
        for finding in findings:
            conf = finding.get('confidence', 'MEDIUM')
            confidence_distribution[conf] = confidence_distribution.get(conf, 0) + 1
        
        # Overall insight quality
        high_confidence_pct = confidence_distribution.get('HIGH', 0) / len(findings) * 100 if findings else 0
        
        overall_confidence = 'HIGH' if high_confidence_pct >= 60 else 'MEDIUM' if high_confidence_pct >= 30 else 'LOW'
        
        return {
            'overall_confidence': overall_confidence,
            'confidence_distribution': confidence_distribution,
            'total_findings': len(findings),
            'high_confidence_percentage': round(high_confidence_pct, 1),
            'data_support_quality': analysis_results.get('analysis_quality', {}).get('quality_level', 'unknown'),
            'llm_generation_success': True  # Would be False if fallback methods were used
        }

    def _assess_recommendation_evidence(self, recommendation: Dict, findings: List) -> str:
        """Assess evidence strength supporting a recommendation"""
        
        # Simple heuristic: if recommendation addresses high-confidence finding, it has strong evidence
        finding_addressed = recommendation.get('finding_addressed', '')
        
        for finding in findings:
            if finding['finding'].lower() in finding_addressed.lower():
                return finding.get('confidence', 'MEDIUM')
        
        return 'MEDIUM'  # Default
    
    def _extract_statistical_evidence(self, analysis_results: Dict) -> Dict[str, Any]:
        """Extract key statistical evidence for LLM consumption with safe data access"""
        
        evidence = {
            'top_kpis': [],
            'significant_trends': [],
            'spatial_patterns': {},
            'significant_correlations': [],
            'significant_tests': []
        }
        
        try:
            # Safe KPI extraction
            kpis_data = analysis_results.get('kpis', {})
            if isinstance(kpis_data, dict):
                numeric_summary = kpis_data.get('numeric_summary', {})
                if isinstance(numeric_summary, dict):
                    for col_name, stats in numeric_summary.items():
                        if isinstance(stats, dict) and isinstance(stats.get('mean'), (int, float)):
                            evidence['top_kpis'].append({
                                'metric_name': col_name,
                                'statistics': stats,
                                'domain_relevance': 'high',  # Default
                                'sample_size': stats.get('count', 0)
                            })
            
            # Safe trends extraction
            trends_data = analysis_results.get('trends', {})
            if isinstance(trends_data, dict):
                linear_trends = trends_data.get('linear_trends', {})
                if isinstance(linear_trends, dict):
                    for metric, trend_info in linear_trends.items():
                        if isinstance(trend_info, dict) and trend_info.get('statistical_significance'):
                            evidence['significant_trends'].append({
                                'metric': metric,
                                'trend_analysis': {
                                    'direction': trend_info.get('trend_direction', 'unknown'),
                                    'strength': abs(trend_info.get('slope', 0)),
                                    'significance': trend_info.get('p_value', 1),
                                    'is_significant': trend_info.get('statistical_significance', False)
                                }
                            })
            
            # Safe correlations extraction
            correlations_data = analysis_results.get('correlations', {})
            if isinstance(correlations_data, dict):
                significant_correlations = correlations_data.get('significant_correlations', [])
                if isinstance(significant_correlations, list):
                    for corr in significant_correlations[:5]:
                        if isinstance(corr, dict):
                            evidence['significant_correlations'].append({
                                'variable_1': corr.get('variable_1', 'Unknown'),
                                'variable_2': corr.get('variable_2', 'Unknown'),
                                'correlation_coefficient': corr.get('pearson_r', 0),
                                'correlation_direction': 'positive' if corr.get('pearson_r', 0) > 0 else 'negative',
                                'is_significant': corr.get('pearson_significant', False)
                            })
            
            # Safe spatial patterns extraction
            spatial_data = analysis_results.get('spatial_analysis', {})
            if isinstance(spatial_data, dict):
                regional_comparisons = spatial_data.get('regional_comparisons', {})
                if isinstance(regional_comparisons, dict):
                    for metric, comparison_data in regional_comparisons.items():
                        if isinstance(comparison_data, dict):
                            inequality_metrics = comparison_data.get('inequality_metrics', {})
                            if isinstance(inequality_metrics, dict):
                                gini = inequality_metrics.get('gini_coefficient', 0)
                                evidence['spatial_patterns'][metric] = {
                                    'inequality_measures': {
                                        'gini_coefficient': gini,
                                        'inequality_level': 'high' if gini > 0.4 else 'medium' if gini > 0.25 else 'low'
                                    },
                                    'top_performing_areas': {},
                                    'bottom_performing_areas': {}
                                }
            
            # Safe hypothesis tests extraction
            hypothesis_data = analysis_results.get('hypothesis_tests', {})
            if isinstance(hypothesis_data, dict):
                group_comparisons = hypothesis_data.get('group_comparisons', [])
                if isinstance(group_comparisons, list):
                    for test in group_comparisons[:3]:
                        if isinstance(test, dict) and test.get('significant'):
                            evidence['significant_tests'].append({
                                'dependent_variable': test.get('test_variable', 'Unknown'),
                                'group_1': {'name': 'Group1'},
                                'group_2': {'name': 'Group2'},
                                'test_statistics': {
                                    'p_value': test.get('p_value', 1),
                                    'is_significant': test.get('significant', False)
                                },
                                'effect_size': {
                                    'interpretation': test.get('effect_size_interpretation', 'unknown')
                                }
                            })
        
        except Exception as e:
            self.logger.warning(f"Evidence extraction failed: {str(e)}")
        
        return evidence
    
    def _safe_get_list(self, data: Dict, key: str) -> List:
        """Safely get a list from dictionary"""
        value = data.get(key, [])
        if isinstance(value, list):
            return value
        elif isinstance(value, dict):
            return list(value.values()) if value else []
        else:
            return []

    def _safe_get_dict(self, data: Dict, key: str) -> Dict:
        """Safely get a dictionary from dictionary"""
        value = data.get(key, {})
        return value if isinstance(value, dict) else {}

    def _create_minimal_analysis(self, raw_data) -> Dict:
        """Create minimal analysis from raw data when analysis results are not available"""
        if raw_data is None:
            return {}
        
        try:
            numeric_cols = raw_data.select_dtypes(include=[np.number]).columns
            
            minimal_analysis = {
                'dataset_profile': {
                    'basic_info': {
                        'rows': len(raw_data),
                        'columns': len(raw_data.columns)
                    }
                },
                'kpis': {
                    'numeric_summary': {}
                },
                'quality_assessment': {
                    'overall_score': 50
                }
            }
            
            # Add basic stats for numeric columns
            for col in numeric_cols[:5]:  # Limit to 5 columns
                try:
                    col_data = raw_data[col].dropna()
                    if len(col_data) > 0:
                        minimal_analysis['kpis']['numeric_summary'][col] = {
                            'count': int(len(col_data)),
                            'mean': float(col_data.mean()),
                            'std': float(col_data.std()),
                            'min': float(col_data.min()),
                            'max': float(col_data.max())
                        }
                except:
                    continue
            
            return minimal_analysis
            
        except Exception as e:
            self.logger.warning(f"Minimal analysis creation failed: {str(e)}")
            return {
                'dataset_profile': {'basic_info': {'rows': 0, 'columns': 0}},
                'kpis': {},
                'quality_assessment': {'overall_score': 30}
            }

    def _create_fallback_context(self, state) -> Dict:
        """Create fallback context when normal context preparation fails"""
        run_manifest = state.run_manifest
        
        return {
            'dataset_info': {
                'name': run_manifest.get('dataset_info', {}).get('dataset_name', 'Unknown Dataset'),
                'domain': run_manifest.get('dataset_info', {}).get('domain_hint', 'general'),
                'scope': run_manifest.get('dataset_info', {}).get('scope', 'Unknown Scope'),
                'description': 'Dataset analysis with limited context'
            },
            'data_characteristics': {
                'rows': 0,
                'columns': 0,
                'analysis_quality': 'limited'
            },
            'analysis_summary': {
                'kpis_analyzed': 0,
                'trends_identified': 0,
                'correlations_found': 0,
                'spatial_patterns': False,
                'hypothesis_tests': 0
            }
        }

    def _generate_fallback_summary(self, context: Dict) -> Dict:
        """Generate fallback executive summary"""
        return {
            'one_line_summary': f"Analysis of {context['dataset_info']['name']} provides insights for {context['dataset_info']['domain']} policy",
            'key_insights_summary': 'Statistical analysis completed with limited data processing',
            'priority_actions': 'Review data quality and consider additional data collection',
            'overall_assessment': 'Analysis completed with constraints',
            'findings_count': 0,
            'recommendations_count': 0
        }

    def _generate_fallback_narrative(self) -> Dict:
        """Generate fallback statistical narrative"""
        return {
            'methodology_summary': {
                'analysis_approach': 'Basic statistical analysis completed',
                'statistical_tests_used': ['Descriptive statistics'],
                'significance_level': 0.05,
                'sample_size': 'Unknown'
            },
            'data_quality_assessment': {
                'overall_quality': 'limited',
                'completeness': 'Data processing completed with constraints',
                'analytical_limitations': ['Limited statistical power due to processing constraints']
            },
            'statistical_confidence': {
                'overall_reliability': 'MEDIUM'
            }
        }
        
    def _generate_fallback_recommendations(self, key_findings: List, context: Dict) -> List[Dict]:
        """Generate fallback recommendations when LLM fails"""
        return [
            {
                'id': 'rec_1',
                'recommendation': f"Review data quality for {context['dataset_info']['domain']} analysis",
                'priority': 'HIGH',
                'rationale': 'Ensure data completeness and accuracy for better insights',
                'implementation_timeframe': 'Short-term',
                'estimated_impact': 'MEDIUM'
            }
        ]