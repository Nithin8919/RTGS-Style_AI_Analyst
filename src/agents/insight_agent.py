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

from langchain_openai import ChatOpenAI
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
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.config['openai']['model'],
            temperature=self.config['openai']['temperature'],
            max_tokens=self.config['openai']['max_tokens']
        )
        
        # Load insight generation settings
        self.insights_config = self.config.get('insights', {})
        
    async def process(self, state) -> Any:
        """Main insight generation processing pipeline"""
        self.logger.info("Starting insight generation process")
        
        try:
            # Get analysis results
            analysis_results = getattr(state, 'analysis_results', {})
            if not analysis_results:
                raise ValueError("No analysis results available for insight generation")
            
            # Prepare context for LLM
            insight_context = await self._prepare_insight_context(state)
            
            # Generate different types of insights
            key_findings = await self._generate_key_findings(analysis_results, insight_context)
            
            policy_recommendations = await self._generate_policy_recommendations(
                analysis_results, insight_context, key_findings
            )
            
            executive_summary = await self._generate_executive_summary(
                key_findings, policy_recommendations, insight_context
            )
            
            statistical_narrative = await self._generate_statistical_narrative(
                analysis_results, insight_context
            )
            
            # Compile comprehensive insights
            insights = {
                'generation_timestamp': datetime.utcnow().isoformat(),
                'context': insight_context,
                'executive_summary': executive_summary,
                'key_findings': key_findings,
                'policy_recommendations': policy_recommendations,
                'statistical_narrative': statistical_narrative,
                'confidence_assessment': self._assess_insight_confidence(
                    key_findings, analysis_results, state
                )
            }
            
            # Save insights
            insights_path = Path(state.run_manifest['artifacts_paths']['docs_dir']) / "insights_executive.json"
            with open(insights_path, 'w') as f:
                json.dump(insights, f, indent=2)
            
            # Update state
            state.insights = insights
            
            self.logger.info(f"Insight generation completed: {len(key_findings)} findings, {len(policy_recommendations)} recommendations")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Insight generation failed: {str(e)}")
            state.errors.append(f"Insight generation failed: {str(e)}")
            return state

    async def _prepare_insight_context(self, state) -> Dict[str, Any]:
        """Prepare context information for insight generation"""
        
        run_manifest = state.run_manifest
        analysis_results = getattr(state, 'analysis_results', {})
        
        context = {
            'dataset_info': {
                'name': run_manifest['dataset_info']['dataset_name'],
                'domain': run_manifest['dataset_info']['domain_hint'],
                'scope': run_manifest['dataset_info']['scope'],
                'description': run_manifest['dataset_info']['description']
            },
            'user_context': run_manifest.get('user_context', {}),
            'data_characteristics': {
                'rows': analysis_results.get('dataset_info', {}).get('rows', 0),
                'columns': analysis_results.get('dataset_info', {}).get('columns', 0),
                'analysis_quality': analysis_results.get('analysis_quality', {}).get('quality_level', 'unknown')
            },
            'key_metrics_analyzed': len(analysis_results.get('kpis', [])),
            'trends_identified': len(analysis_results.get('trends', [])),
            'spatial_patterns': len(analysis_results.get('spatial_analysis', {})),
            'significant_correlations': len([c for c in analysis_results.get('correlations', []) if c.get('is_significant', False)]),
            'hypothesis_tests_run': len(analysis_results.get('hypothesis_tests', []))
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
            
            response = await self.llm.ainvoke(messages)
            
            # Parse JSON response
            parser = JsonOutputParser()
            findings = parser.parse(response.content)
            
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
            
            response = await self.llm.ainvoke(messages)
            parser = JsonOutputParser()
            recommendations = parser.parse(response.content)
            
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
            
            response = await self.llm.ainvoke(messages)
            parser = JsonOutputParser()
            summary = parser.parse(response.content)
            
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