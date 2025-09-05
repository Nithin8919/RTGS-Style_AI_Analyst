"""
RTGS AI Analyst - Analysis Agent
Performs statistical analysis, KPI calculations, and hypothesis testing
"""

import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr, normaltest, levene
import warnings

from src.utils.logging import get_agent_logger, TransformLogger

warnings.filterwarnings('ignore')


class AnalysisAgent:
    """Agent responsible for statistical analysis and KPI generation"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.logger = get_agent_logger("analysis")
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Extract statistical parameters
        self.stats_config = self.config.get('statistics', {})
        self.significance_alpha = self.stats_config.get('significance_alpha', 0.05)
        self.correlation_threshold = self.stats_config.get('correlation_threshold', 0.6)
        self.min_sample_size = self.stats_config.get('min_sample_size', 30)
        
    async def process(self, state) -> Any:
        """Main analysis processing pipeline"""
        self.logger.info("Starting statistical analysis process")
        
        try:
            # Get transformed data
            transformed_data = getattr(state, 'transformed_data', state.cleaned_data)
            if transformed_data is None:
                raise ValueError("No transformed data available for analysis")
            
            # Initialize analysis results container
            analysis_results = {
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'dataset_info': {
                    'rows': len(transformed_data),
                    'columns': len(transformed_data.columns),
                    'domain': state.run_manifest['dataset_info']['domain_hint']
                }
            }
            
            # Perform different types of analysis
            kpi_results = await self._calculate_kpis(transformed_data, state)
            analysis_results['kpis'] = kpi_results
            
            trend_results = await self._analyze_trends(transformed_data, state)
            analysis_results['trends'] = trend_results
            
            spatial_results = await self._analyze_spatial_patterns(transformed_data, state)
            analysis_results['spatial_analysis'] = spatial_results
            
            correlation_results = await self._analyze_correlations(transformed_data)
            analysis_results['correlations'] = correlation_results
            
            hypothesis_results = await self._run_hypothesis_tests(transformed_data, state)
            analysis_results['hypothesis_tests'] = hypothesis_results
            
            quality_assessment = await self._assess_analysis_quality(analysis_results, transformed_data)
            analysis_results['analysis_quality'] = quality_assessment
            
            # Generate summary statistics
            summary_stats = await self._generate_summary_statistics(transformed_data)
            analysis_results['summary_statistics'] = summary_stats
            
            # Save analysis results
            results_path = Path(state.run_manifest['artifacts_paths']['docs_dir']) / "analysis_results.json"
            with open(results_path, 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            
            # Update state
            state.analysis_results = analysis_results
            state.rows_processed = len(transformed_data)
            
            self.logger.info(f"Analysis completed: {len(kpi_results)} KPIs, {len(correlation_results)} correlations")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Statistical analysis failed: {str(e)}")
            state.errors.append(f"Statistical analysis failed: {str(e)}")
            return state

    async def _calculate_kpis(self, df: pd.DataFrame, state) -> List[Dict[str, Any]]:
        """Calculate key performance indicators for numeric columns"""
        self.logger.info("Calculating KPIs")
        
        kpis = []
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        # Domain-specific KPI priorities
        domain = state.run_manifest['dataset_info']['domain_hint']
        domain_metrics = self.config.get('domains', {}).get(domain, {}).get('key_metrics', [])
        
        for col in numeric_columns:
            if df[col].isnull().all():
                continue
                
            # Calculate basic statistics
            col_data = df[col].dropna()
            
            kpi = {
                'metric_name': col,
                'metric_type': 'numeric',
                'domain_relevance': 'high' if any(keyword in col.lower() for keyword in domain_metrics) else 'medium',
                'sample_size': len(col_data),
                'statistics': {
                    'count': int(len(col_data)),
                    'mean': float(col_data.mean()),
                    'median': float(col_data.median()),
                    'std': float(col_data.std()),
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'q25': float(col_data.quantile(0.25)),
                    'q75': float(col_data.quantile(0.75)),
                    'skewness': float(col_data.skew()),
                    'kurtosis': float(col_data.kurtosis())
                },
                'data_quality': {
                    'missing_values': int(df[col].isnull().sum()),
                    'missing_percentage': float(df[col].isnull().sum() / len(df) * 100),
                    'outliers_iqr': self._count_outliers_iqr(col_data),
                    'zero_values': int((col_data == 0).sum())
                }
            }
            
            # Add percentile information for better context
            kpi['percentiles'] = {
                f'p{p}': float(col_data.quantile(p/100))
                for p in [5, 10, 25, 50, 75, 90, 95]
            }
            
            # Add domain-specific insights
            if domain == 'transport' and any(keyword in col.lower() for keyword in ['registration', 'vehicle', 'license']):
                kpi['domain_insights'] = self._get_transport_insights(col_data, col)
            elif domain == 'health' and any(keyword in col.lower() for keyword in ['patient', 'case', 'treatment']):
                kpi['domain_insights'] = self._get_health_insights(col_data, col)
            elif domain == 'education' and any(keyword in col.lower() for keyword in ['student', 'enrollment', 'teacher']):
                kpi['domain_insights'] = self._get_education_insights(col_data, col)
            
            kpis.append(kpi)
        
        # Sort KPIs by domain relevance and data quality
        kpis.sort(key=lambda x: (
            x['domain_relevance'] == 'high',
            -x['data_quality']['missing_percentage'],
            -x['sample_size']
        ), reverse=True)
        
        return kpis

    def _count_outliers_iqr(self, data: pd.Series) -> int:
        """Count outliers using IQR method"""
        try:
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((data < lower_bound) | (data > upper_bound)).sum()
            return int(outliers)
        except:
            return 0

    def _get_transport_insights(self, data: pd.Series, column: str) -> Dict[str, Any]:
        """Generate transport domain-specific insights"""
        insights = {
            'domain': 'transport',
            'metric_category': 'vehicle_statistics'
        }
        
        if 'registration' in column.lower():
            insights['interpretation'] = 'Vehicle registration trends indicate transport infrastructure usage'
            insights['policy_relevance'] = 'Monitor for capacity planning and infrastructure development'
        elif 'license' in column.lower():
            insights['interpretation'] = 'License statistics reflect driving population and compliance'
            insights['policy_relevance'] = 'Useful for road safety and enforcement planning'
        
        # Add growth context if possible
        if data.std() > 0:
            cv = data.std() / data.mean()
            insights['variability'] = 'high' if cv > 0.5 else 'medium' if cv > 0.2 else 'low'
        
        return insights

    def _get_health_insights(self, data: pd.Series, column: str) -> Dict[str, Any]:
        """Generate health domain-specific insights"""
        insights = {
            'domain': 'health',
            'metric_category': 'health_statistics'
        }
        
        if 'patient' in column.lower() or 'case' in column.lower():
            insights['interpretation'] = 'Patient/case statistics indicate healthcare demand and utilization'
            insights['policy_relevance'] = 'Critical for healthcare resource allocation and planning'
        elif 'treatment' in column.lower():
            insights['interpretation'] = 'Treatment metrics reflect healthcare service delivery'
            insights['policy_relevance'] = 'Monitor for quality of care and accessibility'
        
        return insights

    def _get_education_insights(self, data: pd.Series, column: str) -> Dict[str, Any]:
        """Generate education domain-specific insights"""
        insights = {
            'domain': 'education',
            'metric_category': 'education_statistics'
        }
        
        if 'student' in column.lower() or 'enrollment' in column.lower():
            insights['interpretation'] = 'Student/enrollment data reflects education system capacity and demand'
            insights['policy_relevance'] = 'Essential for education infrastructure and resource planning'
        elif 'teacher' in column.lower():
            insights['interpretation'] = 'Teacher statistics indicate education system capacity'
            insights['policy_relevance'] = 'Critical for maintaining education quality and student-teacher ratios'
        
        return insights

    async def _analyze_trends(self, df: pd.DataFrame, state) -> List[Dict[str, Any]]:
        """Analyze time trends in the data"""
        self.logger.info("Analyzing time trends")
        
        trends = []
        
        # Identify time columns
        time_columns = [col for col in df.columns if any(time_word in col.lower() for time_word in ['date', 'time', 'year', 'month', 'quarter'])]
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if not time_columns or len(numeric_columns) == 0:
            self.logger.info("No time columns or numeric data found for trend analysis")
            return trends
        
        # Use the first suitable time column
        time_col = time_columns[0]
        
        try:
            # Convert to datetime if needed
            if df[time_col].dtype != 'datetime64[ns]':
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            
            # Sort by time
            df_sorted = df.sort_values(time_col).dropna(subset=[time_col])
            
            # Analyze trends for top numeric columns
            for col in numeric_columns[:5]:  # Limit to top 5 metrics
                if col == time_col:
                    continue
                    
                # Create time series
                time_series = df_sorted.groupby(time_col)[col].mean().dropna()
                
                if len(time_series) < 3:  # Need at least 3 points for trend
                    continue
                
                # Calculate trend statistics
                x_numeric = range(len(time_series))
                y_values = time_series.values
                
                # Linear regression for trend
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, y_values)
                
                # Seasonality detection (basic autocorrelation)
                seasonality_score = self._detect_seasonality(time_series)
                
                trend_info = {
                    'metric': col,
                    'time_column': time_col,
                    'data_points': len(time_series),
                    'time_range': {
                        'start': str(time_series.index.min()),
                        'end': str(time_series.index.max())
                    },
                    'trend_analysis': {
                        'slope': float(slope),
                        'direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable',
                        'strength': abs(r_value),
                        'significance': p_value,
                        'r_squared': r_value ** 2,
                        'is_significant': p_value < self.significance_alpha
                    },
                    'seasonality': {
                        'score': seasonality_score,
                        'has_seasonality': seasonality_score > 0.3
                    },
                    'summary_statistics': {
                        'mean': float(time_series.mean()),
                        'std': float(time_series.std()),
                        'min': float(time_series.min()),
                        'max': float(time_series.max()),
                        'volatility': float(time_series.std() / time_series.mean()) if time_series.mean() != 0 else 0
                    }
                }
                
                trends.append(trend_info)
                
        except Exception as e:
            self.logger.warning(f"Trend analysis failed: {str(e)}")
        
        # Sort trends by significance and strength
        trends.sort(key=lambda x: (x['trend_analysis']['is_significant'], x['trend_analysis']['strength']), reverse=True)
        
        return trends

    def _detect_seasonality(self, time_series: pd.Series) -> float:
        """Detect seasonality using autocorrelation"""
        try:
            if len(time_series) < 12:  # Need sufficient data
                return 0.0
            
            # Calculate autocorrelation at different lags
            autocorr_scores = []
            for lag in [3, 6, 12]:  # Quarterly, semi-annual, annual patterns
                if lag < len(time_series):
                    autocorr = time_series.autocorr(lag=lag)
                    if not np.isnan(autocorr):
                        autocorr_scores.append(abs(autocorr))
            
            return max(autocorr_scores) if autocorr_scores else 0.0
            
        except:
            return 0.0

    async def _analyze_spatial_patterns(self, df: pd.DataFrame, state) -> Dict[str, Any]:
        """Analyze spatial patterns and geographic inequalities"""
        self.logger.info("Analyzing spatial patterns")
        
        spatial_analysis = {}
        
        # Identify geographic columns
        domain = state.run_manifest['dataset_info']['domain_hint']
        domain_geo_cols = self.config.get('domains', {}).get(domain, {}).get('geo_columns', [])
        
        geo_columns = [col for col in df.columns if any(geo_word in col.lower() for geo_word in ['district', 'mandal', 'village', 'zone', 'area', 'region'] + domain_geo_cols)]
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if not geo_columns or len(numeric_columns) == 0:
            self.logger.info("No geographic columns found for spatial analysis")
            return spatial_analysis
        
        geo_col = geo_columns[0]  # Use first geographic column
        
        try:
            # Analyze spatial distribution for top metrics
            for col in numeric_columns[:3]:  # Limit to top 3 metrics
                
                # Calculate spatial statistics
                spatial_stats = df.groupby(geo_col)[col].agg(['mean', 'sum', 'count', 'std']).fillna(0)
                spatial_stats = spatial_stats[spatial_stats['count'] > 0]  # Remove empty groups
                
                if len(spatial_stats) < 2:
                    continue
                
                # Calculate inequality measures
                values = spatial_stats['mean'].values
                gini_coefficient = self._calculate_gini_coefficient(values)
                cv = spatial_stats['mean'].std() / spatial_stats['mean'].mean() if spatial_stats['mean'].mean() > 0 else 0
                
                # Identify top and bottom performing areas
                top_areas = spatial_stats.nlargest(3, 'mean')[['mean', 'sum', 'count']].to_dict('index')
                bottom_areas = spatial_stats.nsmallest(3, 'mean')[['mean', 'sum', 'count']].to_dict('index')
                
                spatial_analysis[col] = {
                    'geographic_column': geo_col,
                    'total_areas': len(spatial_stats),
                    'inequality_measures': {
                        'gini_coefficient': float(gini_coefficient),
                        'coefficient_of_variation': float(cv),
                        'inequality_level': 'high' if gini_coefficient > 0.4 else 'medium' if gini_coefficient > 0.25 else 'low'
                    },
                    'spatial_distribution': {
                        'mean_across_areas': float(spatial_stats['mean'].mean()),
                        'std_across_areas': float(spatial_stats['mean'].std()),
                        'min_area_value': float(spatial_stats['mean'].min()),
                        'max_area_value': float(spatial_stats['mean'].max()),
                        'range_ratio': float(spatial_stats['mean'].max() / spatial_stats['mean'].min()) if spatial_stats['mean'].min() > 0 else float('inf')
                    },
                    'top_performing_areas': {area: {k: float(v) for k, v in stats.items()} for area, stats in top_areas.items()},
                    'bottom_performing_areas': {area: {k: float(v) for k, v in stats.items()} for area, stats in bottom_areas.items()}
                }
                
        except Exception as e:
            self.logger.warning(f"Spatial analysis failed: {str(e)}")
        
        return spatial_analysis

    def _calculate_gini_coefficient(self, values: np.ndarray) -> float:
        """Calculate Gini coefficient for inequality measurement"""
        try:
            # Remove negative values and sort
            values = np.array(values)
            values = values[values >= 0]
            values = np.sort(values)
            
            n = len(values)
            if n == 0 or values.sum() == 0:
                return 0.0
            
            # Calculate Gini coefficient
            cumsum = np.cumsum(values)
            gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
            return max(0.0, min(1.0, gini))
            
        except:
            return 0.0

    async def _analyze_correlations(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze correlations between numeric variables"""
        self.logger.info("Analyzing correlations")
        
        correlations = []
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) < 2:
            return correlations
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_columns].corr(method='spearman')
        
        # Extract significant correlations
        for i, col1 in enumerate(numeric_columns):
            for j, col2 in enumerate(numeric_columns):
                if i >= j:  # Avoid duplicates and self-correlation
                    continue
                
                correlation_value = corr_matrix.loc[col1, col2]
                
                if abs(correlation_value) >= self.correlation_threshold:
                    
                    # Calculate additional statistics
                    data1 = df[col1].dropna()
                    data2 = df[col2].dropna()
                    
                    # Get common indices for paired analysis
                    common_idx = data1.index.intersection(data2.index)
                    if len(common_idx) < self.min_sample_size:
                        continue
                    
                    paired_data1 = data1.loc[common_idx]
                    paired_data2 = data2.loc[common_idx]
                    
                    # Calculate Pearson correlation as well
                    pearson_corr, pearson_p = pearsonr(paired_data1, paired_data2)
                    
                    correlation_info = {
                        'variable_1': col1,
                        'variable_2': col2,
                        'correlation_coefficient': float(correlation_value),
                        'correlation_strength': self._interpret_correlation_strength(abs(correlation_value)),
                        'correlation_direction': 'positive' if correlation_value > 0 else 'negative',
                        'sample_size': len(common_idx),
                        'pearson_correlation': float(pearson_corr),
                        'pearson_p_value': float(pearson_p),
                        'is_significant': pearson_p < self.significance_alpha,
                        'interpretation': self._interpret_correlation(col1, col2, correlation_value)
                    }
                    
                    correlations.append(correlation_info)
        
        # Sort by absolute correlation strength
        correlations.sort(key=lambda x: abs(x['correlation_coefficient']), reverse=True)
        
        return correlations

    def _interpret_correlation_strength(self, abs_corr: float) -> str:
        """Interpret correlation strength"""
        if abs_corr >= 0.8:
            return 'very_strong'
        elif abs_corr >= 0.6:
            return 'strong'
        elif abs_corr >= 0.4:
            return 'moderate'
        elif abs_corr >= 0.2:
            return 'weak'
        else:
            return 'very_weak'

    def _interpret_correlation(self, var1: str, var2: str, correlation: float) -> str:
        """Generate interpretation of correlation"""
        direction = "positively" if correlation > 0 else "negatively"
        strength = self._interpret_correlation_strength(abs(correlation))
        
        return f"{var1} and {var2} are {strength} {direction} correlated (r={correlation:.3f})"

    async def _run_hypothesis_tests(self, df: pd.DataFrame, state) -> List[Dict[str, Any]]:
        """Run hypothesis tests for group comparisons"""
        self.logger.info("Running hypothesis tests")
        
        hypothesis_tests = []
        
        # Find categorical variables for grouping
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Limit to prevent too many tests
        categorical_cols = categorical_cols[:2]
        numeric_cols = numeric_cols[:3]
        
        for cat_col in categorical_cols:
            # Only test groups with reasonable sample sizes
            group_sizes = df[cat_col].value_counts()
            valid_groups = group_sizes[group_sizes >= self.min_sample_size].index
            
            if len(valid_groups) < 2:
                continue
            
            # Take top 2 groups for comparison
            groups_to_compare = valid_groups[:2]
            
            for num_col in numeric_cols:
                try:
                    # Extract data for each group
                    group1_data = df[df[cat_col] == groups_to_compare[0]][num_col].dropna()
                    group2_data = df[df[cat_col] == groups_to_compare[1]][num_col].dropna()
                    
                    if len(group1_data) < self.min_sample_size or len(group2_data) < self.min_sample_size:
                        continue
                    
                    # Test for normality
                    _, p_norm1 = normaltest(group1_data)
                    _, p_norm2 = normaltest(group2_data)
                    both_normal = p_norm1 > 0.05 and p_norm2 > 0.05
                    
                    # Test for equal variances
                    _, p_levene = levene(group1_data, group2_data)
                    equal_variances = p_levene > 0.05
                    
                    # Choose appropriate test
                    if both_normal and equal_variances:
                        # Independent t-test
                        statistic, p_value = stats.ttest_ind(group1_data, group2_data)
                        test_type = "independent_t_test"
                        assumptions = "normal_distribution_equal_variances"
                    elif both_normal and not equal_variances:
                        # Welch's t-test
                        statistic, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=False)
                        test_type = "welch_t_test"
                        assumptions = "normal_distribution_unequal_variances"
                    else:
                        # Mann-Whitney U test (non-parametric)
                        statistic, p_value = stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')
                        test_type = "mann_whitney_u"
                        assumptions = "no_normality_assumed"
                    
                    # Calculate effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(group1_data) - 1) * group1_data.var() + 
                                        (len(group2_data) - 1) * group2_data.var()) / 
                                       (len(group1_data) + len(group2_data) - 2))
                    cohens_d = (group1_data.mean() - group2_data.mean()) / pooled_std if pooled_std > 0 else 0
                    
                    effect_size_interpretation = self._interpret_effect_size(abs(cohens_d))
                    
                    test_result = {
                        'test_type': test_type,
                        'dependent_variable': num_col,
                        'grouping_variable': cat_col,
                        'group_1': {
                            'name': str(groups_to_compare[0]),
                            'size': len(group1_data),
                            'mean': float(group1_data.mean()),
                            'std': float(group1_data.std())
                        },
                        'group_2': {
                            'name': str(groups_to_compare[1]),
                            'size': len(group2_data),
                            'mean': float(group2_data.mean()),
                            'std': float(group2_data.std())
                        },
                        'test_statistics': {
                            'statistic': float(statistic),
                            'p_value': float(p_value),
                            'is_significant': p_value < self.significance_alpha,
                            'alpha_level': self.significance_alpha
                        },
                        'effect_size': {
                            'cohens_d': float(cohens_d),
                            'interpretation': effect_size_interpretation,
                            'magnitude': abs(cohens_d)
                        },
                        'assumptions': assumptions,
                        'conclusion': self._generate_test_conclusion(groups_to_compare, num_col, p_value, cohens_d)
                    }
                    
                    hypothesis_tests.append(test_result)
                    
                except Exception as e:
                    self.logger.warning(f"Hypothesis test failed for {num_col} by {cat_col}: {str(e)}")
        
        # Sort by significance and effect size
        hypothesis_tests.sort(key=lambda x: (x['test_statistics']['is_significant'], x['effect_size']['magnitude']), reverse=True)
        
        return hypothesis_tests

    def _interpret_effect_size(self, abs_cohens_d: float) -> str:
        """Interpret Cohen's d effect size"""
        thresholds = self.stats_config.get('effect_size_thresholds', {'small': 0.2, 'medium': 0.5, 'large': 0.8})
        
        if abs_cohens_d >= thresholds['large']:
            return 'large'
        elif abs_cohens_d >= thresholds['medium']:
            return 'medium'
        elif abs_cohens_d >= thresholds['small']:
            return 'small'
        else:
            return 'negligible'

    def _generate_test_conclusion(self, groups: List, variable: str, p_value: float, cohens_d: float) -> str:
        """Generate human-readable conclusion for hypothesis test"""
        
        significance = "significant" if p_value < self.significance_alpha else "not significant"
        direction = "higher" if cohens_d > 0 else "lower"
        effect_magnitude = self._interpret_effect_size(abs(cohens_d))
        
        if p_value < self.significance_alpha:
            return f"There is a statistically {significance} difference in {variable} between {groups[0]} and {groups[1]}. {groups[0]} has {direction} values with a {effect_magnitude} effect size (p={p_value:.3f}, d={cohens_d:.3f})."
        else:
            return f"No statistically significant difference found in {variable} between {groups[0]} and {groups[1]} (p={p_value:.3f})."

    async def _assess_analysis_quality(self, analysis_results: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess the quality and reliability of the analysis"""
        
        quality_assessment = {
            'data_adequacy': {
                'sample_size': len(df),
                'sample_size_adequate': len(df) >= self.min_sample_size,
                'missing_data_impact': df.isnull().sum().sum() / (len(df) * len(df.columns)),
                'data_completeness': 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
            },
            'analysis_coverage': {
                'kpis_calculated': len(analysis_results.get('kpis', [])),
                'trends_analyzed': len(analysis_results.get('trends', [])),
                'correlations_found': len(analysis_results.get('correlations', [])),
                'hypothesis_tests_run': len(analysis_results.get('hypothesis_tests', []))
            },
            'statistical_rigor': {
                'significance_level_used': self.significance_alpha,
                'minimum_sample_size': self.min_sample_size,
                'effect_sizes_calculated': len([test for test in analysis_results.get('hypothesis_tests', []) if 'effect_size' in test])
            }
        }
        
        # Calculate overall analysis quality score
        adequacy_score = 100 if quality_assessment['data_adequacy']['sample_size_adequate'] else 50
        completeness_score = quality_assessment['data_adequacy']['data_completeness'] * 100
        coverage_score = min(100, (quality_assessment['analysis_coverage']['kpis_calculated'] * 10))
        
        overall_score = (adequacy_score * 0.4 + completeness_score * 0.4 + coverage_score * 0.2)
        
        quality_assessment['overall_quality_score'] = round(overall_score, 1)
        quality_assessment['quality_level'] = 'high' if overall_score >= 80 else 'medium' if overall_score >= 60 else 'low'
        
        return quality_assessment

    async def _generate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate overall summary statistics for the dataset"""
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        
        summary = {
            'dataset_overview': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'numeric_columns': len(numeric_columns),
                'categorical_columns': len(categorical_columns),
                'missing_values': int(df.isnull().sum().sum()),
                'missing_percentage': float(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
            },
            'numeric_summary': {},
            'categorical_summary': {}
        }
        
        # Numeric summary
        if len(numeric_columns) > 0:
            numeric_data = df[numeric_columns]
            summary['numeric_summary'] = {
                'total_numeric_columns': len(numeric_columns),
                'mean_values': {col: float(numeric_data[col].mean()) for col in numeric_columns if not numeric_data[col].isnull().all()},
                'median_values': {col: float(numeric_data[col].median()) for col in numeric_columns if not numeric_data[col].isnull().all()},
                'std_values': {col: float(numeric_data[col].std()) for col in numeric_columns if not numeric_data[col].isnull().all()},
                'overall_statistics': {
                    'mean_of_means': float(numeric_data.mean().mean()),
                    'mean_of_medians': float(numeric_data.median().mean()),
                    'average_std': float(numeric_data.std().mean())
                }
            }
        
        # Categorical summary
        if len(categorical_columns) > 0:
            summary['categorical_summary'] = {
                'total_categorical_columns': len(categorical_columns),
                'unique_value_counts': {col: int(df[col].nunique()) for col in categorical_columns},
                'most_common_values': {
                    col: df[col].value_counts().head(3).to_dict() 
                    for col in categorical_columns if not df[col].isnull().all()
                },
                'average_unique_values': float(df[categorical_columns].nunique().mean())
            }
        
        return summary