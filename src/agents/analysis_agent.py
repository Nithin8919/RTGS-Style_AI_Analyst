"""
RTGS AI Analyst - Analysis Agent (Complete Implementation)
Performs statistical analysis, KPI calculation, trend analysis, and hypothesis testing
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr, ttest_ind, mannwhitneyu, chi2_contingency, normaltest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mutual_info_score
from typing import Dict, Any, List, Tuple, Optional
import json
import yaml
from pathlib import Path
from datetime import datetime
import warnings

from src.utils.logging import get_agent_logger

warnings.filterwarnings('ignore')

class AnalysisAgent:
    """Agent responsible for statistical analysis and KPI computation"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.logger = get_agent_logger("analysis")
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Analysis configuration
        self.analysis_config = self.config.get('analysis', {})
        self.significance_level = self.config.get('data_quality', {}).get('SIGNIFICANCE_ALPHA', 0.05)
        
    async def process(self, state) -> Any:
        """Main analysis processing pipeline with robust error handling"""
        self.logger.info("Starting statistical analysis")
        
        try:
            # Get best available data with fallback
            input_data = None
            data_sources = ['transformed_data', 'cleaned_data', 'standardized_data', 'raw_data']
            
            for source in data_sources:
                if hasattr(state, source) and getattr(state, source) is not None:
                    input_data = getattr(state, source)
                    self.logger.info(f"Using {source} for analysis")
                    break
            
            if input_data is None or len(input_data) == 0:
                raise ValueError("No data available for analysis")
            
            # Initialize error tracking
            if not hasattr(state, 'errors'):
                state.errors = []
            if not hasattr(state, 'warnings'):
                state.warnings = []
            
            self.logger.info(f"Analyzing dataset: {len(input_data)} rows × {len(input_data.columns)} columns")
            
            # Clean data types to prevent analysis errors
            input_data = self._clean_data_types(input_data)
            
            # Perform analysis with individual error handling
            analysis_results = {
                'analysis_metadata': {
                    'timestamp': datetime.utcnow().isoformat(),
                    'dataset_shape': list(input_data.shape),
                    'analysis_version': '1.0',
                    'significance_level': self.significance_level
                }
            }
            
            # Dataset profile
            try:
                dataset_profile = await self._create_dataset_profile(input_data)
                analysis_results['dataset_profile'] = dataset_profile
            except Exception as e:
                self.logger.warning(f"Dataset profile creation failed: {str(e)}")
                analysis_results['dataset_profile'] = {'error': str(e)}
            
            # KPI computation
            try:
                kpi_results = await self._compute_kpis(input_data)
                analysis_results['kpis'] = kpi_results
            except Exception as e:
                self.logger.warning(f"KPI computation failed: {str(e)}")
                analysis_results['kpis'] = {'error': str(e)}
            
            # Trend analysis
            try:
                trend_analysis = await self._analyze_trends(input_data)
                analysis_results['trends'] = trend_analysis
            except Exception as e:
                self.logger.warning(f"Trend analysis failed: {str(e)}")
                analysis_results['trends'] = {'error': str(e)}
            
            # Correlation analysis
            try:
                correlation_analysis = await self._analyze_correlations(input_data)
                analysis_results['correlations'] = correlation_analysis
            except Exception as e:
                self.logger.warning(f"Correlation analysis failed: {str(e)}")
                analysis_results['correlations'] = {'error': str(e)}
            
            # Hypothesis tests
            try:
                hypothesis_tests = await self._perform_hypothesis_tests(input_data)
                analysis_results['hypothesis_tests'] = hypothesis_tests
            except Exception as e:
                self.logger.warning(f"Hypothesis testing failed: {str(e)}")
                analysis_results['hypothesis_tests'] = {'error': str(e)}
            
            # Spatial analysis
            try:
                spatial_analysis = await self._analyze_spatial_patterns(input_data)
                analysis_results['spatial_analysis'] = spatial_analysis
            except Exception as e:
                self.logger.warning(f"Spatial analysis failed: {str(e)}")
                analysis_results['spatial_analysis'] = {'error': str(e)}
            
            # Distribution analysis
            try:
                distribution_analysis = await self._analyze_distributions(input_data)
                analysis_results['distributions'] = distribution_analysis
            except Exception as e:
                self.logger.warning(f"Distribution analysis failed: {str(e)}")
                analysis_results['distributions'] = {'error': str(e)}
            
            # Outlier analysis
            try:
                outlier_analysis = await self._analyze_outliers(input_data)
                analysis_results['outliers'] = outlier_analysis
            except Exception as e:
                self.logger.warning(f"Outlier analysis failed: {str(e)}")
                analysis_results['outliers'] = {'error': str(e)}
            
            # Analysis quality
            try:
                analysis_quality = await self._calculate_analysis_quality(input_data)
                analysis_results['quality_assessment'] = analysis_quality
            except Exception as e:
                self.logger.warning(f"Analysis quality calculation failed: {str(e)}")
                analysis_results['quality_assessment'] = {'error': str(e)}
            
            # Summary statistics - FIX THE METHOD CALL
            try:
                summary_statistics = await self._create_summary_statistics(input_data)
                analysis_results['summary_statistics'] = summary_statistics
            except Exception as e:
                self.logger.warning(f"Summary statistics creation failed: {str(e)}")
                analysis_results['summary_statistics'] = {'error': str(e)}
            
            # Ensure output directories exist
            docs_dir = Path(state.run_manifest['artifacts_paths']['docs_dir'])
            docs_dir.mkdir(parents=True, exist_ok=True)
            
            # Save analysis results
            try:
                output_path = docs_dir / "analysis_results.json"
                with open(output_path, 'w') as f:
                    json.dump(analysis_results, f, indent=2, default=str)
                self.logger.info(f"Saved analysis results to {output_path}")
            except Exception as e:
                self.logger.warning(f"Failed to save analysis results: {str(e)}")
            
            # Update state
            state.analysis_results = analysis_results
            
            self.logger.info("Statistical analysis completed successfully")
            return state
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            if not hasattr(state, 'errors'):
                state.errors = []
            state.errors.append(f"Analysis error: {str(e)}")
            
            # Create minimal analysis results to prevent cascade failure
            state.analysis_results = {
                'error': str(e),
                'analysis_metadata': {
                    'timestamp': datetime.utcnow().isoformat(),
                    'status': 'failed'
                }
            }
            
            return state
    
    async def _create_dataset_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create comprehensive dataset profile"""
        
        profile = {
            'basic_info': {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
                'duplicate_rows': df.duplicated().sum(),
                'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100
            },
            'column_types': {
                'numeric': len(df.select_dtypes(include=[np.number]).columns),
                'categorical': len(df.select_dtypes(include=['object', 'category']).columns),
                'datetime': len(df.select_dtypes(include=['datetime64']).columns),
                'boolean': len(df.select_dtypes(include=['bool']).columns)
            },
            'missing_data': {
                'total_missing': df.isnull().sum().sum(),
                'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
                'columns_with_missing': df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
                'complete_rows': len(df.dropna()),
                'complete_rows_percentage': (len(df.dropna()) / len(df)) * 100
            },
            'data_quality_flags': self._assess_data_quality_flags(df)
        }
        
        return profile
    
    async def _compute_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute comprehensive key performance indicators"""
        
        kpis = {
            'numeric_summary': {},
            'categorical_summary': {},
            'time_series_summary': {},
            'top_performers': {},
            'trends_summary': {}
        }
        
        # Numeric KPIs
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            try:
                series = df[col].dropna()
                if len(series) == 0:
                    continue
                
                kpis['numeric_summary'][col] = {
                    'count': int(series.count()),
                    'mean': float(series.mean()),
                    'median': float(series.median()),
                    'std': float(series.std()),
                    'min': float(series.min()),
                    'max': float(series.max()),
                    'sum': float(series.sum()),
                    'q25': float(series.quantile(0.25)),
                    'q75': float(series.quantile(0.75)),
                    'iqr': float(series.quantile(0.75) - series.quantile(0.25)),
                    'coefficient_of_variation': float(series.std() / series.mean()) if series.mean() != 0 else 0,
                    'skewness': float(series.skew()),
                    'kurtosis': float(series.kurtosis()),
                    'range': float(series.max() - series.min())
                }
                
                # Percentile analysis
                percentiles = [5, 10, 90, 95, 99]
                for p in percentiles:
                    kpis['numeric_summary'][col][f'p{p}'] = float(series.quantile(p/100))
                
            except Exception as e:
                self.logger.warning(f"Failed to compute KPIs for {col}: {e}")
        
        # Categorical KPIs
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            try:
                series = df[col].dropna()
                if len(series) == 0:
                    continue
                
                value_counts = series.value_counts()
                
                kpis['categorical_summary'][col] = {
                    'unique_count': len(value_counts),
                    'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                    'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    'most_frequent_percentage': float(value_counts.iloc[0] / len(series) * 100) if len(value_counts) > 0 else 0,
                    'top_5_values': value_counts.head(5).to_dict(),
                    'entropy': float(-sum(p * np.log2(p) for p in (value_counts / len(series)) if p > 0)),
                    'concentration_ratio': float(value_counts.head(3).sum() / len(series))  # Top 3 concentration
                }
                
            except Exception as e:
                self.logger.warning(f"Failed to compute categorical KPIs for {col}: {e}")
        
        # Time series KPIs (if time columns exist)
        time_cols = [col for col in df.columns if 'year' in col.lower() or 'date' in col.lower() or 'time' in col.lower()]
        
        if time_cols:
            kpis['time_series_summary'] = self._compute_time_series_kpis(df, time_cols)
        
        # Top performers analysis
        kpis['top_performers'] = self._identify_top_performers(df)
        
        return kpis
    
    async def _analyze_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal trends in the data"""
        
        trends = {
            'linear_trends': {},
            'seasonal_patterns': {},
            'growth_rates': {},
            'trend_significance': {}
        }
        
        # Find time columns
        time_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['year', 'month', 'date', 'time'])]
        numeric_cols = self._filter_numeric_columns(df)
        
        if not time_cols or not numeric_cols:
            return trends
        
        # Use first time column
        time_col = time_cols[0]
        
        try:
            # Sort by time
            df_sorted = df.sort_values(time_col)
            
            for metric_col in numeric_cols[:10]:  # Limit to 10 columns for performance
                try:
                    # Linear trend analysis
                    valid_data = df_sorted[[time_col, metric_col]].dropna()
                    if len(valid_data) < 3:
                        continue
                    
                    # Convert time to numeric for correlation
                    if df_sorted[time_col].dtype == 'datetime64[ns]':
                        time_numeric = pd.to_numeric(df_sorted[time_col])
                    else:
                        time_numeric = pd.to_numeric(df_sorted[time_col], errors='coerce')
                    
                    if time_numeric.isna().all():
                        continue
                    
                    # Calculate trend slope using linear regression
                    valid_indices = ~(time_numeric.isna() | df_sorted[metric_col].isna())
                    if valid_indices.sum() < 3:
                        continue
                    
                    x = time_numeric[valid_indices].values.reshape(-1, 1)
                    y = df_sorted[metric_col][valid_indices].values
                    
                    # Simple linear regression
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x.flatten(), y)
                    
                    trends['linear_trends'][metric_col] = {
                        'slope': float(slope),
                        'intercept': float(intercept),
                        'r_squared': float(r_value ** 2),
                        'p_value': float(p_value),
                        'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable',
                        'trend_strength': 'strong' if abs(r_value) > 0.7 else 'moderate' if abs(r_value) > 0.3 else 'weak',
                        'statistical_significance': p_value < self.significance_level
                    }
                    
                    # Growth rate analysis
                    if len(valid_data) > 1:
                        first_value = valid_data[metric_col].iloc[0]
                        last_value = valid_data[metric_col].iloc[-1]
                        
                        if first_value != 0:
                            total_growth = ((last_value - first_value) / first_value) * 100
                            periods = len(valid_data) - 1
                            compound_growth = ((last_value / first_value) ** (1/periods) - 1) * 100 if periods > 0 else 0
                            
                            trends['growth_rates'][metric_col] = {
                                'total_growth_percent': float(total_growth),
                                'compound_annual_growth_rate': float(compound_growth),
                                'periods_analyzed': int(periods),
                                'start_value': float(first_value),
                                'end_value': float(last_value)
                            }
                    
                    # Seasonal pattern detection (if enough data points)
                    if len(valid_data) >= 12:  # Need at least 12 points for seasonal analysis
                        seasonal_analysis = self._detect_seasonality(valid_data[metric_col])
                        if seasonal_analysis:
                            trends['seasonal_patterns'][metric_col] = seasonal_analysis
                    
                except Exception as e:
                    self.logger.warning(f"Failed to analyze trends for {metric_col}: {e}")
        
        except Exception as e:
            self.logger.warning(f"Trend analysis failed: {e}")
        
        return trends
    
    async def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between variables with robust error handling"""
        
        correlations = {
            'pearson_correlations': {},
            'spearman_correlations': {},
            'significant_correlations': [],
            'correlation_summary': {},
            'strong_correlations': []
        }
        
        numeric_cols = self._filter_numeric_columns(df)
        
        if len(numeric_cols) < 2:
            return correlations
        
        try:
            # Clean data first - remove columns with all NaN or constant values
            clean_numeric_cols = []
            for col in numeric_cols:
                col_data = df[col].dropna()
                if len(col_data) > 1 and col_data.nunique() > 1 and self._is_numeric_safe(df[col]):
                    clean_numeric_cols.append(col)

            if len(clean_numeric_cols) < 2:
                return correlations

            # Compute correlation matrices
            df_clean = df[clean_numeric_cols].dropna()
            
            if len(df_clean) < 3:
                return correlations
            
            # Pearson correlations
            try:
                pearson_corr = df_clean.corr(method='pearson')
                correlations['pearson_correlations'] = pearson_corr.fillna(0).to_dict()
            except Exception as e:
                self.logger.warning(f"Pearson correlation failed: {e}")
            
            # Spearman correlations
            try:
                spearman_corr = df_clean.corr(method='spearman')
                correlations['spearman_correlations'] = spearman_corr.fillna(0).to_dict()
            except Exception as e:
                self.logger.warning(f"Spearman correlation failed: {e}")
            
            # Find significant correlations with robust error handling
            significant_pairs = []
            strong_correlations = []
            
            for i in range(len(clean_numeric_cols)):
                for j in range(i+1, len(clean_numeric_cols)):
                    col1, col2 = clean_numeric_cols[i], clean_numeric_cols[j]
                    
                    try:
                        # Get clean data for both columns
                        data1 = df_clean[col1].dropna()
                        data2 = df_clean[col2].dropna()
                        
                        # Ensure same length
                        common_idx = df_clean[[col1, col2]].dropna().index
                        if len(common_idx) < 3:
                            continue
                        
                        data1_aligned = df_clean.loc[common_idx, col1]
                        data2_aligned = df_clean.loc[common_idx, col2]
                        
                        # Check for variance
                        if data1_aligned.var() == 0 or data2_aligned.var() == 0:
                            continue
                        
                            # Pearson correlation test
                        try:
                            pearson_r, p_value_pearson = pearsonr(data1_aligned, data2_aligned)
                            if np.isnan(pearson_r):
                                continue
                        except Exception as e:
                            self.logger.warning(f"Pearson test failed for {col1}-{col2}: {e}")
                            continue
                            
                            # Spearman correlation test
                        try:
                            spearman_r, p_value_spearman = spearmanr(data1_aligned, data2_aligned)
                            if np.isnan(spearman_r):
                                spearman_r = 0.0
                                p_value_spearman = 1.0
                        except Exception as e:
                            self.logger.warning(f"Spearman test failed for {col1}-{col2}: {e}")
                            spearman_r = 0.0
                            p_value_spearman = 1.0
                            
                            correlation_pair = {
                                'variable_1': col1,
                                'variable_2': col2,
                                'pearson_r': float(pearson_r),
                                'spearman_r': float(spearman_r),
                                'pearson_p_value': float(p_value_pearson),
                                'spearman_p_value': float(p_value_spearman),
                                'pearson_significant': p_value_pearson < self.significance_level,
                                'spearman_significant': p_value_spearman < self.significance_level,
                            'correlation_strength': self._classify_correlation_strength(abs(pearson_r)),
                            'sample_size': len(data1_aligned)
                            }
                            
                            if p_value_pearson < self.significance_level or p_value_spearman < self.significance_level:
                                significant_pairs.append(correlation_pair)
                            
                            if abs(pearson_r) > 0.7 or abs(spearman_r) > 0.7:
                                strong_correlations.append(correlation_pair)
                                
                    except Exception as e:
                        self.logger.warning(f"Failed to test correlation significance for {col1}-{col2}: {e}")
                        continue
            
            correlations['significant_correlations'] = significant_pairs
            correlations['strong_correlations'] = strong_correlations
            
            # Correlation summary
            correlations['correlation_summary'] = {
                'total_pairs_tested': len(clean_numeric_cols) * (len(clean_numeric_cols) - 1) // 2,
                'significant_pairs': len(significant_pairs),
                'strong_correlations': len(strong_correlations),
                'columns_analyzed': len(clean_numeric_cols)
            }
        
        except Exception as e:
            self.logger.warning(f"Correlation analysis failed: {e}")
            correlations['error'] = str(e)
        
        return correlations
    
    async def _perform_hypothesis_tests(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical hypothesis tests"""
        
        hypothesis_tests = {
            'group_comparisons': [],
            'independence_tests': [],
            'normality_tests': {},
            'test_summary': {}
        }
        
        numeric_cols = self._filter_numeric_columns(df)
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Group comparison tests
        for cat_col in categorical_cols[:3]:  # Limit to 3 categorical columns
            unique_groups = df[cat_col].dropna().unique()
            
            if 2 <= len(unique_groups) <= 5:  # Only test if reasonable number of groups
                for num_col in numeric_cols[:5]:  # Limit to 5 numeric columns
                    try:
                        groups_data = []
                        group_names = []
                        
                        for group in unique_groups:
                            group_data = df[df[cat_col] == group][num_col].dropna()
                            if len(group_data) >= 3:  # Minimum sample size
                                groups_data.append(group_data)
                                group_names.append(group)
                        
                        if len(groups_data) >= 2:
                            # Perform appropriate test based on number of groups
                            if len(groups_data) == 2:
                                # Two-sample tests
                                group1, group2 = groups_data[0], groups_data[1]
                                
                                # Check normality first
                                _, p_norm1 = stats.shapiro(group1.sample(min(len(group1), 5000)))
                                _, p_norm2 = stats.shapiro(group2.sample(min(len(group2), 5000)))
                                
                                normal_dist = p_norm1 > 0.05 and p_norm2 > 0.05
                                
                                if normal_dist:
                                    # t-test
                                    statistic, p_value = ttest_ind(group1, group2)
                                    test_name = "Independent t-test"
                                else:
                                    # Mann-Whitney U test
                                    statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
                                    test_name = "Mann-Whitney U test"
                                
                                effect_size = abs(group1.mean() - group2.mean()) / np.sqrt(((group1.std()**2) + (group2.std()**2)) / 2)
                                
                                hypothesis_tests['group_comparisons'].append({
                                    'grouping_variable': cat_col,
                                    'test_variable': num_col,
                                    'test_type': test_name,
                                    'groups_tested': group_names,
                                    'statistic': float(statistic),
                                    'p_value': float(p_value),
                                    'significant': p_value < self.significance_level,
                                    'effect_size': float(effect_size),
                                    'effect_size_interpretation': self._classify_effect_size(effect_size),
                                    'group_means': {str(name): float(data.mean()) for name, data in zip(group_names, groups_data)}
                                })
                            
                            else:
                                # Multiple groups - ANOVA or Kruskal-Wallis
                                try:
                                    # Check normality assumption for ANOVA
                                    normality_ok = all(stats.shapiro(group.sample(min(len(group), 5000)))[1] > 0.05 
                                                     for group in groups_data)
                                    
                                    if normality_ok:
                                        # One-way ANOVA
                                        statistic, p_value = stats.f_oneway(*groups_data)
                                        test_name = "One-way ANOVA"
                                    else:
                                        # Kruskal-Wallis test
                                        statistic, p_value = stats.kruskal(*groups_data)
                                        test_name = "Kruskal-Wallis test"
                                    
                                    hypothesis_tests['group_comparisons'].append({
                                        'grouping_variable': cat_col,
                                        'test_variable': num_col,
                                        'test_type': test_name,
                                        'groups_tested': group_names,
                                        'statistic': float(statistic),
                                        'p_value': float(p_value),
                                        'significant': p_value < self.significance_level,
                                        'group_means': {str(name): float(data.mean()) for name, data in zip(group_names, groups_data)}
                                    })
                                    
                                except Exception as e:
                                    self.logger.warning(f"Multi-group test failed for {cat_col}-{num_col}: {e}")
                    
                    except Exception as e:
                        self.logger.warning(f"Group comparison test failed for {cat_col}-{num_col}: {e}")
        
        # Independence tests (Chi-square)
        for i, cat_col1 in enumerate(categorical_cols[:3]):
            for cat_col2 in categorical_cols[i+1:4]:  # Avoid duplicate tests
                try:
                    contingency_table = pd.crosstab(df[cat_col1], df[cat_col2])
                    
                    if contingency_table.size > 1 and contingency_table.sum().sum() > 5:
                        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                        
                        # Cramér's V for effect size
                        n = contingency_table.sum().sum()
                        cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
                        
                        hypothesis_tests['independence_tests'].append({
                            'variable_1': cat_col1,
                            'variable_2': cat_col2,
                            'test_type': "Chi-square test of independence",
                            'chi2_statistic': float(chi2),
                            'p_value': float(p_value),
                            'degrees_of_freedom': int(dof),
                            'significant': p_value < self.significance_level,
                            'cramers_v': float(cramers_v),
                            'effect_size_interpretation': self._classify_cramers_v(cramers_v),
                            'contingency_table': contingency_table.to_dict()
                        })
                
                except Exception as e:
                    self.logger.warning(f"Independence test failed for {cat_col1}-{cat_col2}: {e}")
        
        # Normality tests for numeric variables
        for col in numeric_cols[:10]:  # Limit to 10 columns
            try:
                data = df[col].dropna()
                if len(data) > 3:
                    # Shapiro-Wilk test (for smaller samples)
                    if len(data) <= 5000:
                        statistic, p_value = stats.shapiro(data)
                        test_name = "Shapiro-Wilk"
                    else:
                        # Kolmogorov-Smirnov test (for larger samples)
                        statistic, p_value = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
                        test_name = "Kolmogorov-Smirnov"
                    
                    hypothesis_tests['normality_tests'][col] = {
                        'test_type': test_name,
                        'statistic': float(statistic),
                        'p_value': float(p_value),
                        'normal_distribution': p_value > self.significance_level,
                        'sample_size': len(data)
                    }
            
            except Exception as e:
                self.logger.warning(f"Normality test failed for {col}: {e}")
        
        # Test summary
        hypothesis_tests['test_summary'] = {
            'total_group_comparisons': len(hypothesis_tests['group_comparisons']),
            'significant_group_differences': sum(1 for test in hypothesis_tests['group_comparisons'] if test['significant']),
            'total_independence_tests': len(hypothesis_tests['independence_tests']),
            'significant_associations': sum(1 for test in hypothesis_tests['independence_tests'] if test['significant']),
            'variables_tested_for_normality': len(hypothesis_tests['normality_tests']),
            'normally_distributed_variables': sum(1 for test in hypothesis_tests['normality_tests'].values() if test['normal_distribution'])
        }
        
        return hypothesis_tests
    
    async def _analyze_spatial_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze spatial/geographic patterns in the data"""
        
        spatial_analysis = {
            'geographic_columns': [],
            'spatial_distribution': {},
            'regional_comparisons': {},
            'geographic_trends': {}
        }
        
        # Identify geographic columns
        geo_keywords = ['district', 'state', 'city', 'region', 'location', 'area', 'zone', 'mandal', 'tehsil']
        geo_cols = []
        
        for col in df.columns:
            if any(keyword in col.lower() for keyword in geo_keywords):
                geo_cols.append(col)
        
        spatial_analysis['geographic_columns'] = geo_cols
        
        if not geo_cols:
            return spatial_analysis
        
        # Use primary geographic column
        primary_geo_col = geo_cols[0]
        numeric_cols = self._filter_numeric_columns(df)
        
        try:
            # Spatial distribution analysis
            geo_summary = {}
            for geo_unit in df[primary_geo_col].dropna().unique():
                geo_data = df[df[primary_geo_col] == geo_unit]
                
                geo_summary[str(geo_unit)] = {
                    'record_count': len(geo_data),
                    'data_completeness': (geo_data.notna().sum().sum() / (len(geo_data) * len(geo_data.columns))) * 100
                }
                
                # Add numeric summaries
                for num_col in numeric_cols[:5]:  # Limit to 5 columns
                    if num_col in geo_data.columns:
                        values = geo_data[num_col].dropna()
                        if len(values) > 0:
                            geo_summary[str(geo_unit)][f"{num_col}_total"] = float(values.sum())
                            geo_summary[str(geo_unit)][f"{num_col}_mean"] = float(values.mean())
                            geo_summary[str(geo_unit)][f"{num_col}_median"] = float(values.median())
            
            spatial_analysis['spatial_distribution'] = geo_summary
            
            # Regional comparisons (top and bottom performers)
            for num_col in numeric_cols[:3]:  # Limit to 3 columns
                try:
                    regional_stats = df.groupby(primary_geo_col)[num_col].agg(['sum', 'mean', 'count']).reset_index()
                    regional_stats = regional_stats.dropna()
                    
                    if len(regional_stats) > 1:
                        # Top and bottom performers by total
                        top_performers = regional_stats.nlargest(5, 'sum')
                        bottom_performers = regional_stats.nsmallest(5, 'sum')
                        
                        # Calculate inequality metrics with safe error handling
                        total_values = regional_stats['sum'].values
                        if len(total_values) > 1 and total_values.sum() > 0:
                            try:
                                # Gini coefficient
                                gini = self._calculate_gini_coefficient(total_values)
                                
                                # Coefficient of variation
                                cv = float(regional_stats['sum'].std() / regional_stats['sum'].mean()) if regional_stats['sum'].mean() > 0 else 0.0
                                
                                # Max-min ratio
                                max_min_ratio = float(regional_stats['sum'].max() / regional_stats['sum'].min()) if regional_stats['sum'].min() > 0 else float('inf')
                                
                                spatial_analysis['regional_comparisons'][num_col] = {
                                    'top_5_regions': top_performers.to_dict('records'),
                                    'bottom_5_regions': bottom_performers.to_dict('records'),
                                    'inequality_metrics': {
                                        'gini_coefficient': gini,
                                        'coefficient_of_variation': cv,
                                        'max_min_ratio': max_min_ratio
                                    },
                                    'regional_summary': {
                                        'total_regions': len(regional_stats),
                                        'total_value': float(regional_stats['sum'].sum()),
                                        'average_per_region': float(regional_stats['sum'].mean()),
                                        'median_per_region': float(regional_stats['sum'].median())
                                    }
                                }
                            except Exception as e:
                                self.logger.warning(f"Inequality metrics calculation failed for {num_col}: {e}")
                
                except Exception as e:
                    self.logger.warning(f"Regional comparison failed for {num_col}: {e}")
        
        except Exception as e:
            self.logger.warning(f"Spatial analysis failed: {e}")
        
        return spatial_analysis
    
    async def _analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze statistical distributions of numeric variables"""
        
        distributions = {
            'distribution_tests': {},
            'distribution_parameters': {},
            'outlier_analysis': {},
            'skewness_analysis': {}
        }
        
        numeric_cols = self._filter_numeric_columns(df)
        
        for col in numeric_cols[:10]:  # Limit to 10 columns
            try:
                data = df[col].dropna()
                if len(data) < 3:
                    continue
                
                # Basic distribution parameters
                distributions['distribution_parameters'][col] = {
                    'mean': float(data.mean()),
                    'median': float(data.median()),
                    'mode': float(data.mode().iloc[0]) if len(data.mode()) > 0 else float(data.median()),
                    'std': float(data.std()),
                    'variance': float(data.var()),
                    'skewness': float(data.skew()),
                    'kurtosis': float(data.kurtosis()),
                    'range': float(data.max() - data.min()),
                    'iqr': float(data.quantile(0.75) - data.quantile(0.25))
                }
                
                # Skewness analysis
                skewness = data.skew()
                if abs(skewness) < 0.5:
                    skew_interpretation = "approximately symmetric"
                elif abs(skewness) < 1:
                    skew_interpretation = "moderately skewed"
                else:
                    skew_interpretation = "highly skewed"
                
                skew_direction = "right" if skewness > 0 else "left" if skewness < 0 else "symmetric"
                
                distributions['skewness_analysis'][col] = {
                    'skewness_value': float(skewness),
                    'skew_direction': skew_direction,
                    'skew_interpretation': skew_interpretation
                }
                
                # Test for common distributions
                distribution_tests = {}
                
                # Test for normal distribution
                if len(data) <= 5000:
                    _, p_normal = stats.shapiro(data)
                else:
                    _, p_normal = stats.jarque_bera(data)
                
                distribution_tests['normal'] = {
                    'p_value': float(p_normal),
                    'is_normal': p_normal > self.significance_level
                }
                
                # Test for exponential distribution
                try:
                    if data.min() >= 0:  # Exponential distribution requires non-negative values
                        scale_param = data.mean()
                        _, p_exp = stats.kstest(data, lambda x: stats.expon.cdf(x, scale=scale_param))
                        distribution_tests['exponential'] = {
                            'p_value': float(p_exp),
                            'is_exponential': p_exp > self.significance_level
                        }
                except:
                    pass
                
                distributions['distribution_tests'][col] = distribution_tests
                
            except Exception as e:
                self.logger.warning(f"Distribution analysis failed for {col}: {e}")
        
        return distributions
    
    async def _analyze_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive outlier analysis"""
        
        outlier_analysis = {
            'outlier_detection': {},
            'outlier_summary': {},
            'outlier_impact': {}
        }
        
        numeric_cols = self._filter_numeric_columns(df)
        
        for col in numeric_cols[:10]:  # Limit to 10 columns
            try:
                data = df[col].dropna()
                if len(data) < 4:
                    continue
                
                # Multiple outlier detection methods
                outlier_methods = {}
                
                # 1. IQR method
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                iqr_outliers = (data < lower_bound) | (data > upper_bound)
                outlier_methods['iqr'] = {
                    'outlier_count': int(iqr_outliers.sum()),
                    'outlier_percentage': float(iqr_outliers.sum() / len(data) * 100),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound)
                }
                
                # 2. Z-score method
                z_scores = np.abs(stats.zscore(data))
                z_outliers = z_scores > 3
                outlier_methods['zscore'] = {
                    'outlier_count': int(z_outliers.sum()),
                    'outlier_percentage': float(z_outliers.sum() / len(data) * 100),
                    'threshold': 3.0
                }
                
                # 3. Modified Z-score method (using median)
                median = data.median()
                mad = np.median(np.abs(data - median))
                modified_z_scores = 0.6745 * (data - median) / mad if mad > 0 else np.zeros_like(data)
                modified_z_outliers = np.abs(modified_z_scores) > 3.5
                outlier_methods['modified_zscore'] = {
                    'outlier_count': int(modified_z_outliers.sum()),
                    'outlier_percentage': float(modified_z_outliers.sum() / len(data) * 100),
                    'threshold': 3.5
                }
                
                outlier_analysis['outlier_detection'][col] = outlier_methods
                
                # Outlier impact analysis
                if iqr_outliers.any():
                    data_without_outliers = data[~iqr_outliers]
                    
                    impact = {
                        'mean_change': float(abs(data.mean() - data_without_outliers.mean())),
                        'median_change': float(abs(data.median() - data_without_outliers.median())),
                        'std_change': float(abs(data.std() - data_without_outliers.std())),
                        'most_extreme_outlier': float(data[iqr_outliers].iloc[0]) if len(data[iqr_outliers]) > 0 else None
                    }
                    
                    outlier_analysis['outlier_impact'][col] = impact
                
            except Exception as e:
                self.logger.warning(f"Outlier analysis failed for {col}: {e}")
        
        # Overall outlier summary
        total_outliers = sum(
            methods.get('iqr', {}).get('outlier_count', 0) 
            for methods in outlier_analysis['outlier_detection'].values()
        )
        
        total_data_points = len(df) * len(numeric_cols)
        
        outlier_analysis['outlier_summary'] = {
            'total_outliers_detected': total_outliers,
            'overall_outlier_percentage': (total_outliers / total_data_points * 100) if total_data_points > 0 else 0,
            'columns_with_outliers': len([col for col in outlier_analysis['outlier_detection'] if outlier_analysis['outlier_detection'][col]['iqr']['outlier_count'] > 0]),
            'recommendation': 'Consider outlier treatment' if total_outliers > total_data_points * 0.05 else 'Outlier levels are acceptable'
        }
        
        return outlier_analysis
    
    async def _calculate_analysis_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate overall analysis quality score"""
        
        scores = {
            'completeness_score': 0,
            'sample_size_score': 0,
            'variability_score': 0,
            'data_type_diversity_score': 0,
            'overall_score': 0
        }
        
        # Completeness score (0-100)
        completeness = (1 - df.isna().sum().sum() / (len(df) * len(df.columns))) * 100
        scores['completeness_score'] = max(0, min(100, completeness))
        
        # Sample size score (0-100)
        if len(df) >= 1000:
            scores['sample_size_score'] = 100
        elif len(df) >= 100:
            scores['sample_size_score'] = 80
        elif len(df) >= 30:
            scores['sample_size_score'] = 60
        else:
            scores['sample_size_score'] = 30
        
        # Variability score (0-100)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            non_constant_cols = sum(1 for col in numeric_cols if df[col].nunique() > 1)
            scores['variability_score'] = (non_constant_cols / len(numeric_cols)) * 100
        else:
            scores['variability_score'] = 50
        
        # Data type diversity score (0-100)
        data_types = df.dtypes.value_counts()
        type_diversity = min(len(data_types) / 3, 1) * 100  # Max score if 3+ different types
        scores['data_type_diversity_score'] = type_diversity
        
        # Overall score (weighted average)
        weights = {
            'completeness_score': 0.3,
            'sample_size_score': 0.25,
            'variability_score': 0.25,
            'data_type_diversity_score': 0.2
        }
        
        overall = sum(scores[key] * weights[key] for key in weights) / 100
        scores['overall_score'] = round(overall, 3)
        
        return scores
    
    async def _generate_statistical_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive statistical summary"""
        
        summary = {
            'data_overview': {
                'shape': df.shape,
                'memory_usage_mb': float(df.memory_usage(deep=True).sum() / (1024 * 1024)),
                'data_types': df.dtypes.value_counts().to_dict()
            },
            'completeness': {
                'overall_completeness': float((1 - df.isna().sum().sum() / (len(df) * len(df.columns))) * 100),
                'columns_with_missing': len([col for col in df.columns if df[col].isna().any()]),
                'complete_rows': int((~df.isna().any(axis=1)).sum())
            },
            'variability': {},
            'data_quality_flags': []
        }
        
        # Variability analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            cv_values = []
            for col in numeric_cols:
                mean_val = df[col].mean()
                if mean_val != 0:
                    cv = df[col].std() / mean_val
                    cv_values.append(cv)
            
            if cv_values:
                summary['variability'] = {
                    'average_coefficient_variation': float(np.mean(cv_values)),
                    'high_variability_columns': len([cv for cv in cv_values if cv > 1.0])
                }
        
        # Data quality flags
        if summary['completeness']['overall_completeness'] < 90:
            summary['data_quality_flags'].append("High level of missing data detected")
        
        if df.duplicated().sum() > len(df) * 0.05:
            summary['data_quality_flags'].append("High number of duplicate rows detected")
        
        return summary
    
    def _assess_data_quality_flags(self, df: pd.DataFrame) -> List[str]:
        """Assess data quality and return list of issues/flags"""
        flags = []
        
        # Check completeness
        missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_percentage > 10:
            flags.append(f"High level of missing data ({missing_percentage:.1f}%)")
        
        # Check duplicates
        duplicate_percentage = (df.duplicated().sum() / len(df)) * 100
        if duplicate_percentage > 5:
            flags.append(f"High number of duplicate rows ({duplicate_percentage:.1f}%)")
        
        # Check data type consistency
        for col in df.columns:
            if df[col].dtype == 'object':
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio > 0.8 and df[col].nunique() > 100:
                    flags.append(f"Column '{col}' has very high cardinality")
        
        # Check for potential outliers in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            outlier_count = self._count_outliers_iqr(df[col])
            if outlier_count > len(df) * 0.05:
                flags.append(f"Column '{col}' has many outliers ({outlier_count} values)")
        
        # Check for constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            flags.append(f"Constant columns detected: {', '.join(constant_cols)}")
        
        return flags
    
    def _identify_top_performers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify top performing entities based on key metrics"""
        top_performers = {
            'analysis_method': 'Multi-metric scoring',
            'top_entities': [],
            'performance_metrics': {},
            'ranking_criteria': []
        }
        
        try:
            # Find potential entity identifier columns (area, district, region, etc.)
            entity_cols = []
            for col in df.columns:
                if any(keyword in col.lower() for keyword in 
                       ['area', 'district', 'region', 'zone', 'division', 'mandal', 'block', 'village']):
                    if df[col].dtype == 'object' and df[col].nunique() > 1:
                        entity_cols.append(col)
            
            if not entity_cols:
                top_performers['note'] = 'No clear entity identifier columns found'
                return top_performers
            
            # Use the first suitable entity column
            entity_col = entity_cols[0]
            top_performers['entity_column'] = entity_col
            top_performers['ranking_criteria'].append(f'Grouped by {entity_col}')
            
            # Find numeric performance metrics
            numeric_cols = self._filter_numeric_columns(df)
            
            # Remove outlier flag columns and utility columns
            performance_cols = [col for col in numeric_cols 
                              if not col.endswith('_outlier_flag') 
                              and not any(x in col.lower() for x in ['id', 'code', 'year', 'flag'])]
            
            if len(performance_cols) < 2:
                top_performers['note'] = 'Insufficient numeric metrics for performance analysis'
                return top_performers
            
            # Aggregate by entity and compute performance scores
            entity_performance = []
            
            for entity in df[entity_col].dropna().unique():
                entity_data = df[df[entity_col] == entity]
                
                if len(entity_data) == 0:
                    continue
                
                # Calculate performance metrics
                performance_score = 0
                metric_count = 0
                entity_metrics = {'entity': entity}
                
                for col in performance_cols[:8]:  # Limit to top 8 metrics
                    values = entity_data[col].dropna()
                    if len(values) > 0:
                        metric_value = values.mean()
                        entity_metrics[f'{col}_avg'] = float(metric_value)
                        
                        # Normalize score (higher is better, assuming positive metrics)
                        col_max = df[col].max()
                        col_min = df[col].min()
                        if col_max > col_min:
                            normalized_score = (metric_value - col_min) / (col_max - col_min)
                            performance_score += normalized_score
                            metric_count += 1
                
                # Calculate final performance score
                if metric_count > 0:
                    entity_metrics['overall_performance_score'] = float(performance_score / metric_count)
                    entity_metrics['metrics_count'] = metric_count
                    entity_performance.append(entity_metrics)
            
            # Sort by performance score and get top performers
            entity_performance.sort(key=lambda x: x['overall_performance_score'], reverse=True)
            
            # Store top 10 performers
            top_performers['top_entities'] = entity_performance[:10]
            top_performers['total_entities_analyzed'] = len(entity_performance)
            
            # Store performance metrics summary
            if entity_performance:
                scores = [e['overall_performance_score'] for e in entity_performance]
                top_performers['performance_metrics'] = {
                    'highest_score': float(max(scores)),
                    'lowest_score': float(min(scores)),
                    'average_score': float(np.mean(scores)),
                    'score_std': float(np.std(scores))
                }
            
            top_performers['ranking_criteria'].extend([
                f'Based on {len(performance_cols)} numeric metrics',
                'Normalized scoring (0-1 scale)',
                'Higher scores indicate better performance'
            ])
            
        except Exception as e:
            top_performers['error'] = f"Top performers analysis failed: {str(e)}"
            self.logger.warning(f"Top performers analysis failed: {str(e)}")
        
        return top_performers
    
    def _classify_correlation_strength(self, correlation: float) -> str:
        """Classify correlation strength based on absolute value"""
        abs_corr = abs(correlation)
        if abs_corr >= 0.7:
            return 'strong'
        elif abs_corr >= 0.5:
            return 'moderate'
        elif abs_corr >= 0.3:
            return 'weak'
        else:
            return 'negligible'
    
    def _classify_cramers_v(self, cramers_v: float) -> str:
        """Classify Cramer's V strength"""
        if cramers_v >= 0.25:
            return 'strong'
        elif cramers_v >= 0.15:
            return 'moderate'
        elif cramers_v >= 0.05:
            return 'weak'
        else:
            return 'negligible'
    
    def _calculate_gini_coefficient(self, values) -> float:
        """Calculate Gini coefficient for inequality measurement"""
        try:
            # Convert to numpy array and handle various input types
            if hasattr(values, 'values'):
                values = values.values
            
            values = np.array(values, dtype=float)
            values = values[~np.isnan(values)]  # Remove NaN values
            
            if len(values) <= 1:
                return 0.0
            
            values = np.sort(values)
            n = len(values)
            
            if np.sum(values) == 0:
                return 0.0
            
            index = np.arange(1, n + 1)
            return float((2 * np.sum(index * values)) / (n * np.sum(values)) - (n + 1) / n)
        except Exception as e:
            return 0.0
    
    def _classify_effect_size(self, effect_size: float) -> str:
        """Classify effect size (Cohen's d)"""
        abs_effect = abs(effect_size)
        if abs_effect >= 0.8:
            return 'large'
        elif abs_effect >= 0.5:
            return 'medium'
        elif abs_effect >= 0.2:
            return 'small'
        else:
            return 'negligible'
    
    async def _create_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create comprehensive summary statistics - FIXED METHOD SIGNATURE"""
        
        summary = {
            'dataset_overview': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'memory_usage_mb': float(df.memory_usage(deep=True).sum() / (1024 * 1024)),
                'missing_data_percentage': float((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100),
                'duplicate_rows': int(df.duplicated().sum())
            },
            'column_summary': {
                'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
                'datetime_columns': len(df.select_dtypes(include=['datetime64']).columns)
            },
            'data_quality': {
                'completeness_score': float((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100),
                'columns_with_missing': len([col for col in df.columns if df[col].isnull().any()]),
                'high_cardinality_columns': len([col for col in df.select_dtypes(include=['object']).columns 
                                               if df[col].nunique() / len(df) > 0.8])
            },
            'basic_statistics': {}
        }
        
        # Add basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            for col in numeric_cols[:5]:  # Limit to 5 columns
                try:
                    col_data = df[col].dropna()
                    if len(col_data) > 0:
                        summary['basic_statistics'][col] = {
                            'count': int(len(col_data)),
                            'mean': float(col_data.mean()),
                            'median': float(col_data.median()),
                            'std': float(col_data.std()),
                            'min': float(col_data.min()),
                            'max': float(col_data.max())
                        }
                except Exception as e:
                    self.logger.warning(f"Failed to compute basic statistics for {col}: {e}")
        
        return summary
    
    # Helper methods
    def _count_outliers_iqr(self, series: pd.Series) -> int:
        """Count outliers using IQR method"""
        try:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return int(((series < lower_bound) | (series > upper_bound)).sum())
        except:
            return 0
    
    def _count_outliers_zscore(self, series: pd.Series, threshold: float = 3.0) -> int:
        """Count outliers using Z-score method"""
        try:
            z_scores = np.abs(stats.zscore(series))
            return int((z_scores > threshold).sum())
        except:
            return 0
    
    def _test_normality(self, series: pd.Series) -> Dict[str, Any]:
        """Test for normality of a series"""
        try:
            if len(series) < 20:
                return {'test': 'sample_too_small', 'is_normal': False}
            
            stat, p_value = normaltest(series)
            return {
                'test': 'dangostino_pearson',
                'statistic': float(stat),
                'p_value': float(p_value),
                'is_normal': p_value > 0.05
            }
        except:
            return {'test': 'failed', 'is_normal': False}
    
    def _detect_seasonality(self, series: pd.Series) -> Dict[str, Any]:
        """Basic seasonality detection using autocorrelation"""
        try:
            if len(series) < 24:
                return {'seasonality_score': 0.0, 'note': 'insufficient_data'}
            
            # Calculate autocorrelation at various lags
            max_lag = min(len(series) // 4, 12)
            autocorrs = []
            for lag in range(1, max_lag + 1):
                try:
                    autocorr = series.autocorr(lag=lag)
                    if pd.notna(autocorr):
                        autocorrs.append(abs(autocorr))
                except:
                    continue
            
            if not autocorrs:
                return {'seasonality_score': 0.0, 'note': 'autocorr_failed'}
            
            return {
                'seasonality_score': float(max(autocorrs)),
                'max_lag': max_lag,
                'significant_lags': len([ac for ac in autocorrs if ac > 0.3])
            }
        except Exception as e:
            return {'seasonality_score': 0.0, 'error': str(e)}
    
    def _calculate_effect_size(self, group1: pd.Series, group2: pd.Series) -> float:
        """Calculate Cohen's d effect size"""
        try:
            mean_diff = group1.mean() - group2.mean()
            pooled_std = np.sqrt(((len(group1) - 1) * group1.var() + (len(group2) - 1) * group2.var()) / 
                               (len(group1) + len(group2) - 2))
            
            if pooled_std == 0:
                return 0.0
            
            return float(mean_diff / pooled_std)
        except:
            return 0.0

    def _compute_time_series_kpis(self, df: pd.DataFrame, time_cols: List[str]) -> Dict[str, Any]:
        """Compute time series specific KPIs"""
        time_kpis = {}
        
        for time_col in time_cols:
            try:
                if time_col in df.columns:
                    series = df[time_col].dropna()
                    time_kpis[time_col] = {
                        'unique_periods': len(series.unique()),
                        'date_range': {
                            'start': str(series.min()),
                            'end': str(series.max())
                        },
                        'missing_count': df[time_col].isnull().sum()
                    }
            except Exception as e:
                self.logger.warning(f"Time series KPI failed for {time_col}: {e}")
        
        return time_kpis

    def _is_numeric_safe(self, series: pd.Series) -> bool:
        """Check if series can be safely used in numeric operations"""
        try:
            if series.dtype in ['object', 'string']:
                return False
            if hasattr(series.dtype, 'dtype') and series.dtype.dtype == 'object':
                return False
            return pd.api.types.is_numeric_dtype(series)
        except:
            return False

    def _filter_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """Get truly numeric columns, excluding problematic ones"""
        numeric_cols = []
        for col in df.select_dtypes(include=[np.number]).columns:
            try:
                # Skip problematic columns
                if any(skip_word in col.lower() for skip_word in ['weekofyear', '_flag', '_id']):
                    continue
                
                series = df[col].dropna()
                if len(series) > 0:
                    # Test basic operations
                    _ = float(series.iloc[0])
                    _ = series.mean()
                    _ = series.std()
                    numeric_cols.append(col)
            except Exception as e:
                self.logger.warning(f"Excluding {col} from numeric analysis: {e}")
                continue
        return numeric_cols

    def _clean_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize data types to prevent analysis errors"""
        df_clean = df.copy()
        
        for col in df_clean.columns:
            try:
                # Skip if already numeric and working
                if pd.api.types.is_numeric_dtype(df_clean[col]) and not df_clean[col].dtype == 'object':
                    continue
                    
                # Fix object columns that should be numeric
                if df_clean[col].dtype == 'object':
                    # Special handling for week/date columns
                    if 'week' in col.lower() or 'day' in col.lower() or 'month' in col.lower() or 'year' in col.lower():
                        numeric_converted = pd.to_numeric(df_clean[col], errors='coerce')
                        if not numeric_converted.isna().all():
                            df_clean[col] = numeric_converted
                            self.logger.info(f"Converted {col} from object to numeric")
                            continue
                    
                    # Try general numeric conversion
                    numeric_converted = pd.to_numeric(df_clean[col], errors='coerce')
                    if not numeric_converted.isna().all():
                        non_null_ratio = numeric_converted.notna().sum() / len(df_clean)
                        if non_null_ratio > 0.7:  # If 70%+ can be converted
                            df_clean[col] = numeric_converted
                            self.logger.info(f"Converted {col} from object to numeric")
                    
            except Exception as e:
                self.logger.warning(f"Could not clean data type for column {col}: {e}")
        
        return df_clean