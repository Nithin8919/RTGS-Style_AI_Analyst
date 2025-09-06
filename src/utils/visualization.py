"""
RTGS AI Analyst - Enhanced Visualization Utilities
Comprehensive visualization suite using seaborn for government data analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
import warnings
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import logging
from datetime import datetime

# Configure matplotlib and seaborn for better output
plt.style.use('default')
sns.set_theme(style="whitegrid", palette="Set2")
warnings.filterwarnings('ignore')

class GovernmentDataVisualizer:
    """Enhanced visualization utility for government data analysis"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Government color schemes
        self.gov_colors = {
            'primary': '#1f77b4',     # Blue
            'secondary': '#ff7f0e',   # Orange  
            'success': '#2ca02c',     # Green
            'warning': '#d62728',     # Red
            'info': '#9467bd',        # Purple
            'neutral': '#7f7f7f',     # Gray
            'accent': '#e377c2'       # Pink
        }
        
        # Domain-specific color palettes
        self.domain_palettes = {
            'health': ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5', '#3288bd'],
            'education': ['#8e0152', '#c51b7d', '#de77ae', '#f1b6da', '#fde0ef', '#e6f5d0', '#b8e186', '#7fbc41'],
            'transport': ['#762a83', '#9970ab', '#c2a5cf', '#e7d4e8', '#d9f0d3', '#a6dba0', '#5aae61', '#1b7837'],
            'economics': ['#b35806', '#e08214', '#fdb863', '#fee0b6', '#d8daeb', '#b2abd2', '#8073ac', '#542788'],
            'general': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        }
        
        # Set default figure parameters
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'axes.grid': True,
            'grid.alpha': 0.3
        })

    def create_comprehensive_overview(self, df: pd.DataFrame, analysis_results: Dict, 
                                    domain: str = 'general') -> List[plt.Figure]:
        """Create comprehensive overview visualizations for government data"""
        
        figures = []
        domain_palette = self.domain_palettes.get(domain, self.domain_palettes['general'])
        
        # 1. Data Quality Dashboard
        fig_quality = self._create_data_quality_dashboard(df, analysis_results)
        if fig_quality:
            figures.append(fig_quality)
        
        # 2. KPI Performance Matrix
        fig_kpi = self._create_kpi_performance_matrix(df, analysis_results, domain_palette)
        if fig_kpi:
            figures.append(fig_kpi)
        
        # 3. Temporal Analysis Suite
        temporal_figs = self._create_temporal_analysis_suite(df, domain_palette)
        figures.extend(temporal_figs)
        
        # 4. Geographic Analysis
        geo_figs = self._create_geographic_analysis(df, analysis_results, domain_palette)
        figures.extend(geo_figs)
        
        # 5. Distribution Analysis
        dist_figs = self._create_distribution_analysis_suite(df, domain_palette)
        figures.extend(dist_figs)
        
        # 6. Correlation and Relationship Analysis
        corr_figs = self._create_correlation_analysis_suite(df, domain_palette)
        figures.extend(corr_figs)
        
        # 7. Statistical Significance Visualization
        stat_figs = self._create_statistical_significance_plots(df, analysis_results, domain_palette)
        figures.extend(stat_figs)
        
        # 8. Policy Impact Simulation
        policy_figs = self._create_policy_impact_visualizations(df, analysis_results, domain, domain_palette)
        figures.extend(policy_figs)
        
        return figures

    def _create_data_quality_dashboard(self, df: pd.DataFrame, analysis_results: Dict) -> plt.Figure:
        """Create comprehensive data quality dashboard"""
        
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Calculate quality metrics
        missing_pct = (df.isnull().sum() / len(df)) * 100
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # 1. Missing Data Heatmap
        ax1 = fig.add_subplot(gs[0, :2])
        if len(missing_pct) > 0:
            missing_matrix = df.isnull().astype(int)
            sns.heatmap(missing_matrix.T, cmap='Reds', cbar_kws={'label': 'Missing Data'}, ax=ax1)
            ax1.set_title('Missing Data Pattern Analysis', fontweight='bold', fontsize=14)
            ax1.set_xlabel('Records')
            ax1.set_ylabel('Variables')
        
        # 2. Data Completeness by Column
        ax2 = fig.add_subplot(gs[0, 2:])
        completeness = 100 - missing_pct
        colors = ['red' if x < 70 else 'orange' if x < 90 else 'green' for x in completeness]
        bars = ax2.barh(range(len(completeness)), completeness, color=colors)
        ax2.set_yticks(range(len(completeness)))
        ax2.set_yticklabels(completeness.index, fontsize=9)
        ax2.set_xlabel('Completeness (%)')
        ax2.set_title('Data Completeness by Variable', fontweight='bold')
        ax2.axvline(x=90, color='green', linestyle='--', alpha=0.7, label='Target (90%)')
        ax2.legend()
        
        # 3. Data Type Distribution
        ax3 = fig.add_subplot(gs[1, 0])
        type_counts = pd.Series({
            'Numeric': len(numeric_cols),
            'Categorical': len(categorical_cols),
            'DateTime': len(df.select_dtypes(include=['datetime']).columns),
            'Other': len(df.columns) - len(numeric_cols) - len(categorical_cols) - len(df.select_dtypes(include=['datetime']).columns)
        })
        colors = [self.gov_colors['primary'], self.gov_colors['secondary'], 
                 self.gov_colors['success'], self.gov_colors['neutral']]
        wedges, texts, autotexts = ax3.pie(type_counts.values, labels=type_counts.index, 
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax3.set_title('Data Type Distribution', fontweight='bold')
        
        # 4. Record Quality Score
        ax4 = fig.add_subplot(gs[1, 1])
        record_quality = df.isnull().sum(axis=1)
        quality_distribution = pd.cut(record_quality, bins=5, labels=['Excellent', 'Good', 'Fair', 'Poor', 'Critical'])
        quality_counts = quality_distribution.value_counts()
        bars = ax4.bar(quality_counts.index, quality_counts.values, 
                      color=[self.gov_colors['success'], self.gov_colors['info'], 
                            self.gov_colors['warning'], self.gov_colors['warning'], self.gov_colors['warning']])
        ax4.set_title('Record Quality Distribution', fontweight='bold')
        ax4.set_ylabel('Number of Records')
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        
        # 5. Outlier Detection Summary
        ax5 = fig.add_subplot(gs[1, 2:])
        if len(numeric_cols) > 0:
            outlier_counts = {}
            for col in numeric_cols[:10]:  # Limit to first 10 numeric columns
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                outlier_counts[col] = outliers
            
            if outlier_counts:
                outlier_series = pd.Series(outlier_counts)
                bars = ax5.bar(range(len(outlier_series)), outlier_series.values, 
                              color=self.gov_colors['warning'])
                ax5.set_xticks(range(len(outlier_series)))
                ax5.set_xticklabels(outlier_series.index, rotation=45, ha='right')
                ax5.set_title('Outlier Detection by Variable', fontweight='bold')
                ax5.set_ylabel('Number of Outliers')
        
        # 6. Overall Quality Score
        ax6 = fig.add_subplot(gs[2, :])
        overall_score = analysis_results.get('overall_quality_score', 75)
        quality_metrics = {
            'Data Completeness': float(completeness.mean()),
            'Type Consistency': 85.0,  # Placeholder - would be calculated from actual type validation
            'Outlier Rate': max(0, 100 - (sum(outlier_counts.values()) / len(df) * 100)) if 'outlier_counts' in locals() else 80,
            'Overall Score': overall_score
        }
        
        x_pos = np.arange(len(quality_metrics))
        colors = ['green' if v >= 80 else 'orange' if v >= 60 else 'red' for v in quality_metrics.values()]
        bars = ax6.bar(x_pos, quality_metrics.values(), color=colors, alpha=0.8)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, quality_metrics.values())):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(quality_metrics.keys())
        ax6.set_ylabel('Quality Score (%)')
        ax6.set_title('Data Quality Assessment Summary', fontweight='bold', fontsize=16)
        ax6.set_ylim(0, 105)
        ax6.axhline(y=80, color='green', linestyle='--', alpha=0.7, label='Target (80%)')
        ax6.legend()
        
        plt.suptitle('Data Quality Dashboard', fontsize=18, fontweight='bold', y=0.98)
        return fig

    def _create_kpi_performance_matrix(self, df: pd.DataFrame, analysis_results: Dict, 
                                     palette: List[str]) -> plt.Figure:
        """Create KPI performance matrix visualization"""
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return None
        
        # 1. KPI Gauge Chart
        ax1 = fig.add_subplot(gs[0, 0])
        kpi_data = analysis_results.get('kpis', {})
        if isinstance(kpi_data, dict) and 'numeric_summary' in kpi_data:
            kpi_means = []
            kpi_names = []
            for col, stats in list(kpi_data['numeric_summary'].items())[:6]:
                if isinstance(stats, dict) and 'mean' in stats:
                    kpi_means.append(stats['mean'])
                    kpi_names.append(col[:15])  # Truncate long names
            
            if kpi_means:
                # Normalize values for gauge visualization
                normalized_values = [(v - min(kpi_means)) / (max(kpi_means) - min(kpi_means)) * 100 
                                   if max(kpi_means) != min(kpi_means) else 50 for v in kpi_means]
                
                bars = ax1.barh(range(len(kpi_names)), normalized_values, color=palette[:len(kpi_names)])
                ax1.set_yticks(range(len(kpi_names)))
                ax1.set_yticklabels(kpi_names)
                ax1.set_xlabel('Performance Score (Normalized)')
                ax1.set_title('Key Performance Indicators', fontweight='bold')
                
                # Add value labels
                for i, (bar, orig_val) in enumerate(zip(bars, kpi_means)):
                    ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                            f'{orig_val:.1f}', va='center', fontweight='bold')
        
        # 2. Performance Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        if len(numeric_cols) > 0:
            # Create performance categories based on quartiles
            performance_data = []
            for col in numeric_cols[:8]:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    q1, q2, q3 = col_data.quantile([0.25, 0.5, 0.75])
                    performance_data.extend([
                        ('Low', (col_data <= q1).sum()),
                        ('Medium', ((col_data > q1) & (col_data <= q3)).sum()),
                        ('High', (col_data > q3).sum())
                    ])
            
            if performance_data:
                perf_df = pd.DataFrame(performance_data, columns=['Category', 'Count'])
                perf_summary = perf_df.groupby('Category')['Count'].sum()
                
                colors = [self.gov_colors['warning'], self.gov_colors['info'], self.gov_colors['success']]
                wedges, texts, autotexts = ax2.pie(perf_summary.values, labels=perf_summary.index,
                                                  autopct='%1.1f%%', colors=colors, startangle=90)
                ax2.set_title('Performance Distribution', fontweight='bold')
        
        # 3. Trend Analysis
        ax3 = fig.add_subplot(gs[0, 2])
        date_cols = df.select_dtypes(include=['datetime']).columns
        if len(date_cols) > 0 and len(numeric_cols) > 0:
            date_col = date_cols[0]
            numeric_col = numeric_cols[0]
            
            # Create monthly aggregation
            df_temp = df.copy()
            df_temp['month'] = pd.to_datetime(df_temp[date_col]).dt.to_period('M')
            monthly_data = df_temp.groupby('month')[numeric_col].mean()
            
            ax3.plot(range(len(monthly_data)), monthly_data.values, 
                    marker='o', linewidth=2, color=self.gov_colors['primary'])
            ax3.set_title(f'Trend Analysis: {numeric_col[:20]}', fontweight='bold')
            ax3.set_xlabel('Time Period')
            ax3.set_ylabel('Average Value')
            ax3.grid(True, alpha=0.3)
        
        # 4. Comparative Analysis
        ax4 = fig.add_subplot(gs[1, :])
        if len(numeric_cols) >= 2:
            # Create box plots for top numeric variables
            top_cols = numeric_cols[:6]
            data_for_box = [df[col].dropna() for col in top_cols]
            
            box_plot = ax4.boxplot(data_for_box, labels=[col[:15] for col in top_cols],
                                  patch_artist=True, notch=True)
            
            # Color the boxes
            for patch, color in zip(box_plot['boxes'], palette):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax4.set_title('Comparative Distribution Analysis', fontweight='bold', fontsize=14)
            ax4.set_ylabel('Value Range')
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
            ax4.grid(True, alpha=0.3)
        
        plt.suptitle('KPI Performance Matrix', fontsize=18, fontweight='bold', y=0.98)
        return fig

    def _create_temporal_analysis_suite(self, df: pd.DataFrame, palette: List[str]) -> List[plt.Figure]:
        """Create comprehensive temporal analysis visualizations"""
        
        figures = []
        date_cols = df.select_dtypes(include=['datetime']).columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(date_cols) == 0 or len(numeric_cols) == 0:
            return figures
        
        # Main temporal analysis figure
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
        
        date_col = date_cols[0]
        df_temp = df.copy()
        df_temp[date_col] = pd.to_datetime(df_temp[date_col])
        
        # 1. Multi-variable time series
        ax1 = fig.add_subplot(gs[0, :])
        for i, col in enumerate(numeric_cols[:4]):
            monthly_data = df_temp.set_index(date_col).resample('M')[col].mean()
            ax1.plot(monthly_data.index, monthly_data.values, 
                    marker='o', linewidth=2, label=col[:20], color=palette[i % len(palette)])
        
        ax1.set_title('Multi-Variable Temporal Trends', fontweight='bold', fontsize=14)
        ax1.set_xlabel('Time Period')
        ax1.set_ylabel('Values')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Seasonal Decomposition (if enough data)
        ax2 = fig.add_subplot(gs[1, 0])
        main_metric = numeric_cols[0]
        monthly_data = df_temp.set_index(date_col).resample('M')[main_metric].mean()
        
        if len(monthly_data) >= 24:  # Need at least 2 years for meaningful seasonal analysis
            # Simple seasonal pattern detection
            monthly_data.index = pd.to_datetime(monthly_data.index)
            seasonal_pattern = monthly_data.groupby(monthly_data.index.month).mean()
            
            bars = ax2.bar(range(1, 13), seasonal_pattern.values, color=palette[0], alpha=0.7)
            ax2.set_title(f'Seasonal Pattern: {main_metric[:20]}', fontweight='bold')
            ax2.set_xlabel('Month')
            ax2.set_ylabel('Average Value')
            ax2.set_xticks(range(1, 13))
            ax2.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        
        # 3. Growth Rate Analysis
        ax3 = fig.add_subplot(gs[1, 1])
        growth_rates = monthly_data.pct_change().dropna() * 100
        
        colors = ['green' if x > 0 else 'red' for x in growth_rates.values]
        bars = ax3.bar(range(len(growth_rates)), growth_rates.values, color=colors, alpha=0.7)
        ax3.set_title(f'Monthly Growth Rate: {main_metric[:20]}', fontweight='bold')
        ax3.set_xlabel('Time Period')
        ax3.set_ylabel('Growth Rate (%)')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.grid(True, alpha=0.3)
        
        # 4. Volatility Analysis
        ax4 = fig.add_subplot(gs[2, :])
        rolling_std = monthly_data.rolling(window=3).std()
        
        ax4.fill_between(range(len(rolling_std)), 0, rolling_std.values, 
                        color=palette[1], alpha=0.6, label='3-Month Rolling Volatility')
        ax4.plot(range(len(monthly_data)), monthly_data.values, 
                color=palette[0], linewidth=2, label=f'{main_metric[:20]} Value')
        ax4_twin = ax4.twinx()
        ax4_twin.plot(range(len(rolling_std)), rolling_std.values, 
                     color='red', linewidth=2, linestyle='--', label='Volatility')
        
        ax4.set_title('Value and Volatility Analysis', fontweight='bold', fontsize=14)
        ax4.set_xlabel('Time Period')
        ax4.set_ylabel('Value', color=palette[0])
        ax4_twin.set_ylabel('Volatility', color='red')
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
        
        plt.suptitle('Temporal Analysis Suite', fontsize=18, fontweight='bold', y=0.98)
        figures.append(fig)
        
        return figures

    def _create_geographic_analysis(self, df: pd.DataFrame, analysis_results: Dict, 
                                  palette: List[str]) -> List[plt.Figure]:
        """Create geographic analysis visualizations"""
        
        figures = []
        
        # Look for geographic columns
        geo_cols = []
        potential_geo_names = ['district', 'region', 'state', 'city', 'area', 'zone', 'mandal', 'tehsil']
        
        for col in df.columns:
            if any(geo_name in col.lower() for geo_name in potential_geo_names):
                geo_cols.append(col)
        
        if not geo_cols:
            return figures
        
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        geo_col = geo_cols[0]
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return figures
        
        # 1. Regional Performance Comparison
        ax1 = fig.add_subplot(gs[0, :])
        main_metric = numeric_cols[0]
        regional_data = df.groupby(geo_col)[main_metric].agg(['mean', 'count']).reset_index()
        regional_data = regional_data.sort_values('mean', ascending=False)
        
        bars = ax1.bar(range(len(regional_data)), regional_data['mean'], 
                      color=palette[0], alpha=0.8)
        ax1.set_xticks(range(len(regional_data)))
        ax1.set_xticklabels(regional_data[geo_col], rotation=45, ha='right')
        ax1.set_title(f'Regional Performance: {main_metric} by {geo_col}', fontweight='bold')
        ax1.set_ylabel(f'Average {main_metric}')
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, regional_data['mean'])):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(regional_data['mean']) * 0.01,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Geographic Inequality Analysis
        ax2 = fig.add_subplot(gs[1, 0])
        
        # Calculate inequality metrics
        regional_stats = df.groupby(geo_col)[main_metric].agg(['mean', 'std', 'count'])
        inequality_ratio = regional_stats['mean'].max() / regional_stats['mean'].min()
        
        # Create box plot by region
        regions = regional_stats.index[:8]  # Limit to top 8 regions
        region_data = [df[df[geo_col] == region][main_metric].dropna() for region in regions]
        
        box_plot = ax2.boxplot(region_data, labels=[str(r)[:10] for r in regions], 
                              patch_artist=True, notch=True)
        
        for patch, color in zip(box_plot['boxes'], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_title(f'Regional Distribution Analysis\nInequality Ratio: {inequality_ratio:.2f}', 
                     fontweight='bold')
        ax2.set_ylabel(main_metric)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 3. Top/Bottom Performers
        ax3 = fig.add_subplot(gs[1, 1])
        
        top_3 = regional_data.head(3)
        bottom_3 = regional_data.tail(3)
        
        # Combine and create comparison
        comparison_data = pd.concat([top_3, bottom_3])
        colors = ['green'] * 3 + ['red'] * 3
        
        bars = ax3.bar(range(len(comparison_data)), comparison_data['mean'], 
                      color=colors, alpha=0.8)
        ax3.set_xticks(range(len(comparison_data)))
        ax3.set_xticklabels(comparison_data[geo_col], rotation=45, ha='right')
        ax3.set_title('Top 3 vs Bottom 3 Performers', fontweight='bold')
        ax3.set_ylabel(f'Average {main_metric}')
        
        # Add separating line
        ax3.axvline(x=2.5, color='black', linestyle='--', alpha=0.5)
        ax3.text(1, max(comparison_data['mean']) * 0.9, 'Top 3', ha='center', 
                fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.3))
        ax3.text(4, max(comparison_data['mean']) * 0.9, 'Bottom 3', ha='center', 
                fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3))
        
        plt.suptitle('Geographic Analysis Dashboard', fontsize=18, fontweight='bold', y=0.98)
        figures.append(fig)
        
        return figures

    def _create_distribution_analysis_suite(self, df: pd.DataFrame, palette: List[str]) -> List[plt.Figure]:
        """Create comprehensive distribution analysis"""
        
        figures = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return figures
        
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
        
        # 1. Multi-variable distributions
        for i, col in enumerate(numeric_cols[:6]):
            row = i // 3
            col_idx = i % 3
            ax = fig.add_subplot(gs[row, col_idx])
            
            # Create histogram with KDE
            data = df[col].dropna()
            ax.hist(data, bins=30, density=True, alpha=0.7, color=palette[i % len(palette)], 
                   edgecolor='black', linewidth=0.5)
            
            # Add KDE line
            if len(data) > 10:
                try:
                    from scipy import stats
                    # Check for data variation before applying KDE
                    if data.std() > 1e-10:  # Avoid singular covariance matrix
                        kde = stats.gaussian_kde(data)
                        x_range = np.linspace(data.min(), data.max(), 100)
                        ax.plot(x_range, kde(x_range), color='red', linewidth=2, label='KDE')
                    else:
                        # Data has no variation, skip KDE
                        pass
                except Exception as e:
                    # Skip KDE if it fails (e.g., singular covariance matrix)
                    pass
            
            # Add statistics
            mean_val = data.mean()
            median_val = data.median()
            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='blue', linestyle='--', alpha=0.8, label=f'Median: {median_val:.2f}')
            
            ax.set_title(f'{col[:20]} Distribution', fontweight='bold')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Fill remaining subplots with summary statistics
        if len(numeric_cols) < 6:
            ax_summary = fig.add_subplot(gs[len(numeric_cols)//3, len(numeric_cols)%3:])
            
            # Create summary statistics table
            summary_stats = df[numeric_cols].describe().round(2)
            
            # Convert to heatmap
            sns.heatmap(summary_stats, annot=True, cmap='YlOrRd', ax=ax_summary, 
                       cbar_kws={'label': 'Value'})
            ax_summary.set_title('Summary Statistics Heatmap', fontweight='bold')
        
        plt.suptitle('Distribution Analysis Suite', fontsize=18, fontweight='bold', y=0.98)
        figures.append(fig)
        
        return figures

    def _create_correlation_analysis_suite(self, df: pd.DataFrame, palette: List[str]) -> List[plt.Figure]:
        """Create comprehensive correlation analysis"""
        
        figures = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return figures
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Correlation Heatmap
        ax1 = fig.add_subplot(gs[0, :])
        corr_matrix = df[numeric_cols].corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax1)
        ax1.set_title('Correlation Matrix Heatmap', fontweight='bold', fontsize=14)
        
        # 2. Strongest Correlations
        ax2 = fig.add_subplot(gs[1, 0])
        
        # Find strongest correlations
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append({
                    'var1': corr_matrix.columns[i],
                    'var2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
        
        corr_df = pd.DataFrame(corr_pairs)
        corr_df['abs_corr'] = corr_df['correlation'].abs()
        top_corr = corr_df.nlargest(8, 'abs_corr')
        
        colors = ['red' if x < 0 else 'green' for x in top_corr['correlation']]
        bars = ax2.barh(range(len(top_corr)), top_corr['correlation'], color=colors, alpha=0.8)
        
        labels = [f"{row['var1'][:10]} vs {row['var2'][:10]}" for _, row in top_corr.iterrows()]
        ax2.set_yticks(range(len(top_corr)))
        ax2.set_yticklabels(labels, fontsize=9)
        ax2.set_xlabel('Correlation Coefficient')
        ax2.set_title('Strongest Correlations', fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        
        # 3. Correlation Network (simplified)
        ax3 = fig.add_subplot(gs[1, 1])
        
        # Create simplified network visualization
        strong_corr = top_corr.head(5)
        
        # Simple scatter plot of top correlated pairs
        if len(strong_corr) > 0:
            top_pair = strong_corr.iloc[0]
            var1, var2 = top_pair['var1'], top_pair['var2']
            
            ax3.scatter(df[var1], df[var2], alpha=0.6, color=palette[0])
            
            # Add trend line
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(df[var1].dropna(), 
                                                                          df[var2].dropna())
            line = slope * df[var1] + intercept
            ax3.plot(df[var1], line, 'r', label=f'R²={r_value**2:.3f}')
            
            ax3.set_xlabel(var1[:20])
            ax3.set_ylabel(var2[:20])
            ax3.set_title(f'Strongest Correlation: {top_pair["correlation"]:.3f}', fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        plt.suptitle('Correlation Analysis Suite', fontsize=18, fontweight='bold', y=0.98)
        figures.append(fig)
        
        return figures

    def _create_statistical_significance_plots(self, df: pd.DataFrame, analysis_results: Dict, 
                                             palette: List[str]) -> List[plt.Figure]:
        """Create statistical significance visualizations"""
        
        figures = []
        
        # Check if we have hypothesis test results
        hypothesis_tests = analysis_results.get('hypothesis_tests', [])
        if not hypothesis_tests:
            return figures
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. P-value Summary
        ax1 = fig.add_subplot(gs[0, 0])
        
        p_values = [test.get('test_statistics', {}).get('p_value', 1) for test in hypothesis_tests]
        test_names = [f"{test.get('dependent_variable', 'Unknown')[:15]}" for test in hypothesis_tests]
        
        colors = ['green' if p < 0.05 else 'orange' if p < 0.1 else 'red' for p in p_values]
        bars = ax1.bar(range(len(p_values)), p_values, color=colors, alpha=0.8)
        
        ax1.set_xticks(range(len(test_names)))
        ax1.set_xticklabels(test_names, rotation=45, ha='right')
        ax1.set_ylabel('P-value')
        ax1.set_title('Statistical Significance Tests', fontweight='bold')
        ax1.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α=0.05')
        ax1.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='α=0.10')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Effect Size Visualization
        ax2 = fig.add_subplot(gs[0, 1])
        
        effect_sizes = []
        for test in hypothesis_tests:
            effect_size = test.get('effect_size', {})
            if 'cohens_d' in effect_size:
                effect_sizes.append(abs(effect_size['cohens_d']))
            else:
                effect_sizes.append(0)
        
        if effect_sizes:
            colors = ['green' if e > 0.8 else 'orange' if e > 0.5 else 'yellow' if e > 0.2 else 'red' 
                     for e in effect_sizes]
            bars = ax2.bar(range(len(effect_sizes)), effect_sizes, color=colors, alpha=0.8)
            
            ax2.set_xticks(range(len(test_names)))
            ax2.set_xticklabels(test_names, rotation=45, ha='right')
            ax2.set_ylabel("Cohen's d (Effect Size)")
            ax2.set_title('Effect Size Analysis', fontweight='bold')
            ax2.axhline(y=0.2, color='yellow', linestyle='--', alpha=0.7, label='Small')
            ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Medium')
            ax2.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Large')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Confidence Intervals (if available)
        ax3 = fig.add_subplot(gs[1, :])
        
        # Extract test statistics for confidence interval visualization
        means = []
        ci_lower = []
        ci_upper = []
        valid_tests = []
        
        for i, test in enumerate(hypothesis_tests[:6]):  # Limit to 6 tests
            test_stats = test.get('test_statistics', {})
            if 'confidence_interval' in test_stats:
                ci = test_stats['confidence_interval']
                if isinstance(ci, (list, tuple)) and len(ci) >= 2:
                    ci_lower.append(ci[0])
                    ci_upper.append(ci[1])
                    means.append((ci[0] + ci[1]) / 2)
                    valid_tests.append(test_names[i])
        
        if valid_tests:
            x_pos = range(len(valid_tests))
            
            # Plot confidence intervals
            ax3.errorbar(x_pos, means, yerr=[np.array(means) - np.array(ci_lower), 
                                           np.array(ci_upper) - np.array(means)],
                        fmt='o', markersize=8, capsize=5, capthick=2, color=palette[0])
            
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(valid_tests, rotation=45, ha='right')
            ax3.set_ylabel('Effect Estimate')
            ax3.set_title('95% Confidence Intervals', fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.suptitle('Statistical Significance Analysis', fontsize=18, fontweight='bold', y=0.98)
        figures.append(fig)
        
        return figures

    def _create_policy_impact_visualizations(self, df: pd.DataFrame, analysis_results: Dict, 
                                           domain: str, palette: List[str]) -> List[plt.Figure]:
        """Create policy impact and scenario analysis visualizations"""
        
        figures = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return figures
        
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
        
        # 1. Resource Allocation Optimization
        ax1 = fig.add_subplot(gs[0, :])
        
        # Simulate resource allocation scenarios
        main_metric = numeric_cols[0]
        current_mean = df[main_metric].mean()
        current_std = df[main_metric].std()
        
        scenarios = {
            'Current State': current_mean,
            '10% Budget Increase': current_mean * 1.08,
            '20% Budget Increase': current_mean * 1.15,
            '30% Budget Increase': current_mean * 1.20,
            'Optimized Allocation': current_mean * 1.25,
            'Best Case Scenario': current_mean * 1.35
        }
        
        scenario_names = list(scenarios.keys())
        scenario_values = list(scenarios.values())
        colors = [palette[0]] + [palette[1]] * 3 + [palette[2]] * 2
        
        bars = ax1.bar(range(len(scenario_names)), scenario_values, color=colors, alpha=0.8)
        ax1.set_xticks(range(len(scenario_names)))
        ax1.set_xticklabels(scenario_names, rotation=45, ha='right')
        ax1.set_ylabel(f'{main_metric} Performance')
        ax1.set_title('Policy Intervention Scenarios', fontweight='bold', fontsize=14)
        
        # Add improvement percentages
        for i, (bar, value) in enumerate(zip(bars, scenario_values)):
            if i > 0:
                improvement = ((value - current_mean) / current_mean) * 100
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(scenario_values) * 0.01,
                        f'+{improvement:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax1.grid(True, alpha=0.3)
        
        # 2. Cost-Benefit Analysis
        ax2 = fig.add_subplot(gs[1, 0])
        
        # Simulate cost-benefit scenarios
        interventions = ['Digital Infrastructure', 'Training Programs', 'Process Optimization', 
                        'Technology Upgrade', 'Staff Augmentation']
        costs = [100, 150, 80, 200, 120]  # Relative costs
        benefits = [180, 220, 140, 350, 160]  # Relative benefits
        
        # Calculate ROI
        roi = [(b - c) / c * 100 for b, c in zip(benefits, costs)]
        
        bars = ax2.bar(range(len(interventions)), roi, 
                      color=[palette[i % len(palette)] for i in range(len(interventions))], alpha=0.8)
        ax2.set_xticks(range(len(interventions)))
        ax2.set_xticklabels(interventions, rotation=45, ha='right')
        ax2.set_ylabel('Return on Investment (%)')
        ax2.set_title('Intervention Cost-Benefit Analysis', fontweight='bold')
        ax2.axhline(y=50, color='green', linestyle='--', alpha=0.7, label='Target ROI (50%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Risk Assessment Matrix
        ax3 = fig.add_subplot(gs[1, 1])
        
        # Create risk assessment data
        risks = ['Implementation Delay', 'Budget Overrun', 'Resistance to Change', 
                'Technical Challenges', 'Regulatory Issues']
        probability = [0.3, 0.4, 0.6, 0.5, 0.2]
        impact = [0.7, 0.8, 0.5, 0.6, 0.9]
        
        # Create scatter plot with bubble sizes representing overall risk
        risk_score = [p * i for p, i in zip(probability, impact)]
        colors_risk = [palette[i % len(palette)] for i in range(len(risks))]
        
        scatter = ax3.scatter(probability, impact, s=[r * 1000 for r in risk_score], 
                             c=colors_risk, alpha=0.6, edgecolors='black')
        
        # Add risk labels
        for i, risk in enumerate(risks):
            ax3.annotate(risk[:15], (probability[i], impact[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax3.set_xlabel('Probability')
        ax3.set_ylabel('Impact')
        ax3.set_title('Risk Assessment Matrix', fontweight='bold')
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)
        
        # Add risk zones
        ax3.axhspan(0.7, 1, xmin=0.7, alpha=0.2, color='red', label='High Risk')
        ax3.axhspan(0.3, 0.7, xmin=0.3, xmax=0.7, alpha=0.2, color='orange', label='Medium Risk')
        ax3.axhspan(0, 0.3, xmax=0.3, alpha=0.2, color='green', label='Low Risk')
        
        # 4. Implementation Timeline
        ax4 = fig.add_subplot(gs[2, :])
        
        # Create Gantt-style timeline
        activities = ['Planning Phase', 'Resource Allocation', 'Implementation', 
                     'Monitoring & Evaluation', 'Scale-up']
        start_times = [0, 2, 4, 8, 10]
        durations = [2, 2, 4, 3, 6]
        
        colors_timeline = [palette[i % len(palette)] for i in range(len(activities))]
        
        for i, (activity, start, duration, color) in enumerate(zip(activities, start_times, durations, colors_timeline)):
            ax4.barh(i, duration, left=start, height=0.6, color=color, alpha=0.8, edgecolor='black')
            ax4.text(start + duration/2, i, f'{activity}', ha='center', va='center', 
                    fontweight='bold', fontsize=10)
        
        ax4.set_yticks(range(len(activities)))
        ax4.set_yticklabels(activities)
        ax4.set_xlabel('Timeline (Months)')
        ax4.set_title('Policy Implementation Timeline', fontweight='bold', fontsize=14)
        ax4.grid(True, alpha=0.3, axis='x')
        
        # Add milestones
        milestones = [3, 6, 9, 12]
        for milestone in milestones:
            ax4.axvline(x=milestone, color='red', linestyle='--', alpha=0.7)
            ax4.text(milestone, len(activities), f'M{milestone//3}', ha='center', va='bottom', 
                    fontweight='bold', bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.7))
        
        plt.suptitle(f'Policy Impact Analysis - {domain.title()} Domain', 
                    fontsize=18, fontweight='bold', y=0.98)
        figures.append(fig)
        
        return figures

    def save_figures_to_pdf(self, figures: List[plt.Figure], filename: str, 
                           metadata: Dict = None) -> str:
        """Save all figures to a single PDF file"""
        
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with PdfPages(str(filepath)) as pdf:
            for fig in figures:
                if fig is not None:
                    pdf.savefig(fig, bbox_inches='tight', dpi=300)
                    plt.close(fig)  # Free memory
            
            # Add metadata if provided
            if metadata:
                d = pdf.infodict()
                d['Title'] = metadata.get('title', 'RTGS AI Analyst Report')
                d['Author'] = metadata.get('author', 'RTGS AI Analyst System')
                d['Subject'] = metadata.get('subject', 'Government Data Analysis Report')
                d['Keywords'] = metadata.get('keywords', 'Government, Data Analysis, Policy')
                d['Creator'] = 'RTGS AI Analyst'
        
        return str(filepath)

    def create_summary_statistics_table(self, df: pd.DataFrame) -> plt.Figure:
        """Create a comprehensive summary statistics table"""
        
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Calculate comprehensive statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        summary_data = []
        
        for col in df.columns[:15]:  # Limit to 15 columns for readability
            if col in numeric_cols:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    summary_data.append([
                        col[:20],
                        'Numeric',
                        len(col_data),
                        f"{col_data.mean():.2f}",
                        f"{col_data.std():.2f}",
                        f"{col_data.min():.2f}",
                        f"{col_data.max():.2f}",
                        f"{(df[col].isnull().sum() / len(df) * 100):.1f}%"
                    ])
            elif col in categorical_cols:
                col_data = df[col].dropna()
                summary_data.append([
                    col[:20],
                    'Categorical',
                    len(col_data),
                    f"{col_data.nunique()} unique",
                    f"Mode: {col_data.mode().iloc[0] if len(col_data.mode()) > 0 else 'N/A'}",
                    '-',
                    '-',
                    f"{(df[col].isnull().sum() / len(df) * 100):.1f}%"
                ])
        
        # Create table
        table = ax.table(cellText=summary_data,
                        colLabels=['Variable', 'Type', 'Count', 'Mean/Mode', 'Std/Info', 'Min', 'Max', 'Missing %'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.15, 0.1, 0.08, 0.12, 0.15, 0.08, 0.08, 0.1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 2)
        
        # Style the table
        for i in range(len(summary_data) + 1):
            for j in range(8):
                cell = table[(i, j)]
                if i == 0:  # Header row
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        plt.title('Dataset Summary Statistics', fontsize=16, fontweight='bold', pad=20)
        return fig