#!/usr/bin/env python3
"""
Generate proper PDF reports for RTGS AI Analyst
"""
import json
import os
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

def load_analysis_data():
    """Load analysis results and insights"""
    with open('artifacts/docs/analysis_results.json', 'r') as f:
        analysis_results = json.load(f)
    
    with open('artifacts/docs/insights_executive.json', 'r') as f:
        insights = json.load(f)
    
    return analysis_results, insights

def create_technical_pdf():
    """Create technical PDF report"""
    analysis_results, insights = load_analysis_data()
    
    # Create PDF
    filename = f"artifacts/reports/Technical_Report_agricultural_2019_5_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    doc = SimpleDocTemplate(filename, pagesize=A4)
    story = []
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        textColor=colors.darkblue
    )
    
    # Title
    story.append(Paragraph("RTGS AI ANALYST - TECHNICAL REPORT", title_style))
    story.append(Paragraph(f"Agricultural Dataset Analysis - {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Executive Summary
    story.append(Paragraph("EXECUTIVE SUMMARY", heading_style))
    story.append(Paragraph(insights['executive_summary']['one_line_summary'], styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Dataset Information
    story.append(Paragraph("DATASET INFORMATION", heading_style))
    dataset_info = analysis_results['dataset_profile']['basic_info']
    dataset_table = Table([
        ['Metric', 'Value'],
        ['Total Rows', f"{dataset_info['rows']:,}"],
        ['Total Columns', str(dataset_info['columns'])],
        ['Memory Usage', f"{dataset_info['memory_usage_mb']:.2f} MB"],
        ['Duplicate Rows', str(dataset_info['duplicate_rows'])],
        ['Missing Data', f"{analysis_results['dataset_profile']['missing_data']['missing_percentage']:.2%}"]
    ])
    dataset_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(dataset_table)
    story.append(Spacer(1, 20))
    
    # Key Findings
    story.append(Paragraph("KEY FINDINGS", heading_style))
    for i, finding in enumerate(insights['key_findings'], 1):
        story.append(Paragraph(f"<b>Finding {i}:</b> {finding['finding']}", styles['Normal']))
        story.append(Paragraph(f"<b>Evidence:</b> {finding['evidence']}", styles['Normal']))
        story.append(Paragraph(f"<b>Confidence:</b> {finding['confidence']}", styles['Normal']))
        story.append(Spacer(1, 10))
    
    # Statistical Analysis
    story.append(PageBreak())
    story.append(Paragraph("STATISTICAL ANALYSIS", heading_style))
    
    # Column Analysis
    story.append(Paragraph("Column Analysis", styles['Heading3']))
    
    # Get column information from KPIs
    if 'kpis' in analysis_results and 'numeric_summary' in analysis_results['kpis']:
        col_data = [['Column', 'Mean', 'Std Dev', 'Min', 'Max']]
        
        for col_name, col_info in analysis_results['kpis']['numeric_summary'].items():
            if isinstance(col_info, dict) and 'mean' in col_info:
                col_data.append([
                    col_name,
                    f"{col_info.get('mean', 0):.2f}",
                    f"{col_info.get('std', 0):.2f}",
                    f"{col_info.get('min', 0):.2f}",
                    f"{col_info.get('max', 0):.2f}"
                ])
        
        if len(col_data) > 1:
            col_table = Table(col_data)
            col_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(col_table)
    
    story.append(Spacer(1, 20))
    
    # Methodology
    story.append(Paragraph("METHODOLOGY", heading_style))
    story.append(Paragraph("This analysis was conducted using the RTGS AI Analyst system, which employs:", styles['Normal']))
    story.append(Paragraph("• Automated data profiling and quality assessment", styles['Normal']))
    story.append(Paragraph("• Statistical analysis with significance testing", styles['Normal']))
    story.append(Paragraph("• AI-powered pattern recognition and insight generation", styles['Normal']))
    story.append(Paragraph("• Policy recommendation framework", styles['Normal']))
    
    # Build PDF
    doc.build(story)
    print(f"✅ Technical PDF created: {filename}")
    return filename

def create_government_pdf():
    """Create government officials PDF report"""
    analysis_results, insights = load_analysis_data()
    
    # Create PDF
    filename = f"artifacts/reports/Government_Officials_Report_agricultural_2019_5_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    doc = SimpleDocTemplate(filename, pagesize=A4)
    story = []
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkred
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.darkred
    )
    
    # Title
    story.append(Paragraph("GOVERNMENT POLICY BRIEF", title_style))
    story.append(Paragraph("Agricultural Services Analysis & Recommendations", styles['Heading2']))
    story.append(Paragraph(f"Prepared for Government Officials - {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Executive Summary
    story.append(Paragraph("EXECUTIVE SUMMARY", heading_style))
    story.append(Paragraph(insights['executive_summary']['one_line_summary'], styles['Normal']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(insights['executive_summary']['key_insights_summary'], styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Priority Actions
    story.append(Paragraph("IMMEDIATE PRIORITY ACTIONS", heading_style))
    story.append(Paragraph(insights['executive_summary']['priority_actions'], styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Key Findings
    story.append(Paragraph("KEY FINDINGS", heading_style))
    for i, finding in enumerate(insights['key_findings'], 1):
        story.append(Paragraph(f"<b>{i}. {finding['finding']}</b>", styles['Normal']))
        story.append(Paragraph(f"Evidence: {finding['evidence']}", styles['Normal']))
        story.append(Paragraph(f"Policy Relevance: {finding['policy_relevance']}", styles['Normal']))
        story.append(Spacer(1, 12))
    
    # Policy Recommendations
    story.append(PageBreak())
    story.append(Paragraph("POLICY RECOMMENDATIONS", heading_style))
    
    for i, rec in enumerate(insights['policy_recommendations'], 1):
        story.append(Paragraph(f"<b>Recommendation {i}: {rec['recommendation']}</b>", styles['Heading3']))
        story.append(Paragraph(f"Priority: {rec['priority']} | Cost: {rec['estimated_cost']} | Timeframe: {rec['timeframe']}", styles['Normal']))
        story.append(Paragraph(f"Responsible Agency: {rec['responsible_agency']}", styles['Normal']))
        story.append(Paragraph(f"Expected Impact: {rec['estimated_impact']}", styles['Normal']))
        
        story.append(Paragraph("Implementation Steps:", styles['Normal']))
        for step in rec['implementation_steps']:
            story.append(Paragraph(f"• {step}", styles['Normal']))
        
        story.append(Paragraph("Success Metrics:", styles['Normal']))
        for metric in rec['success_metrics']:
            story.append(Paragraph(f"• {metric}", styles['Normal']))
        
        story.append(Spacer(1, 15))
    
    # Data Quality Assessment
    story.append(Paragraph("DATA QUALITY ASSESSMENT", heading_style))
    story.append(Paragraph(f"Overall Data Quality: {insights['executive_summary']['overall_assessment']}", styles['Normal']))
    story.append(Paragraph(f"Analysis Confidence: {insights['confidence_assessment']['overall_confidence']}", styles['Normal']))
    story.append(Paragraph(f"High Confidence Findings: {insights['confidence_assessment']['high_confidence_percentage']:.0f}%", styles['Normal']))
    
    # Build PDF
    doc.build(story)
    print(f"✅ Government Officials PDF created: {filename}")
    return filename

if __name__ == "__main__":
    print("🚀 Generating PDF Reports...")
    
    try:
        tech_pdf = create_technical_pdf()
        gov_pdf = create_government_pdf()
        
        print(f"\n✅ Reports Generated Successfully!")
        print(f"📊 Technical Report: {tech_pdf}")
        print(f"🏛️ Government Report: {gov_pdf}")
        
    except Exception as e:
        print(f"❌ Error generating reports: {str(e)}")
        import traceback
        traceback.print_exc()
