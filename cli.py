#!/usr/bin/env python3
"""
RTGS AI Analyst - Enhanced Command Line Interface
Multi-agent pipeline with LLM-powered analysis for government data insights
"""

import argparse
import asyncio
import json
import yaml
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import uuid
import pandas as pd
import os


# Import LangGraph orchestrator and enhanced components
from src.orchestrator.agent_router import RTGSOrchestrator
from src.agents.report_agent import EnhancedReportAgent,LLMAnalysisEngine
from src.utils.logging import setup_logging, get_logger


class RTGSCLIError(Exception):
    """Custom exception for CLI errors"""
    pass


class EnhancedRTGSCLI:
    """Enhanced RTGS AI Analyst CLI with LLM-powered analysis and LangGraph orchestration"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.orchestrator = None
        self.llm_engine = LLMAnalysisEngine()
        self.enhanced_report_agent = None

    def create_run_manifest(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Create enhanced run manifest with LLM analysis metadata"""
        run_id = f"rtgs-enhanced-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{str(uuid.uuid4())[:8]}"
        
        # Parse dataset context if provided
        dataset_context = {}
        if args.context_file and Path(args.context_file).exists():
            with open(args.context_file, 'r') as f:
                dataset_context = yaml.safe_load(f)
        
        manifest = {
            "run_id": run_id,
            "timestamp_utc": datetime.utcnow().isoformat(),
            "dataset_info": {
                "source_path": str(Path(args.dataset).resolve()),
                "dataset_name": Path(args.dataset).stem,
                "domain_hint": args.domain or dataset_context.get('domain', 'auto'),
                "scope": args.scope or dataset_context.get('scope', 'Regional Analysis'),
                "description": dataset_context.get('description', 'Government dataset for AI-enhanced analysis')
            },
            "user_context": {
                "business_questions": dataset_context.get('business_questions', []),
                "key_metrics": dataset_context.get('key_metrics', []),
                "stakeholders": dataset_context.get('stakeholders', 'Government officials'),
                "time_scope": dataset_context.get('time_scope', 'Not specified'),
                "geo_scope": dataset_context.get('geo_scope', 'Not specified')
            },
            "run_config": {
                "mode": args.mode,
                "sample_rows": args.sample_rows,
                "auto_approve": args.auto_approve,
                "output_dir": str(Path(args.output_dir).resolve()),
                "report_format": args.report_format,
                "llm_enhanced": True,
                "enable_domain_detection": args.domain == 'auto' or args.domain is None
            },
            "agent_version_tags": {
                "orchestrator": "v1.0",
                "ingestion": "v1.0", 
                "schema": "v1.0",
                "cleaning": "v1.0",
                "analysis": "v1.0",
                "insight": "v1.0",
                "enhanced_report": "v2.0-llm",
                "llm_engine": "claude-sonnet-4"
            },
            "llm_analysis": {
                "enabled": True,
                "domain_detection": "auto" if args.domain == 'auto' or args.domain is None else "manual",
                "analysis_depth": "comprehensive",
                "policy_focus": True
            },
            "artifacts_paths": {},
            "confidence_overall": "PENDING",
            "notes": []
        }
        
        return manifest

    def setup_output_directories(self, run_manifest: Dict[str, Any]) -> None:
        """Create comprehensive output directory structure"""
        base_dir = Path(run_manifest["run_config"]["output_dir"])
        run_id = run_manifest["run_id"]
        
        directories = [
            base_dir / "artifacts" / "logs",
            base_dir / "artifacts" / "reports", 
            base_dir / "artifacts" / "plots" / "interactive",
            base_dir / "artifacts" / "plots" / "static",
            base_dir / "artifacts" / "docs",
            base_dir / "artifacts" / "quick_start",
            base_dir / "artifacts" / "llm_analysis",
            base_dir / "data" / "raw",
            base_dir / "data" / "standardized", 
            base_dir / "data" / "cleaned",
            base_dir / "data" / "transformed"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Update manifest with artifact paths
        artifacts_base = base_dir / "artifacts"
        data_base = base_dir / "data"
        run_manifest["artifacts_paths"] = {
            "logs_dir": str(artifacts_base / "logs"),
            "reports_dir": str(artifacts_base / "reports"),
            "plots_dir": str(artifacts_base / "plots"),
            "docs_dir": str(artifacts_base / "docs"),
            "quick_start_dir": str(artifacts_base / "quick_start"),
            "llm_analysis_dir": str(artifacts_base / "llm_analysis"),
            "data_dir": str(data_base),
            "run_manifest": str(artifacts_base / "docs" / "run_manifest.json")
        }

    def interactive_context_collection(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Enhanced interactive context collection with LLM capabilities"""
        print("\n🎯 RTGS AI ANALYST - Enhanced Dataset Context Collection")
        print("=" * 70)
        print("🧠 This system uses advanced AI to provide intelligent analysis")
        
        context = {}
        
        # Domain with auto-detection option
        if not args.domain or args.domain == 'auto':
            print("\n📋 What domain does this dataset belong to?")
            print("Options: transport, health, education, economics, agriculture, environment, urban, social, auto")
            print("💡 Choose 'auto' for AI-powered domain detection")
            domain_input = input("Domain [auto]: ").strip() or "auto"
            context['domain'] = domain_input
        else:
            context['domain'] = args.domain
            
        # Enhanced description
        print(f"\n📝 Brief description of the dataset and analysis goals:")
        context['description'] = input("Description: ").strip() or "Government dataset for AI-enhanced policy analysis"
        
        # Business questions with AI enhancement note
        print(f"\n❓ What are the key policy questions you want answered?")
        print("💡 Our AI will generate additional insights beyond these questions")
        print("(Enter one per line, empty line to finish)")
        business_questions = []
        while True:
            question = input("Question: ").strip()
            if not question:
                break
            business_questions.append(question)
        context['business_questions'] = business_questions or ["What are the key trends and policy implications?"]
        
        # Key metrics with intelligence note
        print(f"\n📊 What are the key metrics/columns of interest? (comma-separated)")
        print("💡 AI will automatically identify additional relevant metrics")
        metrics_input = input("Key metrics: ").strip()
        context['key_metrics'] = [m.strip() for m in metrics_input.split(",")] if metrics_input else []
        
        # Enhanced scope
        if not args.scope:
            print(f"\n🌍 What is the geographic and temporal scope?")
            print("Examples: 'Telangana State 2020-2024', 'Hyderabad District 2023', 'All districts current year'")
            context['scope'] = input("Scope: ").strip() or "Regional Analysis"
        else:
            context['scope'] = args.scope
            
        # Stakeholders with AI recommendation note
        print(f"\n👥 Who are the primary stakeholders/audience?")
        print("💡 AI will tailor recommendations for this audience")
        context['stakeholders'] = input("Stakeholders [Government officials]: ").strip() or "Government officials"
        
        # Analysis preferences
        print(f"\n⚙️ Analysis preferences:")
        print("1. Focus on immediate actionable insights")
        print("2. Comprehensive long-term strategic analysis") 
        print("3. Balanced approach (recommended)")
        analysis_focus = input("Choice [3]: ").strip() or "3"
        context['analysis_focus'] = {
            "1": "immediate",
            "2": "strategic", 
            "3": "balanced"
        }.get(analysis_focus, "balanced")
        
        return context

    async def detect_domain_with_ai(self, dataset_path: str) -> str:
        """Use AI to intelligently detect domain from dataset"""
        print(f"\n🧠 AI Domain Detection: Analyzing dataset structure...")
        
        try:
            # Load sample for analysis
            df = pd.read_csv(dataset_path, nrows=100)
            
            # Prepare context for AI analysis
            column_info = {
                'columns': list(df.columns),
                'sample_data': {}
            }
            
            for col in df.columns[:10]:
                if df[col].dtype == 'object':
                    unique_vals = df[col].dropna().unique()[:5]
                    column_info['sample_data'][col] = [str(val) for val in unique_vals]
                else:
                    stats = {
                        'min': float(df[col].min()) if pd.notnull(df[col].min()) else None,
                        'max': float(df[col].max()) if pd.notnull(df[col].max()) else None,
                        'mean': float(df[col].mean()) if pd.notnull(df[col].mean()) else None
                    }
                    column_info['sample_data'][col] = [f"Numeric: {stats}"]
            
            prompt = f"""You are an expert government data analyst. Analyze this dataset and determine the most likely domain.

DATASET ANALYSIS:
- Filename: {Path(dataset_path).name}
- Columns: {column_info['columns']}
- Sample Data: {json.dumps(column_info['sample_data'], indent=2)}

DOMAIN OPTIONS:
- health: Healthcare services, medical data, public health metrics
- education: Schools, students, learning outcomes, educational infrastructure
- transport: Roads, vehicles, traffic, public transportation, connectivity
- economics: Employment, business, GDP, economic indicators, finance
- agriculture: Farming, crops, livestock, rural development, food security
- environment: Pollution, climate, natural resources, sustainability
- urban: City planning, utilities, municipal services, urban development
- social: Demographics, welfare, social services, community programs

Based on the column names and data patterns, respond with just the domain name (one word) that best matches this dataset.

Domain:"""
            
            result = await self.llm_engine.call_llm(prompt, 100)
            detected_domain = result.strip().lower()
            
            valid_domains = ['health', 'education', 'transport', 'economics', 
                           'agriculture', 'environment', 'urban', 'social']
            
            if detected_domain in valid_domains:
                print(f"   ✅ Detected Domain: {detected_domain.title()}")
                return detected_domain
            else:
                print(f"   ⚠️ Could not determine domain, using 'general'")
                return 'general'
                
        except Exception as e:
            self.logger.warning(f"AI domain detection failed: {e}")
            print(f"   ⚠️ Domain detection failed, using 'general'")
            return 'general'

    async def run_enhanced_pipeline(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Execute the enhanced RTGS pipeline with LLM analysis"""
        try:
            print(f"\n🚀 RTGS AI ANALYST - Enhanced Pipeline")
            print("=" * 60)
            print("🧠 Powered by Claude Sonnet 4 for intelligent analysis")
            
            # Collect context interactively if needed
            if args.interactive:
                dataset_context = self.interactive_context_collection(args)
                # Save context for future use
                context_file = f"context_{Path(args.dataset).stem}.yaml"
                with open(context_file, 'w') as f:
                    yaml.dump(dataset_context, f, default_flow_style=False)
                print(f"\n💾 Context saved to {context_file} for future runs")
                args.context_file = context_file
                
                # Update args with collected context
                if 'domain' in dataset_context:
                    args.domain = dataset_context['domain']
                if 'scope' in dataset_context:
                    args.scope = dataset_context['scope']

            # AI Domain Detection if needed
            if args.domain == 'auto' or args.domain is None:
                detected_domain = await self.detect_domain_with_ai(args.dataset)
                args.domain = detected_domain

            # Create enhanced run manifest
            run_manifest = self.create_run_manifest(args)
            
            # Setup output directories
            self.setup_output_directories(run_manifest)
            
            # Save initial manifest
            manifest_path = run_manifest["artifacts_paths"]["run_manifest"]
            with open(manifest_path, 'w') as f:
                json.dump(run_manifest, f, indent=2)
            
            print(f"\n📋 Pipeline Configuration:")
            print(f"   🆔 Run ID: {run_manifest['run_id']}")
            print(f"   📁 Dataset: {run_manifest['dataset_info']['dataset_name']}")
            print(f"   🏷️ Domain: {run_manifest['dataset_info']['domain_hint']}")
            print(f"   📊 Mode: {run_manifest['run_config']['mode']}")
            print(f"   🧠 LLM Enhanced: ✅")
            print(f"   📁 Outputs: {run_manifest['run_config']['output_dir']}")
            
            # Phase 1: Traditional Pipeline with LangGraph
            print(f"\n🔄 Phase 1: Data Processing Pipeline (LangGraph)")
            print("   ⚙️ Initializing multi-agent orchestrator...")
            
            self.orchestrator = RTGSOrchestrator(run_manifest)
            
            # Execute pipeline based on mode
            if args.mode == "dry-run":
                print("   🔍 Executing dry run (schema inference only)...")
                result = await self.orchestrator.dry_run()
                print("   ✅ Dry run completed - schema analysis ready")
                
            elif args.mode == "preview":
                print("   👀 Executing preview mode...")
                result = await self.orchestrator.preview_run()
                print("   ✅ Preview completed - review transformations")
                
                if not args.auto_approve:
                    approval = input("\n🤔 Continue with full analysis? [y/N]: ").strip().lower()
                    if approval not in ['y', 'yes']:
                        print("❌ Pipeline cancelled by user")
                        return run_manifest
                        
                # Continue with full run after approval
                print("   🚀 Continuing with full analysis...")
                result = await self.orchestrator.full_run()
                
            else:  # full run
                print("   🚀 Executing full pipeline...")
                result = await self.orchestrator.full_run()
            
            # Update manifest with traditional pipeline results
            run_manifest.update(result)
            
            # Phase 2: LLM-Enhanced Analysis
            print(f"\n🧠 Phase 2: LLM-Enhanced Analysis (Claude Sonnet 4)")
            
            # Initialize enhanced report agent
            self.enhanced_report_agent = EnhancedReportAgent()
            
            # Create state object for enhanced analysis
            class EnhancedState:
                def __init__(self, manifest, traditional_results):
                    self.run_manifest = manifest
                    
                    # Load processed data
                    try:
                        cleaned_data_path = Path(manifest['artifacts_paths']['data_dir']) / 'cleaned' / f"{manifest['dataset_info']['dataset_name']}_cleaned.csv"
                        if cleaned_data_path.exists():
                            self.transformed_data = pd.read_csv(cleaned_data_path)
                        else:
                            # Fallback to original data
                            self.transformed_data = pd.read_csv(manifest['dataset_info']['source_path'])
                    except:
                        self.transformed_data = pd.DataFrame()
                    
                    # Use traditional analysis results
                    self.analysis_results = traditional_results.get('analysis_results', {})
                    self.insights = traditional_results.get('insights', {})
                    self.errors = []
                    self.warnings = []
            
            enhanced_state = EnhancedState(run_manifest, result)
            
            # Run enhanced analysis
            print("   🔍 Running AI pattern analysis...")
            enhanced_result = await self.enhanced_report_agent.process(enhanced_state)
            
            # Update manifest with enhanced results
            if hasattr(enhanced_result, 'llm_enhanced_reports'):
                run_manifest['llm_enhanced_analysis'] = enhanced_result.llm_enhanced_reports
                run_manifest['enhanced_cli_summary'] = enhanced_result.cli_summary
            
            # Phase 3: Save Final Results
            print(f"\n💾 Phase 3: Finalizing Results...")
            
            # Save final manifest
            with open(manifest_path, 'w') as f:
                json.dump(run_manifest, f, indent=2)
            
            # Display comprehensive results
            await self.display_enhanced_results(run_manifest, enhanced_result)
            
            return run_manifest
            
        except Exception as e:
            self.logger.error(f"Enhanced pipeline failed: {str(e)}")
            raise RTGSCLIError(f"Enhanced pipeline execution failed: {str(e)}")

    async def display_enhanced_results(self, run_manifest: Dict[str, Any], enhanced_result) -> None:
        """Display comprehensive results showcasing both traditional and LLM capabilities"""
        
        print(f"\n{'='*80}")
        print(f"🎯 RTGS AI ANALYST - ENHANCED ANALYSIS COMPLETED")
        print(f"{'='*80}")
        
        # Core capabilities showcase
        print(f"\n🤖 DUAL-MODE ANALYSIS CAPABILITIES:")
        print(f"   ✅ Traditional Pipeline: Data cleaning, transformation & statistical analysis")
        print(f"   🧠 LLM Enhancement: AI pattern recognition & policy recommendations")
        print(f"   📊 Multi-Agent System: LangGraph orchestration with 8+ specialized agents")
        print(f"   🎯 Domain Adaptive: Automatically adapts to any government data domain")
        
        # Enhanced analysis results
        enhanced_summary = getattr(enhanced_result, 'cli_summary', {})
        llm_reports = getattr(enhanced_result, 'llm_enhanced_reports', {})
        
        confidence = enhanced_summary.get('confidence_badge', '🟡 MEDIUM')
        quality_score = enhanced_summary.get('quality_score', '75/100')
        
        print(f"\n📊 ANALYSIS SUMMARY:")
        dataset_info = run_manifest.get('dataset_info', {})
        print(f"   📁 Dataset: {dataset_info.get('dataset_name', 'Unknown')}")
        print(f"   🏷️ Domain: {dataset_info.get('domain_hint', 'General').title()}")
        print(f"   🧠 AI Confidence: {confidence}")
        print(f"   📈 Quality Score: {quality_score}")
        print(f"   🔍 Patterns Found: {enhanced_summary.get('findings_count', 0)}")
        print(f"   🎯 Actions Identified: {enhanced_summary.get('actions_count', 0)}")
        
        # Key AI-generated insights
        if enhanced_summary.get('key_findings'):
            print(f"\n💡 KEY AI-GENERATED INSIGHTS:")
            for i, finding in enumerate(enhanced_summary['key_findings'][:3], 1):
                print(f"   {i}. {finding}")
        
        # Priority actions from AI
        if enhanced_summary.get('priority_actions'):
            print(f"\n🚨 AI-RECOMMENDED PRIORITY ACTIONS:")
            for i, action in enumerate(enhanced_summary['priority_actions'][:3], 1):
                print(f"   {i}. {action}")
        
        # Data processing pipeline results
        pipeline_stats = run_manifest.get('pipeline_stats', {})
        print(f"\n🔄 DATA PROCESSING PIPELINE:")
        print(f"   📊 Records Processed: {pipeline_stats.get('rows_processed', 'N/A')}")
        print(f"   🧹 Columns Cleaned: {pipeline_stats.get('cleaned_columns', 'N/A')}")
        print(f"   🔄 Features Engineered: {pipeline_stats.get('features_engineered', 'N/A')}")
        print(f"   📈 Statistical Tests: {pipeline_stats.get('statistical_tests_performed', 'N/A')}")
        
        # Enhanced capabilities demonstration
        print(f"\n🚀 ENHANCED AI CAPABILITIES DEMONSTRATED:")
        print(f"   🎯 Domain Detection: {'AI-powered' if run_manifest.get('run_config', {}).get('enable_domain_detection') else 'Manual'}")
        print(f"   🔍 Pattern Recognition: Advanced AI analysis of data relationships")
        print(f"   💡 Policy Insights: Context-aware recommendations for government action")
        print(f"   📊 Multi-Format Reports: Technical analysis + Policy briefs + Interactive dashboards")
        
        # Generated outputs
        print(f"\n📁 COMPREHENSIVE OUTPUTS GENERATED:")
        artifacts = run_manifest['artifacts_paths']
        
        # Traditional outputs
        print(f"   📋 Traditional Analysis:")
        print(f"     🔧 Technical Report: {artifacts['reports_dir']}/technical_report.md")
        print(f"     📊 Analysis Results: {artifacts['docs_dir']}/analysis_results.json")
        
        # Enhanced LLM outputs
        if llm_reports:
            print(f"   🧠 LLM-Enhanced Analysis:")
            if 'technical_quality_pdf' in llm_reports:
                print(f"     📄 Technical PDF: {llm_reports['technical_quality_pdf']}")
            if 'policy_focused_pdf' in llm_reports:
                print(f"     📋 Policy PDF: {llm_reports['policy_focused_pdf']}")
            if 'interactive_dashboard' in llm_reports:
                print(f"     🌐 Dashboard: {llm_reports['interactive_dashboard']}")
        
        # Quick start guides
        print(f"   🚀 Quick Start:")
        print(f"     👔 Executive Summary: {artifacts['quick_start_dir']}/key_outputs_summary.html")
        print(f"     🏆 Demo Script: {artifacts['quick_start_dir']}/demo_script.md")
        
        # Next steps
        print(f"\n⚡ RECOMMENDED NEXT STEPS:")
        print(f"   1. 📊 Review AI insights in the policy dashboard")
        print(f"   2. 🔍 Validate findings with domain experts") 
        print(f"   3. 📋 Prioritize actions based on urgency and feasibility")
        print(f"   4. 🚀 Begin implementation of immediate actions")
        print(f"   5. 📈 Set up monitoring for recommended KPIs")
        
        # System capabilities summary
        print(f"\n🎯 SYSTEM CAPABILITIES SUMMARY:")
        print(f"   ✅ Data Agnostic: Works with any government dataset structure")
        print(f"   ✅ Domain Adaptive: Automatically provides sector-specific insights")
        print(f"   ✅ LLM Enhanced: AI-powered pattern recognition and policy recommendations")
        print(f"   ✅ Production Ready: Complete audit trail and quality assurance")
        print(f"   ✅ Multi-Audience: Reports for technical teams, policy makers, and executives")
        
        print(f"\n{'='*80}")


def main():
    """Enhanced main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="RTGS AI Analyst - Enhanced Government Data to Policy Insights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Enhanced Examples:
  # AI-powered analysis with automatic domain detection
  python enhanced_cli.py run --dataset data/raw/vehicles.csv --interactive
  
  # Quick analysis with AI enhancement
  python enhanced_cli.py run --dataset data/raw/health.csv --domain auto --scope "Telangana 2023"
  
  # Preview mode with LLM insights
  python enhanced_cli.py run --dataset data/raw/education.csv --mode preview --domain auto
  
  # Full enhanced analysis with custom output
  python enhanced_cli.py run --dataset data/raw/transport.csv --output-dir ./analysis_results --report-format pdf

LLM Features:
  • Automatic domain detection using AI analysis
  • Intelligent pattern recognition and anomaly detection  
  • Context-aware policy recommendations
  • Multi-audience report generation (technical + policy + executive)
  • Interactive dashboards with AI insights
        """
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Enhanced run command
    run_parser = subparsers.add_parser('run', help='Execute enhanced RTGS pipeline with LLM analysis')
    run_parser.add_argument('--dataset', required=True, help='Path to dataset file (CSV/XLSX)')
    run_parser.add_argument('--domain', 
                           choices=['transport', 'health', 'education', 'economics', 'agriculture', 
                                   'environment', 'urban', 'social', 'auto'], 
                           default='auto',
                           help='Dataset domain (auto for AI detection)')
    run_parser.add_argument('--scope', help='Analysis scope (e.g., "Telangana 2020-2024")')
    run_parser.add_argument('--context-file', help='YAML file with dataset context')
    run_parser.add_argument('--interactive', action='store_true', help='Interactive context collection with AI guidance')
    run_parser.add_argument('--mode', choices=['dry-run', 'preview', 'run'], default='run', help='Execution mode')
    run_parser.add_argument('--sample-rows', type=int, default=500, help='Sample size for schema inference')
    run_parser.add_argument('--auto-approve', action='store_true', help='Auto-approve transformations')
    run_parser.add_argument('--output-dir', default='.', help='Output directory for artifacts')
    run_parser.add_argument('--report-format', choices=['markdown', 'pdf', 'html'], default='pdf', help='Report format (PDF recommended for LLM reports)')
    
    # Version command
    version_parser = subparsers.add_parser('version', help='Show version info')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'version':
        print("RTGS AI Analyst v2.0.0 - LLM Enhanced")
        print("Multi-agent pipeline with Claude Sonnet 4 integration")
        print("🧠 Features: AI domain detection, pattern recognition, policy recommendations")
        return
    
    # Setup logging
    setup_logging()
    
    # Initialize enhanced CLI
    cli = EnhancedRTGSCLI()
    
    try:
        if args.command == 'run':
            # Validate dataset file exists
            if not Path(args.dataset).exists():
                print(f"❌ Error: Dataset file not found: {args.dataset}")
                sys.exit(1)
            
            # Run enhanced pipeline
            result = asyncio.run(cli.run_enhanced_pipeline(args))
            print("\n✅ Enhanced pipeline completed successfully!")
            print("🧠 AI-powered insights ready for government decision-making!")
            
    except RTGSCLIError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()