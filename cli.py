#!/usr/bin/env python3
"""
RTGS AI Analyst - Command Line Interface
Multi-agent pipeline for government data analysis and policy insights
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

# Import our orchestrator
from src.orchestrator.flow_controller import RTGSOrchestrator
from src.utils.logging import setup_logging, get_logger


class RTGSCLIError(Exception):
    """Custom exception for CLI errors"""
    pass


class RTGSCLI:
    """RTGS AI Analyst Command Line Interface"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.orchestrator = None
        
    def create_run_manifest(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Create initial run manifest with metadata"""
        run_id = f"rtgs-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{str(uuid.uuid4())[:8]}"
        
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
                "domain_hint": args.domain or dataset_context.get('domain', 'unknown'),
                "scope": args.scope or dataset_context.get('scope', 'unspecified'),
                "description": dataset_context.get('description', 'No description provided')
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
                "report_format": args.report_format
            },
            "agent_version_tags": {
                "orchestrator": "v1.0",
                "ingestion": "v1.0", 
                "schema": "v1.0",
                "cleaning": "v1.0",
                "analysis": "v1.0",
                "insight": "v1.0"
            },
            "artifacts_paths": {},
            "confidence_overall": "PENDING",
            "notes": []
        }
        
        return manifest

    def setup_output_directories(self, run_manifest: Dict[str, Any]) -> None:
        """Create output directory structure"""
        base_dir = Path(run_manifest["run_config"]["output_dir"])
        run_id = run_manifest["run_id"]
        
        directories = [
            base_dir / "artifacts" / "logs",
            base_dir / "artifacts" / "reports", 
            base_dir / "artifacts" / "plots" / "interactive",
            base_dir / "artifacts" / "docs",
            base_dir / "artifacts" / "quick_start",
            base_dir / "data" / "raw",
            base_dir / "data" / "standardized", 
            base_dir / "data" / "cleaned",
            base_dir / "data" / "transformed"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Update manifest with artifact paths
        artifacts_base = base_dir / "artifacts"
        run_manifest["artifacts_paths"] = {
            "logs_dir": str(artifacts_base / "logs"),
            "reports_dir": str(artifacts_base / "reports"),
            "plots_dir": str(artifacts_base / "plots"),
            "docs_dir": str(artifacts_base / "docs"),
            "quick_start_dir": str(artifacts_base / "quick_start"),
            "run_manifest": str(artifacts_base / "docs" / "run_manifest.json")
        }

    def interactive_context_collection(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Interactively collect dataset context from user"""
        print("\n🎯 RTGS AI ANALYST - Dataset Context Collection")
        print("=" * 60)
        
        context = {}
        
        # Domain
        if not args.domain:
            print("\n📋 What domain does this dataset belong to?")
            print("Options: transport, health, education, economics, agriculture, energy, other")
            context['domain'] = input("Domain [transport]: ").strip() or "transport"
        else:
            context['domain'] = args.domain
            
        # Description
        print(f"\n📝 Brief description of the dataset:")
        context['description'] = input("Description: ").strip() or "Government dataset for analysis"
        
        # Business questions
        print(f"\n❓ What are the key policy questions you want answered? (one per line, empty line to finish)")
        business_questions = []
        while True:
            question = input("Question: ").strip()
            if not question:
                break
            business_questions.append(question)
        context['business_questions'] = business_questions or ["What are the key trends and patterns?"]
        
        # Key metrics of interest
        print(f"\n📊 What are the key metrics/columns of interest? (comma-separated)")
        metrics_input = input("Key metrics: ").strip()
        context['key_metrics'] = [m.strip() for m in metrics_input.split(",")] if metrics_input else []
        
        # Scope
        if not args.scope:
            print(f"\n🌍 What is the scope of this analysis?")
            print("Example: 'Telangana 2020-2024', 'Hyderabad district 2023', 'All districts current year'")
            context['scope'] = input("Scope: ").strip() or "Telangana state"
        else:
            context['scope'] = args.scope
            
        # Stakeholders
        print(f"\n👥 Who are the primary stakeholders/audience for this analysis?")
        context['stakeholders'] = input("Stakeholders [Government officials]: ").strip() or "Government officials"
        
        return context

    async def run_pipeline(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Execute the complete RTGS pipeline"""
        try:
            # Collect context interactively if needed
            if args.interactive:
                dataset_context = self.interactive_context_collection(args)
                # Save context for future use
                context_file = f"context_{Path(args.dataset).stem}.yaml"
                with open(context_file, 'w') as f:
                    yaml.dump(dataset_context, f, default_flow_style=False)
                print(f"\n💾 Context saved to {context_file} for future runs")
                args.context_file = context_file

            # Create run manifest
            run_manifest = self.create_run_manifest(args)
            
            # Setup output directories
            self.setup_output_directories(run_manifest)
            
            # Save initial manifest
            manifest_path = run_manifest["artifacts_paths"]["run_manifest"]
            with open(manifest_path, 'w') as f:
                json.dump(run_manifest, f, indent=2)
            
            print(f"\n🚀 Starting RTGS AI Analyst Pipeline")
            print(f"📋 Run ID: {run_manifest['run_id']}")
            print(f"📁 Dataset: {run_manifest['dataset_info']['dataset_name']}")
            print(f"🏷️  Domain: {run_manifest['dataset_info']['domain_hint']}")
            print(f"📊 Mode: {run_manifest['run_config']['mode']}")
            print(f"📁 Outputs: {run_manifest['run_config']['output_dir']}")
            
            # Initialize orchestrator
            self.orchestrator = RTGSOrchestrator(run_manifest)
            
            # Execute pipeline based on mode
            if args.mode == "dry-run":
                result = await self.orchestrator.dry_run()
                print("\n✅ Dry run completed - no data transformations applied")
                
            elif args.mode == "preview":
                result = await self.orchestrator.preview_run()
                print("\n👀 Preview completed - check transforms_preview.csv for proposed changes")
                
                if not args.auto_approve:
                    approval = input("\n🤔 Apply these transformations? [y/N]: ").strip().lower()
                    if approval not in ['y', 'yes']:
                        print("❌ Pipeline cancelled by user")
                        return run_manifest
                
                # Continue with full run if approved
                result = await self.orchestrator.full_run()
                
            else:  # full run
                result = await self.orchestrator.full_run()
            
            # Update manifest with results
            run_manifest.update(result)
            
            # Save final manifest
            with open(manifest_path, 'w') as f:
                json.dump(run_manifest, f, indent=2)
            
            # Display results
            self.display_results(run_manifest)
            
            return run_manifest
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise RTGSCLIError(f"Pipeline execution failed: {str(e)}")

    def display_results(self, run_manifest: Dict[str, Any]) -> None:
        """Display final results in CLI"""
        confidence = run_manifest.get('confidence_overall', 'UNKNOWN')
        confidence_color = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(confidence, "⚪")
        
        print(f"\n{'='*80}")
        print(f"🎯 RTGS AI ANALYST COMPLETED")
        print(f"{'='*80}")
        
        # Quick stats
        stats = run_manifest.get('pipeline_stats', {})
        print(f"📊 Dataset: {run_manifest['dataset_info']['dataset_name']}")
        print(f"📈 Rows Processed: {stats.get('rows_processed', 'N/A')}")
        print(f"🔧 Transformations: {stats.get('transformations_applied', 'N/A')}")
        print(f"{confidence_color} Confidence: {confidence}")
        
        # Key insights
        insights = run_manifest.get('key_insights', [])
        if insights:
            print(f"\n💡 KEY POLICY INSIGHTS:")
            for i, insight in enumerate(insights[:3], 1):
                print(f"   {i}. {insight}")
        
        print(f"\n📁 OUTPUTS GENERATED:")
        artifacts = run_manifest['artifacts_paths']
        print(f"👔 Policy Team: {artifacts['quick_start_dir']}/key_outputs_summary.html")
        print(f"🔧 Technical Review: {artifacts['reports_dir']}/technical_report.md")
        print(f"🏆 Hackathon Demo: {artifacts['quick_start_dir']}/demo_script.md")
        
        print(f"\n⚡ Next Steps:")
        print(f"👔 Policy Dashboard: Open {artifacts['plots_dir']}/interactive/policy_dashboard.html")
        print(f"🔧 Full Methodology: Read {artifacts['reports_dir']}/technical_report.md")
        print(f"🏆 Demo Ready: Follow {artifacts['quick_start_dir']}/demo_script.md")
        
        print(f"\n{'='*80}")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="RTGS AI Analyst - Government Data to Policy Insights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive run with context collection
  python cli.py run --dataset data/raw/vehicles.csv --interactive
  
  # Quick run with domain hint
  python cli.py run --dataset data/raw/health.csv --domain health --scope "Telangana 2023"
  
  # Preview mode (show transforms before applying)
  python cli.py run --dataset data/raw/education.csv --mode preview
  
  # Dry run (schema inference only)
  python cli.py run --dataset data/raw/transport.csv --mode dry-run
        """
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Execute RTGS pipeline')
    run_parser.add_argument('--dataset', required=True, help='Path to dataset file (CSV/XLSX)')
    run_parser.add_argument('--domain', choices=['transport', 'health', 'education', 'economics', 'agriculture', 'energy', 'other'], help='Dataset domain hint')
    run_parser.add_argument('--scope', help='Analysis scope (e.g., "Telangana 2020-2024")')
    run_parser.add_argument('--context-file', help='YAML file with dataset context')
    run_parser.add_argument('--interactive', action='store_true', help='Interactive context collection')
    run_parser.add_argument('--mode', choices=['dry-run', 'preview', 'run'], default='run', help='Execution mode')
    run_parser.add_argument('--sample-rows', type=int, default=500, help='Sample size for schema inference')
    run_parser.add_argument('--auto-approve', action='store_true', help='Auto-approve transformations')
    run_parser.add_argument('--output-dir', default='.', help='Output directory for artifacts')
    run_parser.add_argument('--report-format', choices=['markdown', 'pdf', 'html'], default='markdown', help='Report format')
    
    # Version command
    version_parser = subparsers.add_parser('version', help='Show version info')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'version':
        print("RTGS AI Analyst v1.0.0")
        print("Multi-agent pipeline for government data analysis")
        return
    
    # Setup logging
    setup_logging()
    
    # Initialize CLI
    cli = RTGSCLI()
    
    try:
        if args.command == 'run':
            # Validate dataset file exists
            if not Path(args.dataset).exists():
                print(f"❌ Error: Dataset file not found: {args.dataset}")
                sys.exit(1)
            
            # Run pipeline
            result = asyncio.run(cli.run_pipeline(args))
            print("\n✅ Pipeline completed successfully!")
            
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