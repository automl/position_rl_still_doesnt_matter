import logging
from rich.console import Console
from rich.panel import Panel

from processing.scraper import MLPaperProcessor 
from processing.paper_processing import PaperAnalyzer

console = Console()

def download_papers(args):
    processor = MLPaperProcessor(base_dir=args.output_dir)
    processor.process_all_papers(
        start_year=args.start_year,
        conferences=args.conferences
    )

def process_papers(base_path, args):
    if args.quiet:
        logging.getLogger('pypdf').setLevel(logging.ERROR)

    # Input validation
    if args.years:
        for year in args.years:
            if not (2018 <= year <= 2025):
                parser.error(f"Year {year} is outside the valid range (2018-2025)")

    console.print(Panel.fit(
        "[bold blue]ICML Paper Analyzer[/bold blue]\n"
        "Analyzing papers for RL content, code availability, and methodology details",
        border_style="blue"
    ))

    analyzer = PaperAnalyzer(base_path=base_path, years=args.years)
    analyzer.analyze_all_papers()
    analyzer.save_results(args.output)

    # Final summary
    console.print("\n[bold green]Analysis Complete![/bold green]")
    console.print(analyzer.generate_stats_table())

def process_papers():
    pass

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='ML Paper Downloader')
    parser.add_argument('--nodownload', action='store_false',
                      help='Processing without paper downloads')
    parser.add_argument('--start-year', type=int, default=2018,
                        help='Start year for paper collection')
    parser.add_argument('--conferences', nargs='+',
                        choices=['neurips', 'icml', 'iclr', 'rlj'],
                        help='Specific conferences to download from')
    parser.add_argument('--output-dir', type=str, default='ml_papers',
                        help='Directory to store downloaded papers')
    parser.add_argument('--years', type=int, nargs='+',
                      help='Specific years to analyze (e.g., --years 2018 2019)')
    parser.add_argument('--output', type=str, default="paper_analysis_results.json",
                      help='Output file path for the analysis results')
    parser.add_argument('--quiet', action='store_true',
                      help='Suppress PDF parsing warnings')

    args = parser.parse_args()

    if not args.nodownload:
        download_papers(args)

    for conf in args.conferences:
        base_path = args.output_path + f"/{conf}"
        process_papers(base_path, args)
        
    #generate_automated_dataset(args)