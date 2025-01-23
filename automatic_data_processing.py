import logging
import pandas as pd
import os
import json
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

from processing.scraper import MLPaperProcessor 
from processing.paper_processing import PaperAnalyzer

console = Console()

def download_papers(args):
    processor = MLPaperProcessor(base_dir=args.output_dir)
    processor.process_all_papers(
        start_year=args.start_year,
        conferences=args.conferences,
        years=args.years
    )

def process_papers(base_path, output_path, args):
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
    analyzer.save_results(output_path)

    # Final summary
    console.print("\n[bold green]Analysis Complete![/bold green]")
    console.print(analyzer.generate_stats_table())

def make_dataset(base_path):
    dfs = []
    for file in os.listdir(base_path):
        if file.endswith(".json"):
            conference = file.split("_")[0]
            raw_file = json.load(open(Path(base_path)/file))
            for k in raw_file.keys():
                df = pd.DataFrame(raw_file[k])
                df["year"] = k
                df["conference"] = conference
                df["conf_id"] = df["conference"] + "_" + df["year"]
                def bin_seeds(seed):
                    try:
                        seed = int(seed)
                    except:
                        return "0"
                    if seed==0:
                        return "0"
                    elif seed <= 5:
                        return "1-5"
                    elif seed <= 10:
                        return "6-10"
                    else:
                        return "over 10"
                seeds_to_bins = df["num_seeds"].apply(bin_seeds)
                df["seeds"] = seeds_to_bins
                df.drop(columns=["num_seeds"], inplace=True)
                dfs.append(df)
    df = pd.concat(dfs)
    max_keywords = df["matched_keywords"].apply(len).max()
    padded_keywords = df["matched_keywords"].apply(lambda x: x + [""]*(max_keywords-len(x)))
    keyword_dict = {f"keyword_{i}": [] for i in range(max_keywords)}
    for paper in padded_keywords:
        for i, keyword in enumerate(paper):
            keyword_dict[f"keyword_{i}"].append(keyword)
    df = df.reset_index()
    df = pd.concat([df, pd.DataFrame(keyword_dict)], axis=1)

    df["algorithms"].loc[df["algorithms"].isnull()] = df["algorithms"].loc[df["algorithms"].isnull()].apply(lambda x: [])
    max_algorithms = df["algorithms"].apply(len).max()
    padded_algorithms = df["algorithms"].apply(lambda x: x + [""]*(max_algorithms-len(x)))
    algorithm_dict = {f"algorithm_{i}": [] for i in range(max_algorithms)}
    for paper in padded_algorithms:
        for i, algorithm in enumerate(paper):
            algorithm_dict[f"algorithm_{i}"].append(algorithm)
    df = pd.concat([df, pd.DataFrame(algorithm_dict)], axis=1)

    df.drop(columns=["matched_keywords", "algorithms"], inplace=True)
    df = df.reset_index(drop=True)
    if Path("processed_data/automatic_paper_annotations.csv").exists():
        os.remove("processed_data/automatic_paper_annotations.csv")
    df.to_csv("processed_data/automatic_paper_annotations.csv", index=False)

    # make files of keywords for prompting llms. One line per paper, all keywords separated by commas
    # one file for each conference
    for conf in df["conference"].unique():
        for year in df["year"].unique():
            filtered = df[(df["year"]==year) & (df["conference"]==conf)]
            if not filtered.empty:
                with open(f"processed_data/automatic_paper_keywords_{conf}_{year}.txt", "w") as f: 
                    for _, paper in filtered.iterrows():
                        keywords = []
                        for i in range(max_keywords):
                            keywords.append(paper[f"keyword_{i}"])
                        keywords = [k for k in keywords if k!=""]
                        f.write(",".join(keywords) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='ML Paper Downloader')
    parser.add_argument('--nodownload', action='store_true',
                      help='Processing without paper downloads')
    parser.add_argument('--start-year', type=int, default=2018,
                        help='Start year for paper collection')
    parser.add_argument('--conferences', nargs='+',
                        choices=['neurips', 'icml', 'iclr', 'rlj'],
                        default=['neurips', 'icml', 'iclr', 'rlj'],
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
        if (Path(args.output_dir) / f"{conf}_rl_papers.json").exists() or not (Path(args.output_dir) / f"{conf}").exists():
            continue
        base_path = Path(args.output_dir) / f"{conf}"
        output_path = Path(args.output_dir) / f"{conf}.json"
        process_papers(str(base_path), str(output_path), args)

    make_dataset(args.output_dir)