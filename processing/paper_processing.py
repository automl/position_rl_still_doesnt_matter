# requirements:
# pypdf>=3.9.0
# rich>=13.3.0

import os
from pypdf import PdfReader
import re
from collections import defaultdict
import json
import argparse
from typing import List
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from datetime import datetime
import logging

console = Console()


class AnalysisStats:
    def __init__(self):
        self.total_papers = 0
        self.rl_papers = 0
        self.rl_with_code = 0
        self.rl_with_hyperparams = 0
        self.rl_with_env_version = 0
        self.rl_no_seeds = 0
        self.rl_seeds_1_to_5 = 0
        self.rl_seeds_6_to_10 = 0
        self.rl_seeds_over_10 = 0
        self.start_time = datetime.now()
        self.algorithm_counts = defaultdict(int)
        self.papers_with_multiple_algos = 0

    def update_from_paper_info(self, paper_info):
        self.total_papers += 1
        if paper_info["is_rl"]:
            self.rl_papers += 1
            seeds = paper_info.get("num_seeds")

            if not seeds:
                self.rl_no_seeds += 1
            else:
                if 1 <= seeds <= 5:
                    self.rl_seeds_1_to_5 += 1
                elif 6 <= seeds <= 10:
                    self.rl_seeds_6_to_10 += 1
                else:  # seeds > 10
                    self.rl_seeds_over_10 += 1

            if paper_info["code_available"]:
                self.rl_with_code += 1
            if paper_info["hyperparameters_detailed"]:
                self.rl_with_hyperparams += 1
            if paper_info["env_version_specified"]:
                self.rl_with_env_version += 1
            if "algorithms" in paper_info:
                if (
                    "algorithms" in paper_info
                ):  # Note: changed from 'algorithm' to 'algorithms'
                    for algo in paper_info["algorithms"]:
                        self.algorithm_counts[algo] += 1
                    if len(paper_info["algorithms"]) > 1:
                        self.papers_with_multiple_algos += 1


class PaperAnalyzer:
    def __init__(self, base_path="ml_papers/icml", years: List[int] = None):
        self.base_path = base_path
        self.years = years
        self.results = defaultdict(dict)
        self.live_stats = defaultdict(AnalysisStats)

        # Define patterns that will be reused
        self.env_version_pattern = r"(?:Ant|HalfCheetah|Humanoid|Walker|Hopper)-v\d+"

        # Regular expressions for different analyses
        self.rl_keywords = [
            # Must be about implementing/developing RL
            r"we\s+(?:propose|present|introduce|develop)\s+(?:\w+\s+)*?reinforcement\s+learning",
            r"our\s+(?:reinforcement\s+learning|RL)\s+(?:approach|method|algorithm)",
            # RL as the main method
            r"using\s+(?:deep\s+)?reinforcement\s+learning\s+to",
            r"based\s+on\s+(?:deep\s+)?reinforcement\s+learning",
            r"through\s+(?:deep\s+)?reinforcement\s+learning",
            r"trained\s+(?:with\s+|using\s+)?(?:deep\s+|model-based\s+|model-free\s+)?reinforcement\s+learning",
            r"we\s+train\s+(?:a\s+|an\s+)(?:deep\s+|model-based\s+|model-free\s+)?(?:reinforcement\s+learning|RL)",
            # Specific RL algorithms as main contribution
            r"(?:propose|present|introduce|develop)\s+(?:\w+\s+)*?(?:PPO|SAC|TD3|DQN|DDPG|RL)",
            r"our\s+(?:version|variant|implementation)\s+of\s+(?:PPO|SAC|TD3|DQN|DDPG)",
            # Clear technical RL content
            r"policy\s+gradient\s+(?:method|algorithm|approach)",
            r"value\s+function\s+approximation",
            r"Q-function\s+(?:learning|approximation)",
            r"action-value\s+function",
            r"state-action\s+pairs",
            r"(?:discount|reward)\s+function",
            # Environments that strongly indicate RL
            r"(?:MuJoCo|OpenAI\s+Gym|Gymnasium)\s+(?:environment|benchmark|task)",
            self.env_version_pattern,
        ]

        self.code_patterns = [
            r"github\.com",
            r"code.*available.*http",
            r"implementation.*available",
        ]

        self.seed_pattern = r"(?:using|with|across|over|consider)?\s+(\w+)\s+(?:randomly.seeded\s+|seeded\s+|random\s+)?(seeds|runs|trials|training\s+runs)"

        self.hyperparameter_patterns = [
            r"hyperparameter.*appendix",
            r"parameter.*tuning",
            r"grid\s+search",
            r"random\s+search",
        ]

        self.algorithm_patterns = {
            "dqn": [
                r"(?:deep\s+)?[Qq]-(?:network|learning)",
                r"(?:double|dueling|distributional)?\s*(?:DQN|dqn)",
                r"(?:RAINBOW|rainbow)\s*(?:DQN|dqn)",
            ],
            "ddpg": [
                r"(?:deep\s+deterministic\s+policy\s+gradient|DDPG|ddpg)",
                r"(?:TD3-|TD3\s+|td3[-\s])ddpg",
            ],
            "td3": [r"(?:twin\s+delayed|TD3|td3)", r"(?:TD3|td3)(?:\s+algorithm)?"],
            "sac": [
                r"(?:soft\s+actor[-\s]critic|SAC|sac)",
                r"(?:SAC|sac)(?:\s+algorithm)?",
            ],
            "ppo": [
                r"(?:proximal\s+policy\s+optimization|PPO|ppo)",
                r"(?:PPO|ppo)(?:\s+algorithm)?",
            ],
            "trpo": [
                r"(?:trust\s+region\s+policy\s+optimization|TRPO|trpo)",
                r"(?:TRPO|trpo)(?:\s+algorithm)?",
            ],
            "dreamer": [
                r"(?:DREAMER|dreamer)(?:\s+v\d)?(?:\s+algorithm)?",
                r"world\s+model.*(?:DREAMER|dreamer)",
            ],
            "pets": [
                r"(?:probabilistic\s+ensembles\s+with\s+trajectory\s+sampling|PETS|pets)",
                r"(?:PETS|pets)(?:\s+algorithm)?",
            ],
            "td-mpc": [
                r"(?:temporal\s+difference\s+model\s+predictive\s+control|TD-MPC|td-mpc)",
                r"(?:TD[-\s]MPC|td[-\s]mpc)",
            ],
            "a2c": [
                r"(?:advantage\s+actor[-\s]critic|A2C|a2c)",
                r"(?:A2C|a2c)(?:\s+algorithm)?",
            ],
            "a3c": [
                r"(?:asynchronous\s+advantage\s+actor[-\s]critic|A3C|a3c)",
                r"(?:A3C|a3c)(?:\s+algorithm)?",
            ],
        }

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF including handling potential errors."""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                try:
                    text += page.extract_text() + "\n"
                except Exception as e:
                    if not str(e).startswith("Ignoring wrong pointing object"):
                        return (
                            "",
                            f"Warning: Error extracting text from page in {os.path.basename(pdf_path)}: {e}",
                        )
            return text.lower(), None  # Convert to lowercase for easier matching
        except Exception as e:
            if not str(e).startswith(
                ("could not convert string to float", "Ignoring wrong pointing object")
            ):
                return (
                    "",
                    f"Error: Could not read PDF {os.path.basename(pdf_path)}: {e}",
                )
            return "", None

    def is_rl_paper(self, text):
        """Check if paper is about RL based on keywords."""
        matches = []
        for pattern in self.rl_keywords:
            if re.search(pattern, text):
                matches.append(pattern)
        return matches

    def detect_algorithms(self, text):
        """Detect which RL algorithms are actually used/proposed in the paper."""
        matched_algorithms = set()

        # Patterns that indicate mention is just a reference/comparison
        exclusion_contexts = [
            r"(?:compare|compared)\s+(?:to|with)\s+(?:the\s+)?",
            r"unlike",
            r"similar\s+to",
            r"such\s+as",
            r"e\.g\.",
            r"previous\s+work",
            r"prior\s+work",
            r"related\s+work",
            r"other\s+approaches",
            r"existing\s+(?:method|approach|algorithm)",
            r"baseline",
        ]

        # Combine exclusion patterns
        exclusion_pattern = "|".join(exclusion_contexts)

        # Patterns indicating actual usage
        usage_patterns = [
            r"we\s+(?:use|employ|utilize|implement|propose|present|introduce|develop)",
            r"our\s+(?:implementation|version|variant|approach|method)",
            r"based\s+on",
            r"algorithm\s+is",
            r"using\s+(?:a|an|the)",
            r"we\s+build\s+(?:upon|on)",
        ]

        for algo, patterns in self.algorithm_patterns.items():
            for pattern in patterns:
                for usage in usage_patterns:
                    # Check that the algorithm mention is near a usage indicator but not in an exclusion context
                    text_around_usage = re.finditer(
                        f"{usage}.*?{pattern}", text, re.DOTALL
                    )
                    for match in text_around_usage:
                        context = text[
                            max(0, match.start() - 100) : min(
                                len(text), match.end() + 100
                            )
                        ]
                        # Only count if not in an exclusion context
                        if not re.search(exclusion_pattern, context):
                            matched_algorithms.add(algo)
                            break

                    # Also check reverse order (algorithm then usage indicator)
                    text_around_usage = re.finditer(
                        f"{pattern}.*?{usage}", text, re.DOTALL
                    )
                    for match in text_around_usage:
                        context = text[
                            max(0, match.start() - 100) : min(
                                len(text), match.end() + 100
                            )
                        ]
                        if not re.search(exclusion_pattern, context):
                            matched_algorithms.add(algo)
                            break

        if not matched_algorithms:
            return ["other"]

        return list(matched_algorithms)

    def get_num_seeds(self, text):
        """Extract number of seeds used in evaluation."""
        matches = re.findall(self.seed_pattern, text)
        number_strings = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
        }
        if isinstance(matches, list):
            for i in range(len(matches)):
                if isinstance(matches[i], tuple):
                    matches[i] = matches[i][0]
                if matches[i] in number_strings.keys():
                    matches[i] = number_strings[matches[i]]
                try:
                    matches[i] = int(matches[i])
                except:
                    matches[i] = 0
        else:
            if matches in number_strings.keys():
                matches = [number_strings[matches]]
        if matches:
            return max(
                int(x) for x in matches
            )  # Return highest number of seeds mentioned
        return None

    def has_code_available(self, text):
        """Check if code is made available."""
        return any(re.search(pattern, text) for pattern in self.code_patterns)

    def has_hyperparameter_details(self, text):
        """Check if hyperparameter details are provided."""
        return any(re.search(pattern, text) for pattern in self.hyperparameter_patterns)

    def has_env_version(self, text):
        """Check if environment version is specified."""
        return bool(re.search(self.env_version_pattern, text))

    def generate_stats_table(self) -> Table:
        """Generate a rich table with current statistics."""
        table = Table(
            title="Analysis Progress", show_header=True, header_style="bold magenta"
        )
        table.add_column("Year", style="cyan", justify="right")
        table.add_column("Total Papers", justify="right")
        table.add_column("RL Papers", justify="right")
        table.add_column("RL w/Code", justify="right")
        table.add_column("RL w/Params", justify="right")
        table.add_column("RL w/Env", justify="right")
        table.add_column("No Seeds", justify="right", style="red")
        table.add_column("1-5 Seeds", justify="right", style="yellow")
        table.add_column("6-10 Seeds", justify="right", style="green")
        table.add_column(">10 Seeds", justify="right", style="blue")
        table.add_column("Top Algorithms", justify="left")
        table.add_column("Multi-Algo Papers", justify="right")

        for year, stats in sorted(self.live_stats.items()):
            rl_papers = stats.rl_papers
            if rl_papers > 0:
                rl_percentage = rl_papers / stats.total_papers * 100
                no_seeds_pct = stats.rl_no_seeds / rl_papers * 100
                seeds_1_5_pct = stats.rl_seeds_1_to_5 / rl_papers * 100
                seeds_6_10_pct = stats.rl_seeds_6_to_10 / rl_papers * 100
                seeds_over_10_pct = stats.rl_seeds_over_10 / rl_papers * 100
                multi_algo_pct = (
                    stats.papers_with_multiple_algos / stats.rl_papers * 100
                )
                # Get top 3 algorithms
                top_algos = sorted(
                    stats.algorithm_counts.items(),
                    key=lambda x: (-x[1], x[0]),  # Sort by count desc, then name
                )[:3]
                algo_str = ", ".join(f"{algo}({count})" for algo, count in top_algos)

                table.add_row(
                    str(year),
                    str(stats.total_papers),
                    f"{rl_papers} ({rl_percentage:.1f}%)",
                    f"{stats.rl_with_code} ({(stats.rl_with_code / rl_papers * 100):.1f}%)",
                    f"{stats.rl_with_hyperparams} ({(stats.rl_with_hyperparams / rl_papers * 100):.1f}%)",
                    f"{stats.rl_with_env_version} ({(stats.rl_with_env_version / rl_papers * 100):.1f}%)",
                    f"{stats.rl_no_seeds} ({no_seeds_pct:.1f}%)",
                    f"{stats.rl_seeds_1_to_5} ({seeds_1_5_pct:.1f}%)",
                    f"{stats.rl_seeds_6_to_10} ({seeds_6_10_pct:.1f}%)",
                    f"{stats.rl_seeds_over_10} ({seeds_over_10_pct:.1f}%)",
                    algo_str,
                    f"{stats.papers_with_multiple_algos} ({multi_algo_pct:.1f}%)",
                )
            else:
                table.add_row(
                    str(year),
                    str(stats.total_papers),
                    "0 (0.0%)",
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                )

        return table

    def analyze_paper(self, year, filename, task_id):
        """Analyze a single paper and store results."""
        pdf_path = os.path.join(self.base_path, str(year), filename)
        text, error_message = self.extract_text_from_pdf(pdf_path)

        if error_message:
            return error_message

        if not text:
            return None

        rl_matches = self.is_rl_paper(text)
        is_rl = len(rl_matches) > 0
        paper_info = {
            "filename": filename,
            "is_rl": is_rl,
            "rl_matches": rl_matches if is_rl else [],
            "code_available": self.has_code_available(text),
            "hyperparameters_detailed": self.has_hyperparameter_details(text),
            "env_version_specified": self.has_env_version(text),
        }

        if is_rl:
            paper_info["num_seeds"] = self.get_num_seeds(text)
            paper_info["algorithms"] = self.detect_algorithms(text)

        self.results[year][filename] = paper_info
        self.live_stats[year].update_from_paper_info(paper_info)
        return None

    def analyze_all_papers(self):
        """Analyze all papers in the directory structure."""
        years_to_process = self.years or range(2018, 2025)

        # Create layout
        layout = Layout()

        # Split layout into three main sections
        layout.split(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )

        # Split body into statistics and log sections
        layout["body"].split_row(
            Layout(name="stats", ratio=1), Layout(name="log", ratio=1)
        )

        # Configure the sections
        layout["header"].update(
            Panel.fit(
                "[bold blue]ICML Paper Analyzer[/bold blue]\n"
                "Analyzing papers for RL content, code availability, and methodology details",
                border_style="blue",
            )
        )

        log_messages = []

        def update_display():
            # Update stats section
            layout["stats"].update(
                Panel(
                    self.generate_stats_table(),
                    title="Analysis Statistics",
                    border_style="green",
                )
            )

            # Update log section
            layout["log"].update(
                Panel(
                    "\n".join(log_messages[-10:]),  # Show last 10 messages
                    title="Processing Log",
                    border_style="yellow",
                )
            )

        with Live(layout, refresh_per_second=4, screen=True):
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=Console(record=True),
                transient=False,
                expand=True,
            ) as progress:
                layout["footer"].update(progress)

                for year in years_to_process:
                    year_path = os.path.join(self.base_path, str(year))
                    if not os.path.exists(year_path):
                        log_messages.append(f"Directory for year {year} not found")
                        update_display()
                        continue

                    pdf_files = [f for f in os.listdir(year_path) if f.endswith(".pdf")]
                    if not pdf_files:
                        log_messages.append(f"No PDF files found for year {year}")
                        update_display()
                        continue

                    task_id = progress.add_task(
                        f"Processing {year}", total=len(pdf_files)
                    )

                    for filename in pdf_files:
                        # Capture any warnings or errors during paper analysis
                        try:
                            error = self.analyze_paper(year, filename, task_id)
                            if error:
                                log_messages.append(error)
                        except Exception as e:
                            log_messages.append(
                                f"Error processing {filename}: {str(e)}"
                            )

                        progress.advance(task_id)
                        update_display()

    def save_results(self, output_path="paper_analysis_results.json"):
        """Save detailed results to JSON file."""
        base_name = output_path.rsplit(".", 1)[0]

        # Save full results
        with open(output_path, "w") as f:
            json.dump(
                {
                    "detailed_results": dict(self.results),
                    "summary": self.generate_summary(),
                },
                f,
                indent=2,
            )
        console.print(f"[green]Full results saved to {output_path}[/green]")

        # Save RL papers summary
        rl_papers = {}
        for year, papers in self.results.items():
            rl_papers[year] = [
                {
                    "filename": info["filename"],
                    "matched_keywords": info["rl_matches"],
                    "num_seeds": info.get("num_seeds"),
                    "code_available": info["code_available"],
                    "hyperparameters_detailed": info["hyperparameters_detailed"],
                    "env_version_specified": info["env_version_specified"],
                    "algorithms": info.get("algorithms", ["Not specified"]),
                }
                for info in papers.values()
                # if info['is_rl']
            ]

        rl_papers_path = f"{base_name}_rl_papers.json"
        with open(rl_papers_path, "w") as f:
            json.dump(rl_papers, f, indent=2)
        console.print(f"[green]RL papers list saved to {rl_papers_path}[/green]")

        # Create a more readable markdown report for RL papers
        markdown_report = ["# RL Papers Analysis\n"]
        for year, papers in sorted(rl_papers.items()):
            markdown_report.append(f"\n## Year {year} ({len(papers)} RL papers)\n")
            for paper in sorted(papers, key=lambda x: x["filename"]):
                markdown_report.append(f"\n### {paper['filename']}")
                markdown_report.append(
                    f"* Matched keywords: {', '.join(paper['matched_keywords'])}"
                )
                markdown_report.append(
                    f"* Number of seeds: {paper['num_seeds'] if paper['num_seeds'] else 'Not specified'}"
                )
                markdown_report.append(
                    f"* Code available: {'Yes' if paper['code_available'] else 'No'}"
                )
                markdown_report.append(
                    f"* Hyperparameters detailed: {'Yes' if paper['hyperparameters_detailed'] else 'No'}"
                )
                markdown_report.append(
                    f"* Environment version specified: {'Yes' if paper['env_version_specified'] else 'No'}"
                )
                markdown_report.append(f"* Algorithm: {paper['algorithms']}\n")

        markdown_report_path = f"{base_name}_rl_papers.md"
        with open(markdown_report_path, "w") as f:
            f.write("\n".join(markdown_report))
        console.print(
            f"[green]Readable RL papers report saved to {markdown_report_path}[/green]"
        )

    def generate_summary(self):
        """Generate summary statistics."""
        summary = defaultdict(lambda: defaultdict(int))

        for year, papers in self.results.items():
            year_summary = summary[year]
            year_summary["total_papers"] = len(papers)

            for paper_info in papers.values():
                if paper_info["is_rl"]:
                    year_summary["rl_papers"] += 1
                    if paper_info["code_available"]:
                        year_summary["rl_with_code"] += 1
                    if paper_info["hyperparameters_detailed"]:
                        year_summary["rl_with_hyperparams"] += 1
                    if paper_info["env_version_specified"]:
                        year_summary["rl_with_env_version"] += 1

                    seeds = paper_info.get("num_seeds")
                    if not seeds:
                        year_summary["rl_no_seeds"] += 1
                    elif 1 <= seeds <= 5:
                        year_summary["rl_seeds_1_to_5"] += 1
                    elif 6 <= seeds <= 10:
                        year_summary["rl_seeds_6_to_10"] += 1
                    else:
                        year_summary["rl_seeds_over_10"] += 1

        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ICML papers for RL content and metadata."
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        help="Specific years to analyze (e.g., --years 2018 2019)",
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default="ml_papers/icml",
        help="Base path for the ICML papers directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="paper_analysis_results.json",
        help="Output file path for the analysis results",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress PDF parsing warnings"
    )

    args = parser.parse_args()

    if args.quiet:
        logging.getLogger("pypdf").setLevel(logging.ERROR)

    # Input validation
    if args.years:
        for year in args.years:
            if not (2018 <= year <= 2025):
                parser.error(f"Year {year} is outside the valid range (2018-2024)")

    console.print(
        Panel.fit(
            "[bold blue]ICML Paper Analyzer[/bold blue]\n"
            "Analyzing papers for RL content, code availability, and methodology details",
            border_style="blue",
        )
    )

    analyzer = PaperAnalyzer(base_path=args.base_path, years=args.years)
    analyzer.analyze_all_papers()
    analyzer.save_results(args.output)

    # Final summary
    console.print("\n[bold green]Analysis Complete![/bold green]")
    console.print(analyzer.generate_stats_table())


if __name__ == "__main__":
    main()
