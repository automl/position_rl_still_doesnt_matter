import re
import json
from pathlib import Path


def parse_ai_conference(file_path):
    """
    Parse AI conference paper summaries from a file and extract relevant sections.

    Args:
        file_path (str): Path to the file containing the paper summaries.

    Returns:
        list: A list of dictionaries with parsed paper information.
    """
    with open(file_path, "r") as file:
        content = file.read()

    # Split the file content into entries for each paper
    papers = content.split("### ")[1:]  # Skip the initial metadata/header

    parsed_papers = []

    for paper in papers:
        entry = {}

        # Extract title
        title_match = re.search(r"^(.+)", paper)
        entry["title"] = title_match.group(1) if title_match else "Unknown"

        # Extract link
        link_match = re.search(r"Link: (https?://\S+)", paper)
        entry["link"] = link_match.group(1) if link_match else None

        # Extract keywords
        keywords_match = re.search(r"Keywords: (.+)", paper)
        entry["keywords"] = (
            [kw.strip() for kw in keywords_match.group(1).split(",")]
            if keywords_match
            else []
        )

        # Extract research goal
        research_goal_match = re.search(r"Research goal: (.+)", paper)
        entry["research_goal"] = (
            research_goal_match.group(1) if research_goal_match else None
        )

        # Extract empirical
        empirical_match = re.search(r"Empirical: (.+)", paper)
        entry["empirical"] = empirical_match.group(1) if empirical_match else None

        # Extract algorithms
        algorithms_match = re.search(r"Algorithms: (.+)", paper)
        entry["algorithms"] = (
            [alg.strip() for alg in algorithms_match.group(1).split(",")]
            if algorithms_match
            else []
        )

        # Extract number of seeds
        seeds_match = re.search(r"Seeds: (.+)", paper)
        entry["seeds"] = seeds_match.group(1) if seeds_match else None

        # Extract code availability
        code_match = re.search(r"Code: (.+)", paper)
        entry["code_available"] = (
            code_match.group(1).lower() == "yes" if code_match else False
        )

        # Extract environment specification
        env_match = re.search(r"Env: (.+)", paper)
        entry["environment_spec"] = env_match.group(1) if env_match else None

        # Extract hyperparameter specification
        hyperparams_match = re.search(r"Hyperparameters: (.+)", paper)
        entry["hyperparams_spec"] = (
            hyperparams_match.group(1) if hyperparams_match else None
        )

        parsed_papers.append(entry)

    return parsed_papers


# Example usage
base_path = "ml_papers/manual/"
for file_path in [
    "rlc_awards_24.md",
    "iclr_awards_24.md",
    "icml_orals_24.md",
    "neurips_orals_21.md",
    "neurips_orals_22.md",
]:
    output_path = base_path + file_path.split(".")[0] + ".json"
    if not Path(output_path).exists():
        parsed_data = parse_ai_conference(Path(base_path) / file_path)

        # Save the parsed data to a JSON file
        with open(output_path, "w") as json_file:
            json.dump(parsed_data, json_file, indent=4)

        print(f"Parsed data saved to {output_path}")
    else:
        print(f"File {file_path} already parsed.")
