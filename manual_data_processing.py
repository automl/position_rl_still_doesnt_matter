import os
import json
import pandas as pd
from pathlib import Path

from processing.parse_manual_info import parse_ai_conference

def parse_manuals(base_path):
    for file in os.listdir(base_path):
        if file.endswith(".md"):
            output_path = base_path + file.split(".")[0] + ".json"
            if not Path(output_path).exists():
                parsed_data = parse_ai_conference(Path(base_path) / file)

                # Save the parsed data to a JSON file
                with open(output_path, "w") as json_file:
                    json.dump(parsed_data, json_file, indent=4)

                print(f"Parsed data saved to {output_path}")
            else:
                print(f"File {file} already parsed.")

def make_dataset(base_path):
    dfs = []
    for file in os.listdir(base_path):
        if file.endswith(".json"):
            df = pd.read_json(Path(base_path)/file)
            df["year"] = file.split("_")[-1].split(".")[0]
            df["conference"] = file.split("_")[0]
            df["conf_id"] = df["conference"] + "_" + df["year"]
            empirical_to_yes_no_other = df["empirical"].apply(lambda x: "Yes" if x == "yes" else "No" if x == "no" else "other")
            df["empirical"] = empirical_to_yes_no_other
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
                
            seeds_to_bins = df["seeds"].apply(bin_seeds)
            df["seeds"] = seeds_to_bins
            dfs.append(df)
    df = pd.concat(dfs)
    max_keywords = df["keywords"].apply(len).max()
    padded_keywords = df["keywords"].apply(lambda x: x + [""]*(max_keywords-len(x)))
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

    df.drop(columns=["keywords", "algorithms"], inplace=True)
    if Path("processed_data/manual_paper_annotations.csv").exists():
        os.remove("processed_data/manual_paper_annotations.csv")
    df.to_csv("processed_data/manual_paper_annotations.csv", index=False)

    # make files of keywords for prompting llms. One line per paper, all keywords separated by commas
    # one file for each conference
    for conf in df["conference"].unique():
        for year in df["year"].unique():
            filtered = df[(df["year"]==year) & (df["conference"]==conf)]
            if not filtered.empty:
                with open(f"processed_data/manual_paper_keywords_{conf}_{year}.txt", "w") as f: 
                    for _, paper in filtered.iterrows():
                        keywords = []
                        for i in range(max_keywords):
                            keywords.append(paper[f"keyword_{i}"])
                        keywords = [k for k in keywords if k!=""]
                        f.write(",".join(keywords) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='ML Paper Downloader')
    parser.add_argument('--base-path', type=str, default='ml_papers/manual/',
                      help='Location of files to parse')

    args = parser.parse_args()

    parse_manuals(args.base_path)
    make_dataset(args.base_path)