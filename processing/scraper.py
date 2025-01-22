import requests
from bs4 import BeautifulSoup
import os
import time
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import List, Dict
import openreview


class ConferenceScraper(ABC):
    def __init__(self, base_dir: str):
        self.base_dir = os.path.join(base_dir, self.conference_name)
        os.makedirs(self.base_dir, exist_ok=True)
        self.session = requests.Session()

    @property
    @abstractmethod
    def conference_name(self) -> str:
        pass

    @abstractmethod
    def get_conference_urls(self, start_year: int) -> List[str]:
        pass

    @abstractmethod
    def get_paper_links(self, conference_url: str) -> List[Dict[str, str]]:
        pass


class OpenReviewScraper(ConferenceScraper):
    def get_venue(self, client, year):
        venues = client.get_group(id="venues").members
        venues = filter(lambda venue: venue is not None, venues)
        reqd_venues = []
        for venue in venues:
            if self.conference_name in venue.lower():
                reqd_venues.append(venue)
        reqd_venues = [r for r in reqd_venues if "workshop" not in r.lower()]
        reqd_venues = [r for r in reqd_venues if "tiny" not in r.lower()]
        reqd_venues = [r for r in reqd_venues if "blog" not in r.lower()]
        reqd_venues = [r for r in reqd_venues if "competition" not in r.lower()]
        reqd_venues = [r for r in reqd_venues if "creative" not in r.lower()]
        reqd_venues = [r for r in reqd_venues if "school" not in r.lower()]
        reqd_venues = [r for r in reqd_venues if "workshop" not in r.lower()]
        reqd_venues = [r for r in reqd_venues if "data" not in r.lower()]
        reqd_venues = [r for r in reqd_venues if str(year) in r]
        if len(reqd_venues) == 0:
            raise ValueError(f"No venue found for {self.conference_name} in {year}")
        return reqd_venues[0]

    def _get_paper_links(self, year: int) -> List[Dict[str, str]]:
        if year >= self.api_cutoff_year:
            client = openreview.api.OpenReviewClient(
                baseurl="https://api2.openreview.net"
            )
        else:
            client = openreview.Client(baseurl="https://api.openreview.net")
        venue = self.get_venue(client, year)
        papers = client.get_all_notes(content={"venueid": venue})
        paper_infos = []
        try:
            for paper in papers:
                keywords = paper.content.get("keywords")
                title = paper.content.get("title")
                if year >= self.api_cutoff_year:
                    keywords = keywords["value"]
                    title = title["value"]
                rl_paper = False
                if "RL" in title or "Reinforcement Learning" in title:
                    rl_paper = True
                for keyword in keywords:
                    if (
                        "RL" in keyword
                        or "Reinforcement" in keyword
                        or "reinforcement" in keyword
                    ):
                        rl_paper = True
                if rl_paper:
                    if year >= self.api_cutoff_year:
                        author_profiles = openreview.tools.get_profiles(
                            client, paper.content["authorids"]["value"]
                        )
                        url = f"https://openreview.net{paper.content['pdf']['value']}"
                    else:
                        author_profiles = openreview.tools.get_profiles(
                            client, paper.content["authorids"]
                        )
                        url = f"https://openreview.net{paper.content['pdf']}"
                    author_str = ""
                    for author_profile in author_profiles:
                        author_str = ", ".join(
                            [author_str, author_profile.get_preferred_name(pretty=True)]
                        )
                    paper_infos.append(
                        {
                            "title": title,
                            "authors": author_str[2:],
                            "pdf_url": url,
                            "year": str(year),
                            "downloaded": True,
                        }
                    )
                    year_dir = os.path.join(self.base_dir, str(year))
                    os.makedirs(year_dir, exist_ok=True)

                    safe_title = "".join(
                        c for c in title if c.isalnum() or c in (" ", "-", "_")
                    ).rstrip()
                    filename = os.path.join(year_dir, f"{safe_title[:100]}.pdf")
                    f = client.get_attachment(paper.id, "pdf")
                    with open(filename, "wb") as op:
                        op.write(f)
        except Exception as e:
            print(f"Error scraping {self.conference_name}: {str(e)}")
        return paper_infos


class NeurIPSScraper(OpenReviewScraper):
    conference_name = "neurips"
    api_cutoff_year = 2023

    def get_conference_urls(self, start_year: int) -> List[str]:
        earlier = [f"https://papers.nips.cc/paper_files/paper/{year}" for year in range(start_year, 2021)]
        later = [
            f"https://api.openreview.net/notes?content.venue=NeurIPS+{year}+Conference&details=replyCount&offset=0&limit=1000&invitation=NeurIPS.cc/{year}/"
            for year in range(2021, 2024)
        ]
        return earlier + later

    def get_paper_links(self, conference_url: str) -> List[Dict[str, str]]:
        year = int(conference_url.split(".cc")[1].split("/")[-1])
        # use openreview
        if year >= 2021:
            return self._get_paper_links(year)
        # use proceedings
        else:
            papers = []
            try:
                response = self.session.get(conference_url)
                if response.status_code == 404:
                    print(f"Conference URL not found: {conference_url}")
                    return papers

                soup = BeautifulSoup(response.text, "html.parser")

                for paper in soup.find_all("div", class_="paper"):
                    title_elem = paper.find("p", class_="title")
                    pdf_link = paper.find("a", string="Download PDF")

                    if title_elem and pdf_link:
                        volume = conference_url.split("/")[-1].lstrip("v")
                        year = next(
                            (y for y, v in self.volume_dict.items() if str(v) == volume),
                            None,
                        )
                        if year:
                            papers.append(
                                {
                                    "title": title_elem.text.strip(),
                                    "pdf_url": urljoin(conference_url, pdf_link["href"]),
                                    "year": str(year),
                                }
                            )

            except Exception as e:
                print(f"Error scraping NeurIPS {year} {conference_url}: {str(e)}")
            return papers


class ICLRScraper(OpenReviewScraper):
    conference_name = "iclr"
    api_cutoff_year = 2024

    def get_conference_urls(self, start_year: int) -> List[str]:
        return [
            f"https://api.openreview.net/notes?content.venue=ICLR+{year}+Conference&details=replyCount&offset=0&limit=1000&invitation=ICLR.cc/{year}/Conference/-/Blind_Submission"
            for year in range(start_year, 2026)
        ]

    def get_paper_links(self, conference_url: str) -> List[Dict[str, str]]:
        year = int(conference_url.split(".cc")[1].split("/")[1])
        return self._get_paper_links(year)


class ICMLScraper(ConferenceScraper):
    conference_name = "icml"

    def __init__(self, base_dir: str):
        super().__init__(base_dir)
        self.volume_dict = {
            2018: 80,
            2019: 97,
            2020: 119,
            2021: 139,
            2022: 162,
            2023: 202,
            2024: 235,
        }

    def get_conference_urls(self, start_year: int) -> List[str]:
        urls = []
        for year in range(start_year, 2025):
            if volume := self.volume_dict.get(year):
                urls.append(f"https://proceedings.mlr.press/v{volume}")
        return urls

    def get_paper_links(self, conference_url: str) -> List[Dict[str, str]]:
        papers = []
        try:
            response = self.session.get(conference_url)
            if response.status_code == 404:
                print(f"Conference URL not found: {conference_url}")
                return papers

            soup = BeautifulSoup(response.text, "html.parser")

            for paper in soup.find_all("div", class_="paper"):
                title_elem = paper.find("p", class_="title")
                pdf_link = paper.find("a", string="Download PDF")

                if title_elem and pdf_link:
                    volume = conference_url.split("/")[-1].lstrip("v")
                    year = next(
                        (y for y, v in self.volume_dict.items() if str(v) == volume),
                        None,
                    )
                    if year:
                        papers.append(
                            {
                                "title": title_elem.text.strip(),
                                "pdf_url": urljoin(conference_url, pdf_link["href"]),
                                "year": str(year),
                            }
                        )

        except Exception as e:
            print(f"Error scraping ICML {conference_url}: {str(e)}")
        return papers


class RLJScraper(ConferenceScraper):
    conference_name = "rlj"

    def get_conference_urls(self, start_year: int) -> List[str]:
        return [
            f"https://rlj.cs.umass.edu/{year}/{year}issue.html"
            for year in range(start_year, 2025)
        ]

    def get_paper_links(self, conference_url: str) -> List[Dict[str, str]]:
        papers = []
        try:
            response = self.session.get(conference_url)
            if response.status_code == 404:
                print(f"Conference URL not found: {conference_url}")
                return papers

            year = conference_url.split("/")[-2]  # Extract year from URL
            soup = BeautifulSoup(response.text, "html.parser")

            # Find all li elements with paper links
            for li in soup.find_all("li"):
                link = li.find("a")
                if not link or "href" not in link.attrs:
                    continue

                href = link["href"]
                if not href.startswith("papers/Paper"):
                    continue

                # Extract paper ID from href (e.g., "papers/Paper359.html" -> "359")
                paper_id = href.split("Paper")[1].split(".")[0]

                # Get the paper title and authors
                title = link.text.strip()
                authors = li.find("i")
                authors_text = authors.text.strip() if authors else ""

                # Construct the direct PDF URL
                pdf_url = f"https://rlj.cs.umass.edu/{year}/papers/RLJ_RLC_{year}_{paper_id}.pdf"

                papers.append(
                    {
                        "title": title,
                        "authors": authors_text,
                        "pdf_url": pdf_url,
                        "year": year,
                    }
                )

        except Exception as e:
            print(f"Error scraping RLJ {conference_url}: {str(e)}")
        return papers


class MLPaperProcessor:
    def __init__(self, base_dir="ml_papers"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

        self.scrapers = {
            "neurips": NeurIPSScraper(base_dir),
            "icml": ICMLScraper(base_dir),
            "iclr": ICLRScraper(base_dir),
            "rlj": RLJScraper(base_dir),
        }

    def download_paper(self, paper_info: Dict[str, str]) -> Dict[str, str]:
        """Download a single paper with rate limiting and caching."""
        try:
            year_dir = os.path.join(
                self.base_dir, paper_info["conference"], paper_info["year"]
            )
            os.makedirs(year_dir, exist_ok=True)

            safe_title = "".join(
                c for c in paper_info["title"] if c.isalnum() or c in (" ", "-", "_")
            ).rstrip()
            filename = os.path.join(year_dir, f"{safe_title[:100]}.pdf")

            if os.path.exists(filename):
                paper_info["local_path"] = filename
                paper_info["cached"] = True
                return paper_info

            response = requests.get(paper_info["pdf_url"], stream=True)
            if response.status_code != 200:
                print(
                    f"\nFailed to download {paper_info['title']}: HTTP {response.status_code}"
                )
                return None

            total_size = int(response.headers.get("content-length", 0))

            with open(filename, "wb") as f:
                with tqdm(
                    total=total_size,
                    unit="iB",
                    unit_scale=True,
                    unit_divisor=1024,
                    leave=False,
                    desc=f"Downloading {paper_info['conference']}/{safe_title[:30]}...",
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        size = f.write(chunk)
                        pbar.update(size)

            time.sleep(1)  # Rate limiting
            paper_info["local_path"] = filename
            paper_info["cached"] = False
            return paper_info

        except Exception as e:
            print(f"\nError downloading {paper_info['title']}: {str(e)}")
            return None

    def process_all_papers(self, start_year: int = 2018, conferences=None):
        """Process papers from specified conferences."""
        all_papers = []

        if conferences is None:
            conferences = self.scrapers.keys()

        for conference in conferences:
            if conference not in self.scrapers:
                print(f"Unknown conference: {conference}")
                continue

            scraper = self.scrapers[conference]
            conference_urls = scraper.get_conference_urls(start_year)

            for url in conference_urls:
                papers = scraper.get_paper_links(url)
                for paper in papers:
                    paper["conference"] = conference
                all_papers.extend(papers)
                print(f"Found {len(papers)} papers from {conference} {url}")

        if not all_papers:
            print("No papers found. Check conference URLs and try again.")
            return

        print(f"\nFound {len(all_papers)} total papers")

        downloaded_papers = []
        new_downloads = 0
        cached_papers = 0
        predownloaded = len([p for p in all_papers if paper.get("downloaded", False)])
        print(all_papers[0])
        print(all_papers[0].get("downloaded", False))

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(self.download_paper, paper)
                for paper in all_papers
                if not paper.get("downloaded", False)
            ]
            print(len(futures))
            with tqdm(
                total=len(futures),
                desc="Overall progress",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            ) as main_pbar:
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result:
                        downloaded_papers.append(result)
                        if result.get("cached", False):
                            cached_papers += 1
                        else:
                            new_downloads += 1
                    main_pbar.update(1)
                    main_pbar.set_postfix(
                        {
                            "new": new_downloads,
                            "cached": cached_papers,
                            "failed": len(futures) - new_downloads - cached_papers,
                        }
                    )

        print(f"\nDownload complete:")
        print(f"New downloads: {new_downloads}")
        print(f"Retrieved from cache: {cached_papers}")
        print(
            f"Failed downloads: {len(all_papers) - new_downloads - cached_papers - predownloaded}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ML Paper Downloader")
    parser.add_argument(
        "--start-year", type=int, default=2018, help="Start year for paper collection"
    )
    parser.add_argument(
        "--conferences",
        nargs="+",
        choices=["neurips", "icml", "iclr", "rlj"],
        help="Specific conferences to download from",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ml_papers",
        help="Directory to store downloaded papers",
    )

    args = parser.parse_args()

    processor = MLPaperProcessor(base_dir=args.output_dir)
    processor.process_all_papers(
        start_year=args.start_year, conferences=args.conferences
    )
