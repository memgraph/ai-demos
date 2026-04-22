"""OSHA SIC (Standard Industrial Classification) Manual scraper.

This script scrapes the hierarchical SIC code structure from OSHA's website
and generates Cypher queries to import the data into Memgraph as a tree.
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://www.osha.gov"
SIC_MANUAL_URL = f"{BASE_URL}/data/sic-manual"
REQUEST_DELAY = 0.5


@dataclass
class Industry:
    """4-digit SIC code leaf node."""

    code: str
    name: str
    description: str = ""
    examples: list[str] = field(default_factory=list)


@dataclass
class IndustryGroup:
    """3-digit SIC code."""

    code: str
    name: str
    industries: list[Industry] = field(default_factory=list)


@dataclass
class MajorGroup:
    """2-digit SIC code."""

    code: str
    name: str
    description: str = ""
    url: str = ""
    industry_groups: list[IndustryGroup] = field(default_factory=list)


@dataclass
class Division:
    """Top-level division (A-J)."""

    code: str
    name: str
    url: str = ""
    major_groups: list[MajorGroup] = field(default_factory=list)


class SICScraper:
    """Scraper for the OSHA SIC manual."""

    def __init__(self, delay: float = REQUEST_DELAY) -> None:
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "SIC-Research-Bot/1.0"
                )
            }
        )

    def _fetch(self, url: str) -> BeautifulSoup | None:
        """Fetch and parse a URL with rate limiting."""
        try:
            time.sleep(self.delay)
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.text, "html.parser")
        except requests.RequestException as error:
            print(f"Error fetching {url}: {error}")
            return None

    def scrape_main_page(self) -> list[Division]:
        """Scrape the main SIC manual page for divisions and major groups."""
        print(f"Fetching main SIC manual page: {SIC_MANUAL_URL}")
        soup = self._fetch(SIC_MANUAL_URL)
        if not soup:
            return []

        divisions: list[Division] = []
        current_division: Division | None = None

        for link in soup.find_all("a", href=True):
            href = link.get("href", "")
            text = link.get_text(strip=True)

            if "/division-" in href.lower():
                division_match = re.search(
                    r"Division\s+([A-J]):\s*(.+)",
                    text,
                    re.IGNORECASE,
                )
                if division_match:
                    code = division_match.group(1).upper()
                    name = division_match.group(2).strip()
                    current_division = Division(
                        code=code,
                        name=name,
                        url=BASE_URL + href if href.startswith("/") else href,
                    )
                    divisions.append(current_division)
                    print(f"  Found Division {code}: {name}")
            elif "/major-group-" in href.lower() and current_division:
                major_group_match = re.search(
                    r"Major\s+Group\s+(\d+):\s*(.+)",
                    text,
                    re.IGNORECASE,
                )
                if major_group_match:
                    code = major_group_match.group(1).zfill(2)
                    name = major_group_match.group(2).strip()
                    major_group = MajorGroup(
                        code=code,
                        name=name,
                        url=BASE_URL + href if href.startswith("/") else href,
                    )
                    current_division.major_groups.append(major_group)
                    print(f"    Found Major Group {code}: {name}")

        return divisions

    def scrape_major_group(self, major_group: MajorGroup) -> None:
        """Scrape a major group page for industry groups and industries."""
        if not major_group.url:
            return

        print(f"  Scraping Major Group {major_group.code}: {major_group.name}")
        soup = self._fetch(major_group.url)
        if not soup:
            return

        content = soup.find("article") or soup.find("main") or soup

        description_parts = []
        for paragraph in content.find_all("p"):
            paragraph_text = paragraph.get_text(strip=True)
            if (
                paragraph_text
                and not paragraph_text.startswith("Industry Group")
            ):
                description_parts.append(paragraph_text)
            if len(description_parts) >= 2:
                break
        major_group.description = " ".join(description_parts)

        text_content = content.get_text()
        industry_group_pattern = r"Industry\s+Group\s+(\d{3}):\s*([^\n•]+)"
        for match in re.finditer(industry_group_pattern, text_content):
            code = match.group(1)
            name = match.group(2).strip()
            major_group.industry_groups.append(
                IndustryGroup(code=code, name=name)
            )
            print(f"      Found Industry Group {code}: {name}")

        for link in content.find_all("a", href=True):
            href = link.get("href", "")
            text = link.get_text(strip=True)

            if "/sic-manual/" not in href:
                continue

            code_match = re.search(r"/sic-manual/(\d{4})$", href)
            if not code_match:
                continue

            code = code_match.group(1)
            industry_group_code = code[:3]

            for industry_group in major_group.industry_groups:
                if industry_group.code == industry_group_code:
                    industry_group.industries.append(
                        Industry(code=code, name=text)
                    )
                    print(f"        Found Industry {code}: {text}")
                    break
            else:
                industry_group = IndustryGroup(
                    code=industry_group_code,
                    name="Unknown",
                )
                industry_group.industries.append(
                    Industry(code=code, name=text)
                )
                major_group.industry_groups.append(industry_group)

    def scrape_industry(self, industry: Industry) -> None:
        """Scrape an individual industry page for detailed description."""
        url = f"{BASE_URL}/sic-manual/{industry.code}"
        print(f"          Scraping Industry {industry.code}")

        soup = self._fetch(url)
        if not soup:
            return

        content = soup.find("article") or soup.find("main") or soup

        descriptions = []
        for paragraph in content.find_all("p"):
            text = paragraph.get_text(strip=True)
            if (
                text
                and not text.startswith("Division")
                and "Industry Group" not in text
            ):
                descriptions.append(text)
        industry.description = " ".join(descriptions)

        for unordered_list in content.find_all("ul"):
            for item in unordered_list.find_all("li"):
                example = item.get_text(strip=True)
                if example:
                    industry.examples.append(example)

    def scrape_all(self, scrape_industries: bool = True) -> list[Division]:
        """Scrape the full SIC hierarchy."""
        print("Starting SIC Manual scrape...")
        print("=" * 60)

        divisions = self.scrape_main_page()
        for division in divisions:
            print(f"\nProcessing Division {division.code}: {division.name}")
            for major_group in division.major_groups:
                self.scrape_major_group(major_group)
                if scrape_industries:
                    for industry_group in major_group.industry_groups:
                        for industry in industry_group.industries:
                            self.scrape_industry(industry)

        print("\n" + "=" * 60)
        print("Scraping complete!")
        return divisions


class CypherExporter:
    """Export SIC data to Cypher queries for Memgraph."""

    @staticmethod
    def escape_string(value: str) -> str:
        """Escape special characters for Cypher strings."""
        return (
            value.replace("\\", "\\\\")
            .replace("'", "\\'")
            .replace('"', '\\"')
            .replace("\n", " ")
        )

    def generate_cypher(self, divisions: list[Division]) -> str:
        """Generate Cypher queries to create the SIC tree in Memgraph."""
        queries = [
            "CREATE INDEX ON :Division(code);",
            "CREATE INDEX ON :MajorGroup(code);",
            "CREATE INDEX ON :IndustryGroup(code);",
            "CREATE INDEX ON :Industry(code);",
            (
                "CREATE (:SICManual {name: 'Standard Industrial "
                "Classification Manual', source: 'OSHA'});"
            ),
        ]

        for division in divisions:
            name = self.escape_string(division.name)
            queries.append(
                (
                    f"CREATE (:Division {{code: '{division.code}', "
                    f"name: '{name}'}});"
                )
            )

        for division in divisions:
            queries.append(
                (
                    f"MATCH (root:SICManual), "
                    f"(d:Division {{code: '{division.code}'}}) "
                    "CREATE (root)-[:HAS_DIVISION]->(d);"
                )
            )

        for division in divisions:
            for major_group in division.major_groups:
                name = self.escape_string(major_group.name)
                description = self.escape_string(
                    (
                        major_group.description[:500]
                        if major_group.description
                        else ""
                    )
                )
                queries.append(
                    (
                        f"CREATE (:MajorGroup {{code: '{major_group.code}', "
                        f"name: '{name}', "
                        f"description: '{description}'}});"
                    )
                )

        for division in divisions:
            for major_group in division.major_groups:
                queries.append(
                    f"MATCH (d:Division {{code: '{division.code}'}}), "
                    f"(mg:MajorGroup {{code: '{major_group.code}'}}) "
                    "CREATE (d)-[:HAS_MAJOR_GROUP]->(mg);"
                )

        for division in divisions:
            for major_group in division.major_groups:
                for industry_group in major_group.industry_groups:
                    name = self.escape_string(industry_group.name)
                    queries.append(
                        (
                            f"CREATE (:IndustryGroup {{code: "
                            f"'{industry_group.code}', "
                            f"name: '{name}'}});"
                        )
                    )

        for division in divisions:
            for major_group in division.major_groups:
                for industry_group in major_group.industry_groups:
                    queries.append(
                        (
                            f"MATCH (mg:MajorGroup {{code: "
                            f"'{major_group.code}'}}), "
                            f"(ig:IndustryGroup {{code: "
                            f"'{industry_group.code}'}}) "
                            "CREATE (mg)-[:HAS_INDUSTRY_GROUP]->(ig);"
                        )
                    )

        for division in divisions:
            for major_group in division.major_groups:
                for industry_group in major_group.industry_groups:
                    for industry in industry_group.industries:
                        name = self.escape_string(industry.name)
                        description = self.escape_string(
                            (
                                industry.description[:500]
                                if industry.description
                                else ""
                            )
                        )
                        examples = (
                            json.dumps(industry.examples[:10])
                            if industry.examples
                            else "[]"
                        )
                        queries.append(
                            f"CREATE (:Industry {{code: '{industry.code}', "
                            f"name: '{name}', description: '{description}', "
                            f"examples: {examples}}});"
                        )

        for division in divisions:
            for major_group in division.major_groups:
                for industry_group in major_group.industry_groups:
                    for industry in industry_group.industries:
                        queries.append(
                            (
                                f"MATCH (ig:IndustryGroup {{code: "
                                f"'{industry_group.code}'}}), "
                                f"(i:Industry {{code: '{industry.code}'}}) "
                                "CREATE (ig)-[:HAS_INDUSTRY]->(i);"
                            )
                        )

        return "\n".join(queries)


def export_to_json(divisions: list[Division], filepath: str) -> None:
    """Export scraped data to JSON format."""
    data = []
    for division in divisions:
        division_data = {
            "type": "Division",
            "code": division.code,
            "name": division.name,
            "major_groups": [],
        }
        for major_group in division.major_groups:
            major_group_data = {
                "type": "MajorGroup",
                "code": major_group.code,
                "name": major_group.name,
                "description": major_group.description,
                "industry_groups": [],
            }
            for industry_group in major_group.industry_groups:
                industry_group_data = {
                    "type": "IndustryGroup",
                    "code": industry_group.code,
                    "name": industry_group.name,
                    "industries": [],
                }
                for industry in industry_group.industries:
                    industry_group_data["industries"].append(
                        {
                            "type": "Industry",
                            "code": industry.code,
                            "name": industry.name,
                            "description": industry.description,
                            "examples": industry.examples,
                        }
                    )
                major_group_data["industry_groups"].append(industry_group_data)
            division_data["major_groups"].append(major_group_data)
        data.append(division_data)

    with open(filepath, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    print(f"Exported data to {filepath}")


def print_statistics(divisions: list[Division]) -> None:
    """Print statistics about the scraped data."""
    total_divisions = len(divisions)
    total_major_groups = sum(
        len(division.major_groups) for division in divisions
    )
    total_industry_groups = sum(
        len(major_group.industry_groups)
        for division in divisions
        for major_group in division.major_groups
    )
    total_industries = sum(
        len(industry_group.industries)
        for division in divisions
        for major_group in division.major_groups
        for industry_group in major_group.industry_groups
    )

    print("\n" + "=" * 60)
    print("SIC Manual Statistics:")
    print("=" * 60)
    print(f"  Divisions:        {total_divisions}")
    print(f"  Major Groups:     {total_major_groups}")
    print(f"  Industry Groups:  {total_industry_groups}")
    print(f"  Industries:       {total_industries}")
    total_nodes = (
        total_divisions
        + total_major_groups
        + total_industry_groups
        + total_industries
        + 1
    )
    print(f"  Total Nodes:      {total_nodes}")
    print("=" * 60)


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape the OSHA SIC manual and export it for Memgraph"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="./output",
        help="Output directory for generated files",
    )
    parser.add_argument(
        "--no-industry-details",
        action="store_true",
        help="Skip scraping individual industry pages",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=REQUEST_DELAY,
        help=f"Delay between requests in seconds (default: {REQUEST_DELAY})",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scraper = SICScraper(delay=args.delay)
    divisions = scraper.scrape_all(
        scrape_industries=not args.no_industry_details
    )
    print_statistics(divisions)

    json_path = output_dir / "sic_data.json"
    export_to_json(divisions, str(json_path))

    cypher_path = output_dir / "sic_import.cypherl"
    cypher = CypherExporter().generate_cypher(divisions)
    with open(cypher_path, "w", encoding="utf-8") as handle:
        handle.write(cypher)
    print(f"Exported Cypher queries to {cypher_path}")

    print("\n" + "=" * 60)
    print("To import into Memgraph:")
    print("=" * 60)
    print("  1. Start Memgraph")
    print(f"  2. Run: mgconsole < {cypher_path}")
    print("  3. Run the embedding generator")
    print("=" * 60)


if __name__ == "__main__":
    main()
