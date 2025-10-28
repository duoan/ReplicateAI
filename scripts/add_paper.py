#!/usr/bin/env python3
import argparse
import json
import shutil
from datetime import date
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_FILE = BASE_DIR / "PAPER_INDEX.json"
TEMPLATE_DIR = BASE_DIR / "paper_template"

STAGE_MAP = {
    "foundation": "stage1_foundation",
    "representation": "stage2_representation",
    "deep": "stage3_deep_renaissance",
    "statistical": "stage4_statistical",
    "neural": "stage5_neural_origins",
}


def update_notebook_header(notebook_path, paper_title, year):
    import json
    nb = json.load(open(notebook_path))
    nb["cells"][0]["source"][0] = f"# 🧠 ReplicateAI Demo Notebook — {paper_title} ({year})\n"
    with open(notebook_path, "w") as f:
        json.dump(nb, f, indent=2)


def copy_template(template_dir, target_dir, paper_name, year):
    for item in template_dir.iterdir():
        dest = target_dir / item.name

        if item.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
            # ✅ Special handling for notebook folder
            if item.name == "notebook":
                template_nb = item / "template_notebook.ipynb"
                if template_nb.exists():
                    new_name = f"{paper_name}_demo.ipynb"
                    shutil.copy(template_nb, dest / new_name)
                    update_notebook_header(dest / new_name, paper_name, year)
            else:
                for sub in item.iterdir():
                    if sub.is_file():
                        shutil.copy(sub, dest / sub.name)
        else:
            shutil.copy(item, dest)


def add_paper(year, name, org, stage):
    stage_dir = BASE_DIR / STAGE_MAP[stage.lower()]
    paper_dir = stage_dir / f"{year}_{name.replace(' ', '')}"
    paper_dir.mkdir(parents=True, exist_ok=True)

    # ✅ copy full paper skeleton
    copy_template(TEMPLATE_DIR, paper_dir, name, year)

    # ✅ fill placeholders in README/report
    for file_name in ["README.md", "report.md"]:
        file_path = paper_dir / file_name
        if file_path.exists():
            text = file_path.read_text()
            text = (
                text.replace("{{paper_title}}", name)
                    .replace("{{year}}", str(year))
                    .replace("{{organization}}", org)
                    .replace("{{stage}}", stage.capitalize())
            )
            file_path.write_text(text)

    # ✅ update PAPER_INDEX.json (with duplicate check)
    with open(INDEX_FILE, "r") as f:
        index = json.load(f)

    papers = index.get("papers", [])

    # duplicate detection by (year, paper name)
    exists = any(
        (p["year"] == int(year) and p["paper"].lower() == name.lower())
        for p in papers
    )
    if exists:
        print(f"⚠️  Paper '{name}' ({year}) already exists in PAPER_INDEX.json — skipped.")
        return

    paper_entry = {
        "year": int(year),
        "paper": name,
        "organization": org,
        "stage": stage.capitalize() if stage != "neural" else "Neural Origins",
        "status": "planned",
        "tags": [],
        "path": str(paper_dir.relative_to(BASE_DIR)),
        "replicator": "TBD",
        "last_updated": str(date.today()),
    }

    index["papers"].append(paper_entry)
    index["meta"]["last_updated"] = str(date.today())

    with open(INDEX_FILE, "w") as f:
        json.dump(index, f, indent=2)

    print(f"✅ Added {name} ({year}) under {stage_dir}/")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add a new paper to ReplicateAI index")
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--org", required=True, type=str)
    parser.add_argument("--stage", required=True,
                        choices=["foundation", "representation", "deep", "statistical", "neural"])
    args = parser.parse_args()
    add_paper(args.year, args.name, args.org, args.stage)
