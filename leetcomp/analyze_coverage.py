import json
import tomllib
from datetime import datetime, timedelta
from pathlib import Path
import argparse

# --- Configuration ---
# Load config to get the data directory and date format
with open("config.toml", "rb") as f:
    config = tomllib.load(f)

DATA_DIR = Path(config["app"]["data_dir"])
DATE_FMT = config["app"]["date_fmt"]
# We'll consider a gap significant if it's more than this many days
GAP_THRESHOLD_DAYS = 2


def analyze_coverage(comps_path: Path) -> None:
    """
    Analyzes the date coverage of the raw compensation data file.
    """
    if not comps_path.exists():
        print(f"Error: The data file was not found at '{comps_path}'")
        print("Please run the refresh.py script first to generate the data.")
        return

    print(f"Analyzing data coverage in '{comps_path}'...\n")

    timestamps = []
    try:
        with open(comps_path, "r") as f:
            for line in f:
                try:
                    post = json.loads(line)
                    timestamps.append(datetime.strptime(post["creation_date"], DATE_FMT))
                except (json.JSONDecodeError, KeyError):
                    print(f"Warning: Skipping a malformed line in the data file.")
                    continue
    except Exception as e:
        print(f"Error reading the file: {e}")
        return

    if not timestamps:
        print("The data file is empty. No coverage to analyze.")
        return

    # Sort from oldest to newest to find gaps
    timestamps.sort()

    total_posts = len(timestamps)
    oldest_post = timestamps[0]
    newest_post = timestamps[-1]
    total_duration = newest_post - oldest_post

    print("--- Data Coverage Analysis ---")
    print(f"Total Posts Found: {total_posts}")
    print(f"Oldest Post:       {oldest_post.strftime('%Y-%m-%d')}")
    print(f"Newest Post:       {newest_post.strftime('%Y-%m-%d')}")
    print(f"Total Timespan:    {total_duration.days} days")
    print("-" * 30)

    # --- Gap Detection ---
    gaps = []
    gap_threshold = timedelta(days=GAP_THRESHOLD_DAYS)

    for i in range(1, len(timestamps)):
        delta = timestamps[i] - timestamps[i-1]
        if delta > gap_threshold:
            gaps.append({
                "start": timestamps[i-1],
                "end": timestamps[i],
                "duration": delta
            })

    if not gaps:
        print("✅ No significant gaps found.")
        print(f"(A 'gap' is defined as more than {GAP_THRESHOLD_DAYS} days between consecutive posts)")
    else:
        print(f"⚠️ Found {len(gaps)} potential gaps (>{GAP_THRESHOLD_DAYS} days between posts):\n")
        for gap in gaps:
            start_str = gap['start'].strftime('%Y-%m-%d')
            end_str = gap['end'].strftime('%Y-%m-%d')
            print(f"  - Gap of {gap['duration'].days} days found between {start_str} and {end_str}")

    print("\n--- Analysis Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze the date coverage of the raw LeetCode compensation data."
    )
    parser.add_argument(
        "--comps_path",
        type=Path,
        default=DATA_DIR / "raw_comps.jsonl",
        help="Path to the raw compensation data file (e.g., data/raw_comps.jsonl).",
    )
    args = parser.parse_args()
    
    analyze_coverage(args.comps_path)