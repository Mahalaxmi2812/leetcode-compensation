# Project Context

- **Generation Date:** 2025-07-31 18:36:24
- **Root Directory:** `leetcomp`

---

## Project Structure

leetcomp/
    - __init__.py
    - analyze_coverage.py
    - errors.py
    - parse.py
    - prompts.py
    - queries.py
    - refresh.py
    - santise.py
    - utils.py

---

## File Contents


---
**File:** `__init__.py`
---

```py





---
**File:** `analyze_coverage.py`
---

```py
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




---
**File:** `errors.py`
---

```py
class FetchPostsException(Exception):
    pass


class FetchContentException(Exception):
    pass





---
**File:** `parse.py`
---

```py
import json
import os
import re
from datetime import datetime
from typing import Any, Generator

from leetcomp.prompts import PARSING_PROMPT
from leetcomp.utils import (
    config,
    get_model_predict,
    latest_parsed_date,
    mapping,
    parse_json_markdown,
    sort_and_truncate,
)

interview_exp_pattern = re.compile(
    r"https://leetcode.com/discuss/interview-experience/\S+"
)

llm_predict = get_model_predict(config["app"]["llm_predictor"])

yoe_map: dict[tuple[int, int], str] = {
    (0, 1): "Entry (0-1)",
    (2, 6): "Mid (2-6)",
    (7, 10): "Senior (7-10)",
    (11, 30): "Senior + (11+)",
}


def post_should_be_parsed(post: dict[Any, Any]) -> bool:
    if "title" not in post:
        print(f" x skipping {post['id']}; no title")
        return False
    if "|" not in post["title"]:
        print(f" x skipping {post['id']}; | not in title")
        return False
    if "vote_count" not in post:
        print(f" x skipping {post['id']}; no vote_count")
        return False
    if post["vote_count"] < 0:
        print(f" x skipping {post['id']}; negative vote_count")
        return False
    return True


def has_crossed_till_date(
    creation_date: str, till_date: datetime | None = None
) -> bool:
    if till_date is None:
        return False

    dt = datetime.strptime(creation_date, config["app"]["date_fmt"])
    return dt <= till_date


def comps_posts_iter(comps_path: str) -> Generator[dict[Any, Any], None, None]:
    with open(comps_path, "r") as f:
        for line in f:
            yield json.loads(line)


def parsed_content_is_valid(
    post_id: str, parsed_content: list[dict[Any, Any]]
) -> bool:
    if not isinstance(parsed_content, list) or not parsed_content:
        return False

    for item in parsed_content:
        try:
            assert isinstance(item, dict), "item is not a dict"

            assert isinstance(
                item["base_offer"], (int, float)
            ), "base_offer is not a number"

            assert (
                config["parsing"]["min_base_offer"]
                <= item["base_offer"]
                <= config["parsing"]["max_base_offer"]
            ), "base_offer out of range"

            assert isinstance(
                item["total_offer"], (int, float)
            ), "total_offer is not a number"

            assert (
                config["parsing"]["min_total_offer"]
                <= item["total_offer"]
                <= config["parsing"]["max_total_offer"]
            ), "total_offer out of range"

            assert isinstance(item["company"], str), "company is not a string"
            assert isinstance(item["role"], str), "role is not a string"
            assert isinstance(item["yoe"], (int, float)), "yoe is not a number"

            if "non_indian" in item:
                assert item["non_indian"] != "yes", "non_indian is yes"

            # offers as amounts are per month, need a modified prompt for these
            assert "intern" not in item["role"].lower(), "intern in role"
        except (KeyError, AssertionError) as e:
            print(f" x skipping {post_id}; invalid content: {str(e)}")
            return False

    return True  # Parsed content is valid if no assertions fail


def extract_interview_exp(content: str) -> str:
    match = interview_exp_pattern.search(content)
    return match.group(0) if match else "N/A"


def get_parsed_posts(
    raw_post: dict[Any, Any], parsed_content: list[dict[Any, Any]]
) -> list[dict[Any, Any]]:
    return [
        {
            "id": raw_post["id"],
            "vote_count": raw_post["vote_count"],
            "comment_count": raw_post["comment_count"],
            "view_count": raw_post["view_count"],
            "creation_date": raw_post["creation_date"],
            "company": item["company"],
            "role": item["role"],
            "yoe": item["yoe"],
            "base_offer": item["base_offer"],
            "total_offer": item["total_offer"],
            "location": item.get("location", "n/a"),
            "interview_exp": extract_interview_exp(raw_post["content"]),
        }
        for item in parsed_content
    ]


def fill_yoe(parsed_content: list[dict[Any, Any]]) -> None:
    if len(parsed_content) > 1:
        for item in parsed_content[1:]:
            item["yoe"] = parsed_content[0]["yoe"]


def parse_posts(
    in_comps_path: str,
    out_comps_path: str,
    parsed_ids: set[int] | None = None,
    till_date: datetime | None = None,
) -> None:
    n_skips = 0
    parsed_ids = parsed_ids or set()

    for i, post in enumerate(comps_posts_iter(in_comps_path), start=1):
        if i % 20 == 0:
            print(f"Processed {i} posts; {n_skips} skips")

        if post["id"] in parsed_ids or not post_should_be_parsed(post):
            n_skips += 1
            continue

        if has_crossed_till_date(post["creation_date"], till_date):
            break

        input_text = f"{post['title']}\n---\n{post['content']}"
        prompt = PARSING_PROMPT.substitute(leetcode_post=input_text)
        response = llm_predict(prompt)
        parsed_content = parse_json_markdown(response)

        if parsed_content_is_valid(post["id"], parsed_content):
            fill_yoe(parsed_content)
            parsed_posts = get_parsed_posts(post, parsed_content)
            with open(out_comps_path, "a") as f:
                for parsed_post in parsed_posts:
                    f.write(json.dumps(parsed_post) + "\n")
        else:
            n_skips += 1


def get_parsed_ids(out_comps_path: str) -> set[int]:
    with open(out_comps_path, "r") as f:
        return {json.loads(line)["id"] for line in f}


def cleanup_record(record: dict[Any, Any]) -> None:
    record.pop("vote_count", None)
    record.pop("comment_count", None)
    record.pop("view_count", None)

    record["creation_date"] = record["creation_date"][:10]
    record["yoe"] = round(record["yoe"])
    record["base"] = round(float(record["base_offer"]), 2)
    record["total"] = round(float(record["total_offer"]), 2)

    record.pop("base_offer", None)
    record.pop("total_offer", None)


def mapped_record(
    item: str,
    mapping: dict[str, str],
    default: str | None = None,
    extras: list[str] | None = None,
) -> str:
    item = item.lower()
    if extras:
        for role_str in extras:
            if role_str in item:
                return role_str.capitalize()

    return mapping.get(item, default or item.capitalize())


def map_location(location: str, location_map: dict[str, str]) -> str:
    location = location.lower()

    if location == "n/a":
        return location_map[location]

    if "(" in location:
        location = location.split("(")[0].strip()

    for sep in [",", "/"]:
        if sep in location:
            locations = [loc.strip().lower() for loc in location.split(sep)]
            location = "/".join(
                [location_map.get(loc, loc.capitalize()) for loc in locations]
            )
            return location

    return location_map.get(location, location.capitalize())


def map_yoe(yoe: int, yoe_map: dict[tuple[int, int], str]) -> str:
    for (start, end), mapped_yoe in yoe_map.items():
        if start <= yoe <= end:
            return mapped_yoe

    return "Senior +"


def jsonl_to_json(jsonl_path: str, json_path: str) -> None:
    company_map = mapping(config["app"]["data_dir"] / "company_map.json")
    role_map = mapping(config["app"]["data_dir"] / "role_map.json")
    location_map = mapping(config["app"]["data_dir"] / "location_map.json")
    records = []

    with open(jsonl_path, "r") as file:
        for line in file:
            record = json.loads(line)
            cleanup_record(record)
            record["company"] = mapped_record(record["company"], company_map)
            role_to_map = "".join(re.findall(r"\w+", record["role"]))
            record["mapped_role"] = mapped_record(
                role_to_map,
                role_map,
                default=record["role"],
                extras=["analyst", "intern", "associate"],
            )
            record["mapped_yoe"] = map_yoe(record["yoe"], yoe_map)
            record["location"] = map_location(record["location"], location_map)
            records.append(record)

    with open(json_path, "w") as file:
        json.dump(records, file, indent=4)

    print(f"Converted {len(records)} records!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse LeetCode Compensations posts."
    )
    parser.add_argument(
        "--in_comps_path",
        type=str,
        default=config["app"]["data_dir"] / "raw_comps.jsonl",
        help="Path to the file to store posts.",
    )
    parser.add_argument(
        "--out_comps_path",
        type=str,
        default=config["app"]["data_dir"] / "parsed_comps.jsonl",
        help="Path to the file to store parsed posts.",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default=config["app"]["data_dir"] / "parsed_comps.json",
        help="Path to the file to store parsed posts in JSON format.",
    )
    args = parser.parse_args()

    print(f"Parsing comps from {args.in_comps_path}...")

    parsed_ids = (
        get_parsed_ids(args.out_comps_path)
        if os.path.exists(args.out_comps_path)
        else set()
    )
    print(f"Found {len(parsed_ids)} parsed ids...")

    till_date = (
        latest_parsed_date(args.out_comps_path)
        if os.path.exists(args.out_comps_path)
        else None
    )

    parse_posts(args.in_comps_path, args.out_comps_path, parsed_ids, till_date)
    sort_and_truncate(args.out_comps_path, truncate=True)
    jsonl_to_json(args.out_comps_path, args.json_path)





---
**File:** `prompts.py`
---

```py
from string import Template

PARSING_PROMPT = Template("""
You are a helpful assistant tasked with extracting job offer details from posts on LeetCode.
Below is the expected output format.

## Output Format
The output should be a JSON array containing one or more dictionaries, each representing a job offer.
Each dictionary must include the following keys with their respective values:
- company (str): The name of the company offering the job.
- role (str): The job title or role being offered.
- yoe (float): The years of experience of the candidate receiving the offer.
- base_offer (float): The base salary component of the offer, in LPA (INR).
  For example, "29.5 LPA" should be represented as 29.5. Avoid scientific notation.
  Salary in crores like "1.18 cr" should be converted to LPA (1.18 * 100 = 118).
- total_offer (float): The total compensation offered, in LPA (INR).
  For example, "3130000" should be represented as 31.3. Avoid scientific notation.
  Salary in crores like "1.18 cr" should be converted to LPA (1.18 * 100 = 118).
- location (str): The location of the job. Only output the city name (e.g., "Bangalore" instead of "Bangalore, India").
- non_indian (optional str): If the post mentions a location outside of india or the currency is not in INR set this key to "yes"; otherwise omit this key.

## Instructions
- If a key is not present in the post, set its value to "n/a".
- For posts with multiple job offers, include a dictionary for each offer.
- Sometimes users metion details about their current role and salary, ignore these details.

Your goal is to parse the content of the post below and structure the information into a specified JSON format by
following the "Output Format" and "Instructions" mentioned above.

## Post
$leetcode_post

## Parsed Job Offer (Output the JSON inside triple backticks (```). The format is [{...}, {...}, ...])
""")

COMPANY_CLUSTER_PROMPT = Template("""
Given this list of companies, cluster the same companies together and assign a relevant name to the cluster.
Avoid using content within parentheses or brackets for the company cluster name.

Your output format should be [{"cluster_name": ..., "companies": [...]}, ...].

## Companies
$companies

## Output (Output the JSON inside triple backticks (```). Again, the output format is [{"cluster_name": ..., "companies": [...]}, ...])
""")

ROLE_CLUSTER_PROMPT = Template("""
Given this list of roles, cluster the same roles together and assign a relevant name to the cluster.
For example, sde-i, sde1, sde 1, senior engineer should all be clustered together as "SDE I",
similarly sde2, sde-ii, sde 2, senior software engineer should all be clustered together as "SDE II".
Avoid using content within parentheses or brackets for cluster name.

Your output format should be [{"cluster_name": ..., "roles": [...]}, ...].

## Roles
$roles

## Output (Output the JSON inside triple backticks (```). Again, the output format is [{"cluster_name": ..., "roles": [...]}, ...])
""")





---
**File:** `queries.py`
---

```py
COMP_POSTS_QUERY = """
query categoryTopicList($categories: [String!]!, $first: Int!, $orderBy: TopicSortingOption, $skip: Int, $query: String, $tags: [String!]) {
  categoryTopicList(categories: $categories, orderBy: $orderBy, skip: $skip, query: $query, first: $first, tags: $tags) {
    ...TopicsList
    __typename
  }
}

fragment TopicsList on TopicConnection {
  totalNum
  edges {
    node {
      id
      title
      commentCount
      viewCount
      pinned
      tags {
        name
        slug
        __typename
      }
      post {
        id
        voteCount
        creationDate
        isHidden
        author {
          username
          isActive
          nameColor
          activeBadge {
            displayName
            icon
            __typename
          }
          profile {
            userAvatar
            __typename
          }
          __typename
        }
        status
        coinRewards {
          ...CoinReward
          __typename
        }
        __typename
      }
      lastComment {
        id
        post {
          id
          author {
            isActive
            username
            __typename
          }
          peek
          creationDate
          __typename
        }
        __typename
      }
      __typename
    }
    cursor
    __typename
  }
  __typename
}

fragment CoinReward on ScoreNode {
  id
  score
  description
  date
  __typename
}
"""


COMP_POSTS_DATA_QUERY = {
    "operationName": "categoryTopicList",
    "query": COMP_POSTS_QUERY,
    "variables": {
        "orderBy": "newest_to_oldest",
        "query": "",
        "skip": 0,
        "first": 50,
        "tags": [],
        "categories": ["compensation"],
    },
}


COMP_POST_CONTENT_QUERY = """
query DiscussTopic($topicId: Int!) {
  topic(id: $topicId) {
    id
    viewCount
    topLevelCommentCount
    subscribed
    title
    pinned
    tags
    hideFromTrending
    post {
      ...DiscussPost
      __typename
    }
    __typename
  }
}

fragment DiscussPost on PostNode {
  id
  voteCount
  voteStatus
  content
  updationDate
  creationDate
  status
  isHidden
  coinRewards {
    ...CoinReward
    __typename
  }
  author {
    isDiscussAdmin
    isDiscussStaff
    username
    nameColor
    activeBadge {
      displayName
      icon
      __typename
    }
    profile {
      userAvatar
      reputation
      __typename
    }
    isActive
    __typename
  }
  authorIsModerator
  isOwnPost
  __typename
}

fragment CoinReward on ScoreNode {
  id
  score
  description
  date
  __typename
}
"""


COMP_POST_CONTENT_DATA_QUERY = {
    "operationName": "DiscussTopic",
    "query": COMP_POST_CONTENT_QUERY,
    "variables": {"topicId": 0},
}





---
**File:** `refresh.py`
---

```py
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Iterator

import requests  # type: ignore

from leetcomp.errors import FetchContentException, FetchPostsException
from leetcomp.queries import COMP_POST_CONTENT_DATA_QUERY as content_query
from leetcomp.queries import COMP_POSTS_DATA_QUERY as posts_query
from leetcomp.utils import (
    config,
    latest_parsed_date,
    retry_with_exp_backoff,
    sort_and_truncate,
)


@dataclass
class LeetCodePost:
    id: str
    title: str
    content: str
    vote_count: int
    comment_count: int
    view_count: int
    creation_date: datetime


def get_posts_query(skip: int, first: int) -> dict[Any, Any]:
    query = posts_query.copy()
    query["variables"]["skip"] = skip  # type: ignore
    query["variables"]["first"] = first  # type: ignore
    return query


def get_content_query(post_id: int) -> dict[Any, Any]:
    query = content_query.copy()
    query["variables"]["topicId"] = post_id  # type: ignore
    return query


@retry_with_exp_backoff(retries=config["app"]["n_api_retries"])  # type: ignore
def post_content(post_id: int) -> str:
    query = get_content_query(post_id)
    response = requests.post(config["app"]["leetcode_graphql_url"], json=query)

    if response.status_code != 200:
        raise FetchContentException(
            f"Failed to fetch content for post_id={post_id}): {response.text}"
        )

    data = response.json().get("data")
    if not data:
        raise FetchContentException(
            f"Invalid response data for post_id={post_id}"
        )
    
    if not data or not data.get("topic"):
        print(f" ~ Skipping post_id={post_id}; content is null or missing (likely deleted).")
        return ""

    return str(data["topic"]["post"]["content"])


@retry_with_exp_backoff(retries=config["app"]["n_api_retries"])  # type: ignore
def parsed_posts(skip: int, first: int) -> Iterator[LeetCodePost]:
    query = get_posts_query(skip, first)
    response = requests.post(config["app"]["leetcode_graphql_url"], json=query)

    if response.status_code != 200:
        raise FetchPostsException(
            f"Failed to fetch content for skip={skip}, first={first}): {response.text}"
        )

    data = response.json().get("data")
    if not data:
        raise FetchPostsException(
            f"Invalid response data for skip={skip}, first={first}"
        )

    posts = data["categoryTopicList"]["edges"]

    if skip == 0:
        posts = posts[1:]  # Skip pinned post

    for post in posts:
        yield LeetCodePost(
            id=post["node"]["id"],
            title=post["node"]["title"],
            content=str(post_content(post["node"]["id"])),
            vote_count=post["node"]["post"]["voteCount"],
            comment_count=post["node"]["commentCount"],
            view_count=post["node"]["viewCount"],
            creation_date=datetime.fromtimestamp(
                post["node"]["post"]["creationDate"]
            ),
        )


def get_latest_posts(
    comps_path: str, start_date: datetime, till_date: datetime, start_skip: int = 0
) -> None:
    skip, first = start_skip, 50
    has_crossed_till_date = False
    fetched_posts, skips_due_to_lag = 0, 0

    with open(comps_path, "a") as f:
        while not has_crossed_till_date:
            for post in parsed_posts(skip, first):  # type: ignore[unused-ignore]
                if post.creation_date > start_date:
                    skips_due_to_lag += 1
                    continue

                if post.creation_date <= till_date:
                    has_crossed_till_date = True
                    break

                post_dict = asdict(post)
                post_dict["creation_date"] = post.creation_date.strftime(
                    config["app"]["date_fmt"]
                )
                f.write(json.dumps(post_dict) + "\n")
                fetched_posts += 1

                if fetched_posts % 10 == 0:
                    print(
                        f"{post.creation_date} Fetched {fetched_posts} posts..."
                    )

            skip += first

        print(f"Skipped {skips_due_to_lag} posts due to lag...")
        print(f"{post.creation_date} Fetched {fetched_posts} posts in total!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch latest posts from LeetCode Compensations page."
    )
    parser.add_argument(
        "--comps_path",
        type=str,
        default=config["app"]["data_dir"] / "raw_comps.jsonl",
        help="Path to the file to store posts.",
    )
    parser.add_argument(
        "--till_date",
        type=str,
        default="",
        help="Fetch posts till this date (YYYY/MM/DD).",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Number of posts to skip from the beginning.",
    )
    args = parser.parse_args()

    if not args.till_date:
        till_date = latest_parsed_date(args.comps_path)
    else:
        till_date = datetime.strptime(args.till_date, "%Y/%m/%d")
    


    print(f"Fetching posts till {till_date}...")

    start_date = datetime.now() - timedelta(days=config["app"]["lag_days"])
    get_latest_posts(args.comps_path, start_date, till_date, start_skip=args.skip)
    sort_and_truncate(args.comps_path, truncate=False)





---
**File:** `santise.py`
---

```py
import json
from typing import Any

from leetcomp.prompts import COMPANY_CLUSTER_PROMPT, ROLE_CLUSTER_PROMPT
from leetcomp.utils import config


def cluster_companies_prompt(records: list[dict[Any, Any]]) -> str:
    companies = [r["company"].lower() for r in records if r["company"].strip()]
    unique_companies = "\n".join(sorted(set(companies)))
    prompt = COMPANY_CLUSTER_PROMPT.substitute(companies=unique_companies)
    return prompt


def cluster_roles_prompt(records: list[dict[Any, Any]]) -> str:
    roles = [r["role"].lower() for r in records if r["role"].strip()]
    unique_roles = "\n".join(sorted(set(roles)))
    prompt = ROLE_CLUSTER_PROMPT.substitute(roles=unique_roles)
    return prompt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sanitise parsed LeetCode Compensations posts."
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default=config["app"]["data_dir"] / "parsed_comps.json",
        help="Path to the file where parsed posts are stored in JSON format.",
    )
    args = parser.parse_args()

    with open(args.json_path, "r") as f:
        records = json.load(f)

    cluster_roles_prompt_ = cluster_roles_prompt(records)
    print(cluster_roles_prompt_)





---
**File:** `utils.py`
---

```py
import google.generativeai as genai
import json
import os
import random
import re
import time
import tomllib
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import ollama
import requests  # type: ignore
from dotenv import load_dotenv

load_dotenv(override=True)


with open("config.toml", "rb") as f:
    config = tomllib.load(f)

config["app"]["data_dir"] = Path(config["app"]["data_dir"])


def ollama_predict(prompt: str) -> str:
    response = ollama.chat(
        model=config["llms"]["ollama_model"],
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]  # type: ignore


def openrouter_predict(prompt: str) -> str:
    response = requests.post(
        url=config["llms"]["openrouter_url"],
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
        data=json.dumps(
            {
                "model": config["llms"]["openrouter_model"],
                "messages": [{"role": "user", "content": prompt}],
            }
        ),
    )
    time.sleep(60 / config["llms"]["openrouter_req_per_min"])
    return str(response.json()["choices"][0]["message"]["content"])


def vllm_predict(prompt: str) -> str:
    response = requests.post(
        config["llms"]["vllm_url"],
        json={
            "model": config["llms"]["vllm_model"],
            "prompt": prompt,
            "max_tokens": 256,
            "temperature": 0.3,
        },
    )
    return str(response.json()["choices"][0]["text"])

def gemini_predict(prompt: str) -> str:
    """
    Sends a prompt to the Google Gemini API and returns the response.
    """
    try:
        # Configure the API key from environment variables
        genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
        
        # Use the latest Gemini 1.5 Pro model
        model = genai.GenerativeModel(config["llms"]["gemini_model"])
        
        # Set safety settings to be less restrictive for this specific use case
        safety_settings = {
            'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
            'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
            'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
            'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
        }

        response = model.generate_content(prompt, safety_settings=safety_settings)
        
        # The API can sometimes return an empty response if the prompt is blocked,
        # so we check for that.
        if not response.parts:
            print(" ~ Gemini API returned a blocked response. Returning empty string.")
            return ""

        time.sleep(60 / config["llms"]["gemini_req_per_min"])

        return response.text

    except Exception as e:
        print(f"An error occurred with the Gemini API: {e}")
        return "" # Return empty string on error to prevent crashing

def get_model_predict(inf_engine: str) -> Callable[[str], str]:
    match inf_engine.lower():
        case "ollama":
            return ollama_predict
        case "openrouter":
            return openrouter_predict
        case "vllm":
            return vllm_predict
        case "gemini":
            return gemini_predict
        case _:
            raise ValueError("Invalid inference engine")


def retry_with_exp_backoff(retries: int):  # type: ignore[no-untyped-def]
    def decorator(function: Callable):  # type: ignore
        @wraps(function)
        def wrapper(*args, **kwargs):  # type: ignore
            i = 1
            while i <= retries:
                try:
                    return function(*args, **kwargs)
                except Exception as e:
                    sleep_for = random.uniform(2**i, 2 ** (i + 1))
                    err_msg = f"{function.__name__} ({args}, {kwargs}): {e}"
                    print(f"Retry={i} Rest={sleep_for:.1f}s Err={err_msg}")
                    time.sleep(sleep_for)
                    i += 1
                    if i > retries:
                        raise

        return wrapper

    return decorator


def latest_parsed_date(comps_path: str) -> datetime:
    with open(comps_path, "r") as f:
        top_line = json.loads(f.readline())

    return datetime.strptime(
        top_line["creation_date"], config["app"]["date_fmt"]
    )


def parse_json_markdown(json_string: str) -> list[dict[Any, Any]]:
    match = re.search(
        r"""```    # match first occuring triple backticks
        (?:json)?  # zero or one match of string json in non-capturing group
        (.*)```    # greedy match to last triple backticks
        """,
        json_string,
        flags=re.DOTALL | re.VERBOSE,
    )

    if match is None:
        json_str = json_string
    else:
        json_str = match.group(1)

    json_str = json_str.strip()
    try:
        parsed_content = eval(json_str)
    except Exception:
        return []

    return parsed_content  # type: ignore


def sort_and_truncate(comps_path: str, truncate: bool = False) -> None:
    with open(comps_path, "r") as file:
        records = [json.loads(line) for line in file]

    records.sort(
        key=lambda x: datetime.strptime(
            x["creation_date"], config["app"]["date_fmt"]
        ),
        reverse=True,
    )

    if truncate:
        records = records[: config["app"]["max_recs"]]

    with open(comps_path, "w") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")

    print(f"Sorted {len(records)} records!")


def mapping(map_path: str | Path) -> dict[str, str]:
    with open(map_path, "r") as f:
        data = json.load(f)

    mapping = {}
    for d in data:
        for item in d["cluster"]:
            mapping[item] = d["cluster_name"]

    return mapping



