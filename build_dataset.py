import csv
import json
import math
import os
import random
from collections import defaultdict, deque
from collections.abc import Callable
from pathlib import Path
from time import sleep
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import urlopen

ALLOWED_DIFFICULTIES = {"<15 min fix", "15 min - 1 hour"}
MAX_IMAGE_BYTES = 2_000_000_000

# Burst metadata traffic earns HTTP 429 from Docker Hub, so retrying the same request
# after a bounded wait is the fix. Every other status stays loud.
RETRY_STATUS = 429
RETRY_AFTER_HEADER = "Retry-After"
HTTP_ATTEMPTS = 4
RETRY_BASE_DELAY_SECONDS = 1.0
RETRY_MAX_DELAY_SECONDS = 60.0

# Fields select_rows and the generated CSVs cannot work without.
REQUIRED_FIELDS = ("difficulty", "image", "repo", "instance_id")
DATASET = "SWE-bench/SWE-bench_Verified"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
CONFIG = "default"
SPLIT = "test"
EXPECTED_SOURCE_ROWS = 500
PAGE_SIZE = 100
SAMPLE_ROW_COUNT = 20
REPO_ROOT = Path(__file__).resolve().parent
SAMPLE20_PATH = REPO_ROOT / "data" / "sample20.csv"


def _finite_seconds(hint: object) -> float | None:
    """A Retry-After hint as seconds, or None unless it is a plain finite number.

    `float()` is deliberately lenient and also accepts "nan", "inf" and "1_0", none of which
    are valid delay-seconds, so those are rejected in favour of the backoff schedule.
    """
    text = str(hint).strip()
    if "_" in text:
        return None
    try:
        seconds = float(text)
    except ValueError:
        return None
    return seconds if math.isfinite(seconds) else None


def retry_delay(error: HTTPError, attempt: int) -> float:
    """Seconds to wait after `attempt` (0-based) failed: a plain finite hint, else 1s doubling."""
    hint = error.headers.get(RETRY_AFTER_HEADER) if error.headers else None
    wanted = _finite_seconds(hint)
    if wanted is None:  # absent header, an HTTP-date, or a malformed/non-finite number
        wanted = RETRY_BASE_DELAY_SECONDS * 2**attempt
    return max(0.0, min(wanted, RETRY_MAX_DELAY_SECONDS))


def http_json(url: str) -> object:
    # `sleep` is a module attribute so tests can rebind it instead of really waiting.
    for attempt in range(1, HTTP_ATTEMPTS + 1):
        try:
            with urlopen(url, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            if error.code != RETRY_STATUS or attempt == HTTP_ATTEMPTS:
                raise
            sleep(retry_delay(error, attempt - 1))


def docker_hub_tag_url(image: str) -> str:
    image = image.removeprefix("docker.io/")
    repository, tag = image.rsplit(":", 1)
    namespace, name = repository.split("/", 1)
    return (
        "https://hub.docker.com/v2/repositories/"
        f"{quote(namespace, safe='')}/{quote(name, safe='')}/tags/{quote(tag, safe='')}/"
    )


def fetch_source_rows(
    get_json: Callable[[str], object] = http_json,
    expected_rows: int = EXPECTED_SOURCE_ROWS,
    page_size: int = PAGE_SIZE,
) -> tuple[list[str], list[dict[str, object]]]:
    fieldnames: list[str] = []
    rows: list[dict[str, object]] = []
    for offset in range(0, expected_rows, page_size):
        query = urlencode(
            {
                "dataset": DATASET,
                "config": CONFIG,
                "split": SPLIT,
                "offset": offset,
                "length": min(page_size, expected_rows - offset),
            }
        )
        payload = get_json(f"{ROWS_URL}?{query}")
        if not fieldnames:
            fieldnames = [feature["name"] for feature in payload["features"]]
        rows.extend(entry["row"] for entry in payload["rows"])

    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} source rows; found {len(rows)}")
    missing = [name for name in REQUIRED_FIELDS if name not in fieldnames]
    if missing:
        raise ValueError(f"source schema is missing required fields: {', '.join(missing)}")
    seen: set[str] = set()
    duplicates: list[str] = []
    for row in rows:
        instance_id = str(row["instance_id"])
        if instance_id in seen:
            duplicates.append(instance_id)
        seen.add(instance_id)
    if duplicates:
        shown = ", ".join(sorted(set(duplicates))[:5])
        raise ValueError(f"duplicate instance_id in source rows: {shown}")
    return fieldnames, rows


def image_full_size(image: str, get_json: Callable[[str], object] = http_json) -> int | None:
    # A missing tag means the image cannot be pre-pulled; any other HTTP failure
    # is a real outage and must abort rather than silently shrink the pool. 429 is
    # retried by http_json and still aborts once that budget is spent.
    try:
        payload = get_json(docker_hub_tag_url(image))
    except HTTPError as error:
        if error.code == 404:
            return None
        raise
    return int(payload["full_size"])


def select_rows(
    rows: list[dict[str, object]],
    image_size: Callable[[str], int | None],
    count: int = SAMPLE_ROW_COUNT,
    seed: int = 42,
) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["difficulty"] not in ALLOWED_DIFFICULTIES:
            continue
        size = image_size(str(row["image"]))
        if size is None or size > MAX_IMAGE_BYTES:
            continue
        groups[str(row["repo"])].append(row)

    randomizer = random.Random(seed)
    queues: dict[str, deque[dict[str, object]]] = {}
    for repo in sorted(groups):
        randomizer.shuffle(groups[repo])
        queues[repo] = deque(groups[repo])

    selected: list[dict[str, object]] = []
    while len(selected) < count and any(queues.values()):
        for repo in sorted(queues):
            if queues[repo] and len(selected) < count:
                selected.append(queues[repo].popleft())

    if len(selected) != count:
        raise ValueError(f"need {count} eligible rows; found {len(selected)}")
    randomizer.shuffle(selected)
    return selected


def _encode_cell(value: object) -> object:
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    return value


def write_samples(
    rows: list[dict[str, object]],
    fieldnames: list[str],
    sample20: Path,
) -> None:
    """Write the sample CSV atomically."""
    sample20.parent.mkdir(parents=True, exist_ok=True)
    temp = sample20.with_name(sample20.name + ".tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({name: _encode_cell(row[name]) for name in fieldnames})
        os.replace(temp, sample20)
    finally:
        # A failure must leave the previous CSV in place and drop no stray .tmp file.
        temp.unlink(missing_ok=True)


def main() -> int:
    fieldnames, source_rows = fetch_source_rows()
    sizes: dict[str, int | None] = {}

    def memoized_size(image: str) -> int | None:
        if image not in sizes:
            sizes[image] = image_full_size(image)
        return sizes[image]

    selected = select_rows(source_rows, memoized_size, count=SAMPLE_ROW_COUNT)
    write_samples(selected, fieldnames, SAMPLE20_PATH)

    repos = {str(row["repo"]) for row in selected}
    print(f"selected {len(selected)} rows from {len(repos)} repositories")
    print(f"wrote {SAMPLE20_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
