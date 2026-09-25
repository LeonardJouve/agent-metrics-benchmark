import asyncio
import csv
import os
import sys
from pathlib import Path

from daytona import AsyncDaytona, DaytonaConfig


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))

async def create_sandbox(row: dict[str, str], config: DaytonaConfig) -> bool:
    task_id = row.get("instance_id", "<unknown>")
    try:
        daytona = AsyncDaytona(config)
        await daytona.create()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return False
    finally:
        await daytona.close()
    print(f"OK {task_id}")
    return True

async def run_benchmark(
    rows: list[dict[str, str]],
    batch_size: int,
    config: DaytonaConfig
) -> tuple[int, int]:
    semaphore = asyncio.Semaphore(batch_size)

    for row in rows:
        async with semaphore:
            await create_sandbox(row, config)

    results = await asyncio.gather(*(create_sandbox(row) for row in rows))
    succeeded = sum(results)

    return succeeded, len(results) - succeeded

def main() -> int:
    try:
        dataset_path = Path(os.environ.get("DATASET_PATH"))
        batch_size = os.environ.get("BATCH_SIZE")
        config = DaytonaConfig(api_key=os.environ.get("DAYTONA_API_KEY"))
        rows = load_rows(dataset_path)
        run_benchmark(rows, batch_size, config)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
