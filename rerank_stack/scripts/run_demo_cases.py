#!/usr/bin/env python
import argparse
import asyncio
from typing import List, Tuple, Set

from rerank_demo.search import search_and_rerank


DEMO_CASES: List[Tuple[str, str, Set[int], str]] = [
    ("shower", "how to fix a leaky shower", {0, 2}, "howto_plumbing_v1"),
    ("car", "car maintenance", {1, 5, 9}, "howto_automotive_v1"),
    ("bicycle", "bicycle repair", {3, 7}, "howto_bicycle_v1"),
    ("ambiguous", "maintenance", {3, 5, 6, 8}, "generic_v1"),
    ("faucet", "fix leaking faucet", {4}, "howto_plumbing_v1"),
]


def pick_cases(name: str | None) -> List[Tuple[str, str, Set[int], str]]:
    if not name:
        return DEMO_CASES
    lowered = name.lower()
    matches = [case for case in DEMO_CASES if case[0] == lowered]
    if not matches:
        raise ValueError(f"Unknown demo case '{name}'. Available: {[case[0] for case in DEMO_CASES]}")
    return matches


async def run_case(name: str, query: str, relevant: Set[int], policy_id: str) -> None:
    print(f"\n=== Demo: {name} ===")
    try:
        await search_and_rerank(query, relevant_docs_indices=relevant, policy_id=policy_id)
    except Exception as exc:
        print(f"Demo '{name}' failed: {exc}")


async def run_cases(cases: List[Tuple[str, str, Set[int], str]]) -> None:
    for idx, (case_name, query, relevant, policy_id) in enumerate(cases):
        await run_case(case_name, query, relevant, policy_id)
        if idx != len(cases) - 1:
            print("\nWaiting 5s before next demo…")
            await asyncio.sleep(5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rerank demo scenarios against the shared corpus.")
    parser.add_argument("--case", "-c", help="Name of the demo case to execute (default: all cases sequentially)")
    args = parser.parse_args()
    if args.case:
        cases = pick_cases(args.case)
    else:
        cases = DEMO_CASES
    asyncio.run(run_cases(cases))


if __name__ == "__main__":
    main()
