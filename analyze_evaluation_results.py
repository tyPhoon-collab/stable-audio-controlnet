#!/usr/bin/env python3
"""
Analyze evaluation results JSON and display files sorted by accuracy.
"""

import argparse
import json
import sys
from typing import Dict, List, Optional, Tuple


def load_results(json_path: str) -> Dict:
    """Load evaluation results from JSON file."""
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading JSON file: {e}", file=sys.stderr)
        sys.exit(1)


def extract_file_accuracies(results: Dict) -> List[Tuple[str, float, float]]:
    """
    Extract filename, accuracy, and root_accuracy from results.
    Returns list of (filename, accuracy, root_accuracy) sorted by accuracy descending.
    """
    files_data = results.get("chord_metrics", {}).get("files", {})

    file_list = []
    for filename, metrics in files_data.items():
        accuracy = metrics.get("accuracy", 0.0)
        root_accuracy = metrics.get("root_accuracy", 0.0)
        file_list.append((filename, accuracy, root_accuracy))

    # Sort by accuracy descending
    file_list.sort(key=lambda x: x[1], reverse=True)
    return file_list


def print_results(
    file_list: List[Tuple[str, float, float]],
    show_root: bool = False,
    threshold: Optional[float] = None,
    limit: Optional[int] = None,
) -> None:
    """
    Print results in formatted table.

    Args:
        file_list: List of (filename, accuracy, root_accuracy) tuples
        show_root: Whether to show root_accuracy column
        threshold: Only show files with accuracy >= threshold
        limit: Maximum number of files to show
    """
    # Filter by threshold
    if threshold is not None:
        file_list = [(f, a, r) for f, a, r in file_list if a >= threshold]

    # Apply limit
    if limit is not None:
        file_list = file_list[:limit]

    # Print header
    if show_root:
        header = f"{'Rank':<4} {'Accuracy':<10} {'Root Acc':<10} Filename"
        print(header)
        print("-" * 100)
    else:
        header = f"{'Rank':<4} {'Accuracy':<10} Filename"
        print(header)
        print("-" * 100)

    # Print results
    for rank, (filename, accuracy, root_accuracy) in enumerate(file_list, 1):
        if show_root:
            print(f"{rank:<4} {accuracy:<10.4f} {root_accuracy:<10.4f} {filename}")
        else:
            print(f"{rank:<4} {accuracy:<10.4f} {filename}")

    # Print summary
    print("-" * (100 if show_root else 80))
    print(f"Total: {len(file_list)} files")
    if file_list:
        accuracies = [a for _, a, _ in file_list]
        print(f"Average accuracy: {sum(accuracies) / len(accuracies):.4f}")
        print(f"Max accuracy: {max(accuracies):.4f}")
        print(f"Min accuracy: {min(accuracies):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze evaluation results and display files sorted by accuracy"
    )
    parser.add_argument("json_file", help="Path to evaluation results JSON file")
    parser.add_argument(
        "-r", "--show-root", action="store_true", help="Show root accuracy column"
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=None,
        help="Only show files with accuracy >= threshold",
    )
    parser.add_argument(
        "-l", "--limit", type=int, default=None, help="Maximum number of files to show"
    )
    parser.add_argument(
        "--top", type=int, default=None, help="Show top N files (shorthand for --limit)"
    )
    parser.add_argument("--csv", action="store_true", help="Output in CSV format")

    args = parser.parse_args()

    # Load results
    results = load_results(args.json_file)
    file_list = extract_file_accuracies(results)

    # Apply limit or top
    limit = args.top if args.top is not None else args.limit

    if args.csv:
        # CSV output
        if args.show_root:
            print("rank,accuracy,root_accuracy,filename")
            for rank, (filename, accuracy, root_accuracy) in enumerate(
                file_list[:limit] if limit else file_list, 1
            ):
                if args.threshold is None or accuracy >= args.threshold:
                    print(f"{rank},{accuracy},{root_accuracy},{filename}")
        else:
            print("rank,accuracy,filename")
            for rank, (filename, accuracy, _) in enumerate(
                file_list[:limit] if limit else file_list, 1
            ):
                if args.threshold is None or accuracy >= args.threshold:
                    print(f"{rank},{accuracy},{filename}")
    else:
        # Table output
        print_results(file_list, args.show_root, args.threshold, limit)


if __name__ == "__main__":
    main()
