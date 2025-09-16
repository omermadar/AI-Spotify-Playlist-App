#!/usr/bin/env python3
"""
Plot language distribution from a CSV containing a 'language' column.

Default input file: tracks_added_languages_(Old-Spanish-Problem).csv

Usage examples:
  - Show distribution (counts) for all languages:
      python plot_language_distribution.py
  - Show top 30 languages as percentages:
      python plot_language_distribution.py --top 30 --normalize
  - Exclude 'Unknown' and require at least 20 occurrences:
      python plot_language_distribution.py --no-unknown --min-count 20
  - Save the chart instead of (or in addition to) showing it:
      python plot_language_distribution.py --save Language-Distribution.png

This script does not modify any existing project files.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


DEFAULT_CSV = 'tracks_added_languages.csv'
DEFAULT_COLUMN = 'language'


def load_data(csv_path: Path, column: str) -> pd.Series:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in {csv_path}. Available columns: {list(df.columns)}")

    # Normalize/clean language values: fillna and strip whitespace
    lang_series = df[column].fillna('Unknown').astype(str).str.strip()
    # Collapse empty strings to 'Unknown'
    lang_series = lang_series.replace({'': 'Unknown'})
    return lang_series


MESSAGE = (
    "Plot the distribution of languages from a CSV containing a 'language' column."
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=MESSAGE)
    parser.add_argument('--csv', default=DEFAULT_CSV, help=f"Path to CSV file (default: {DEFAULT_CSV})")
    parser.add_argument('--column', default=DEFAULT_COLUMN, help=f"Language column name (default: {DEFAULT_COLUMN})")
    parser.add_argument('--top', type=int, default=None, help='Plot only the top N languages by frequency')
    parser.add_argument('--min-count', type=int, default=None, help='Filter out languages with total count below this threshold')
    parser.add_argument('--normalize', action='store_true', help='Plot percentages instead of raw counts')
    parser.add_argument('--save', default=None, help='Path to save the plot image (e.g., Language-Distribution.png)')
    parser.add_argument('--show', action='store_true', help='Force showing the plot window (default: show if no --save)')
    parser.add_argument('--no-unknown', dest='unknown', action='store_false', help="Exclude 'Unknown' category from the plot")
    parser.add_argument('--show-unknown', dest='unknown', action='store_true', help="Include 'Unknown' category (default)")
    parser.set_defaults(unknown=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    csv_path = Path(args.csv)
    try:
        lang_series = load_data(csv_path, args.column)
    except (FileNotFoundError, KeyError) as e:
        print(f"Error: {e}")
        return 1

    # Compute counts
    counts = lang_series.value_counts(dropna=False)

    if not args.unknown:
        counts = counts[counts.index != 'Unknown']

    if args.min_count is not None:
        counts = counts[counts >= args.min_count]

    # Keep only top N if requested
    if args.top is not None and args.top > 0:
        counts = counts.sort_values(ascending=False).head(args.top)
    else:
        counts = counts.sort_values(ascending=False)

    if counts.empty:
        print("Nothing to plot after filtering. Adjust --top/--min-count/--no-unknown settings.")
        return 0

    # Prepare plotting data
    values = counts
    ylabel = 'Percentage (%)' if args.normalize else 'Count'
    if args.normalize:
        values = (counts / counts.sum()) * 100.0

    # Plot
    plt.figure(figsize=(12, max(6, min(12, 0.35 * len(values) + 3))))
    # Build a DataFrame for seaborn and set hue to avoid deprecation warning
    df_plot = pd.DataFrame({
        'Language': values.index,
        'Value': values.values
    })
    sns.barplot(
        data=df_plot,
        x='Value',
        y='Language',
        hue='Language',
        orient='h',
        dodge=False,
        palette='viridis',
        legend=False
    )
    plt.title('Language Distribution')
    plt.xlabel(ylabel)
    plt.ylabel('Language')

    # Annotate bars with counts or percentages
    for i, (label, val) in enumerate(values.items()):
        if args.normalize:
            text = f"{val:.1f}%"
        else:
            text = f"{int(counts[label])}"
        plt.text(val + (values.max() * 0.01), i, text, va='center')

    plt.tight_layout()

    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        print(f"Saved language distribution plot to: {out_path}")

    # Show the plot by default if not only saving, or if --show is provided
    if args.show or not args.save:
        plt.show()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
