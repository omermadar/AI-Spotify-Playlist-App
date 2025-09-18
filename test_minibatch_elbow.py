import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple

from DataCleaning import DataCleaner
from config import FEATURES
from sklearn.metrics import silhouette_score
import numpy as np


def get_algorithm(algo: str, **kwargs):
    """
    Factory to return a clustering estimator for the given algorithm name.
    Currently supports:
      - 'minibatchkmeans' (default)
    Designed to be extended later (e.g., 'kmeans', 'birch', etc.).
    """
    algo = (algo or 'minibatchkmeans').strip().lower()
    if algo in ('minibatchkmeans', 'mini-batch-kmeans', 'mini_batch_kmeans'):
        from sklearn.cluster import MiniBatchKMeans
        n_clusters = kwargs.pop('n_clusters')
        batch_size = kwargs.pop('batch_size', 4096)
        random_state = kwargs.pop('random_state', 42)
        return MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=random_state, **kwargs)
    # Placeholder for future extension
    raise ValueError(f"Unsupported algorithm '{algo}'. Supported: minibatchkmeans")


def compute_inertias(X, algo: str, k_range: range, batch_size: int, random_state: int) -> List[Tuple[int, float]]:
    """Compute inertia for each k using the specified clustering algorithm."""
    results = []
    for k in k_range:
        est = get_algorithm(algo, n_clusters=k, batch_size=batch_size, random_state=random_state)
        est.fit(X)
        inertia = float(getattr(est, 'inertia_', float('nan')))
        results.append((k, inertia))
        print(f"k={k:2d}  inertia={inertia:.4f}")
    return results


def plot_elbow(results: List[Tuple[int, float]], title: str, save_path: str | None):
    ks = [k for k, _ in results]
    inertias = [i for _, i in results]
    plt.figure(figsize=(8, 5))
    plt.plot(ks, inertias, marker='o')
    plt.xticks(ks)
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia (within-cluster sum of squares)')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved elbow plot to: {save_path}")
    plt.show()


def load_and_scale(data_path: str):
    df = pd.read_csv(data_path)
    cleaner = DataCleaner()
    X, _ = cleaner.clean_data(df)
    X_scaled = cleaner.scale_features(X)
    # Ensure only configured features are used
    cols = [c for c in FEATURES if c in X_scaled.columns]
    return X_scaled[cols].values


def compute_elbow_and_silhouette(X_fit, X_sil, algo: str, k_range: range, batch_size: int, random_state: int, do_silhouette: bool):
    """Compute inertia and silhouette (on X_sil) for each k using specified algorithm.
    Returns a list of tuples: (k, inertia, silhouette_or_nan)
    """
    results = []
    for k in k_range:
        est = get_algorithm(algo, n_clusters=k, batch_size=batch_size, random_state=random_state)
        est.fit(X_fit)
        inertia = float(getattr(est, 'inertia_', float('nan')))
        sil = float('nan')
        if do_silhouette:
            try:
                labels = est.predict(X_sil)
                sil = float(silhouette_score(X_sil, labels, metric='euclidean'))
            except Exception as e:
                print(f"k={k:2d}  silhouette computation failed: {e}")
                sil = float('nan')
        print(f"k={k:3d}  inertia={inertia:.4f}  silhouette={sil if isinstance(sil, float) and not np.isnan(sil) else 'nan'}")
        results.append((k, inertia, sil))
    return results


def plot_elbow_with_silhouette(results: List[Tuple[int, float, float]], title: str, save_path: str | None):
    ks = [k for k, _, _ in results]
    inertias = [i for _, i, _ in results]
    silhouettes = [s for _, _, s in results]

    fig, ax1 = plt.subplots(figsize=(9, 5))

    color1 = 'tab:blue'
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Inertia (WSS)', color=color1)
    ax1.plot(ks, inertias, marker='o', color=color1, label='Inertia')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(ks)
    ax1.grid(True, linestyle='--', alpha=0.4)

    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.set_ylabel('Silhouette score', color=color2)
    ax2.plot(ks, silhouettes, marker='s', linestyle='--', color=color2, label='Silhouette (sample)')
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title(title)
    fig.tight_layout()

    # Compose a joint legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved combined elbow+silhouette plot to: {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Elbow + Silhouette for MiniBatchKMeans (extensible)")
    parser.add_argument('--data', type=str, default='tracks_added_languages.csv', help='Path to CSV data file')
    parser.add_argument('--algo', type=str, default='minibatchkmeans', help="Clustering algorithm (default: minibatchkmeans)")
    parser.add_argument('--min-k', type=int, default=3, dest='min_k', help='Minimum k (inclusive)')
    parser.add_argument('--max-k', type=int, default=200, dest='max_k', help='Maximum k (inclusive)')
    parser.add_argument('--step', type=int, default=5, help='Step between k values')
    parser.add_argument('--batch-size', type=int, default=4096, dest='batch_size', help='MiniBatchKMeans batch size')
    parser.add_argument('--random-state', type=int, default=42, dest='random_state', help='Random seed')
    parser.add_argument('--sil-sample-size', type=int, default=20000, dest='sil_sample_size', help='Rows to randomly sample for silhouette score (without replacement)')
    parser.add_argument('--sample-both', action='store_true', help='Also compute inertia on the sampled subset (faster, but approximate)')
    parser.add_argument('--no-silhouette', action='store_true', help='Disable silhouette computation entirely')
    parser.add_argument('--save', type=str, default='K-Decision-Pictures/Elbow-MiniBatchKMeans.png', help='Path to save the combined plot image')
    args = parser.parse_args()

    if args.min_k < 1 or args.max_k < args.min_k or args.step < 1:
        print("Invalid k parameters. Ensure 1 <= min_k <= max_k and step >= 1.")
        sys.exit(1)

    try:
        X = load_and_scale(args.data)
    except Exception as e:
        print(f"Failed to load and scale data from '{args.data}': {e}")
        sys.exit(1)

    n_total = len(X)
    sil_n = min(args.sil_sample_size, n_total)
    rng = np.random.default_rng(args.random_state)
    if sil_n < n_total:
        sample_idx = rng.choice(n_total, size=sil_n, replace=False)
        X_sil = X[sample_idx]
        print(f"Silhouette will be computed on a random sample of {sil_n:,} rows out of {n_total:,}.")
    else:
        X_sil = X
        print(f"Silhouette will be computed on the full dataset of {n_total:,} rows (requested sample >= data size).")

    X_fit = X_sil if args.sample_both else X

    k_range = range(args.min_k, args.max_k + 1, args.step)
    print(f"Running metrics for {args.algo} with k in [{args.min_k}..{args.max_k}] step={args.step} ...")
    results = compute_elbow_and_silhouette(
        X_fit=X_fit,
        X_sil=X_sil,
        algo=args.algo,
        k_range=k_range,
        batch_size=args.batch_size,
        random_state=args.random_state,
        do_silhouette=(not args.no_silhouette)
    )

    title = f"Elbow + Silhouette — {args.algo} ({'sampled' if args.sample_both else 'full'} fit, silhouette on sample)"
    save_path = args.save
    if args.save == 'K-Decision-Pictures/Elbow-MiniBatchKMeans.png' and args.algo.lower() != 'minibatchkmeans':
        base = args.algo.replace('_', '-').title().replace('-', '')
        save_path = f"K-Decision-Pictures/Elbow-{base}.png"

    plot_elbow_with_silhouette(results, title=title, save_path=save_path)


if __name__ == '__main__':
    main()
