"""
analysis.py — Post-training analysis for SNN run records.

Usage:
    python analysis.py
    python analysis.py --record run_record.npz
    python analysis.py --record run_record.npz --snapshots run_record_snapshots.npz
    python analysis.py --record run_record.npz --no-snapshots
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_record(record_path: str) -> dict:
    path = Path(record_path)
    if not path.exists():
        print(f"[error] record file not found: {path}")
        sys.exit(1)
    data = np.load(path, allow_pickle=False)
    print(f"[info] loaded record: {path}")
    print(f"       keys      : {list(data.files)}")
    print(f"       samples   : {data['firing_rates'].shape[0]}")
    print(f"       layers    : {data['firing_rates'].shape[1]}")
    print(f"       elapsed   : {float(data['elapsed_seconds']):.1f}s")
    return data


def load_snapshots(snapshots_path: str) -> dict | None:
    path = Path(snapshots_path)
    if not path.exists():
        print(
            f"[warn] snapshots file not found: {path} — skipping receptive field plots"
        )
        return None
    data = np.load(path, allow_pickle=False)
    print(f"[info] loaded snapshots: {path}")
    print(f"       checkpoints: {len(data.files)}")
    return data


def smooth(x: np.ndarray, window: int = 200) -> np.ndarray:
    """Simple moving average for readability on long runs."""
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def save(fig: plt.Figure, name: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[saved] {path}")


def plot_learning_dynamics(data: dict, output_dir: Path) -> None:
    weight_stats = data["weight_stats"]
    firing_rates = data["firing_rates"]
    n_synapses = weight_stats.shape[1]
    samples = np.arange(len(firing_rates))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Learning Dynamics", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    for s in range(n_synapses):
        y = weight_stats[:, s, 0]
        ax.plot(smooth(y), label=f"Synapse {s}", alpha=0.9)
    ax.set_title("Weight mean over training")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Mean weight")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for s in range(n_synapses):
        y = weight_stats[:, s, 1]
        ax.plot(smooth(y), label=f"Synapse {s}", alpha=0.9)
    ax.set_title("Weight std over training  (↑ = specialisation)")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Std weight")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for s in range(n_synapses):
        y = weight_stats[:, s, 3]
        ax.plot(smooth(y), label=f"Synapse {s}", alpha=0.9)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8, label="50% warning")
    ax.set_title("Fraction of weights at w_max  (↑ = STDP stalling)")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Fraction clipped high")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    n_layers = firing_rates.shape[1]
    for layer_idx in range(n_layers):
        y = firing_rates[:, layer_idx]
        ax.plot(smooth(y), label=f"Layer {layer_idx}", alpha=0.8)
    ax.set_title("Mean firing rate per layer")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Mean firing rate")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save(fig, "learning_dynamics.png", output_dir)
    plt.close(fig)


def plot_neuron_health(data: dict, output_dir: Path) -> None:
    silent = data["silent_fraction"]
    saturated = data["saturated_fraction"]
    n_layers = silent.shape[1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Neuron Health", fontsize=14, fontweight="bold")

    ax = axes[0]
    for layer_idx in range(n_layers):
        y = silent[:, layer_idx]
        ax.plot(smooth(y), label=f"Layer {layer_idx}", alpha=0.8)
    ax.axhline(
        0.1, color="orange", linestyle="--", linewidth=0.8, label="10% threshold"
    )
    ax.set_title("Silent neuron fraction  (want < 10%)")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    for layer_idx in range(n_layers):
        y = saturated[:, layer_idx]
        ax.plot(smooth(y), label=f"Layer {layer_idx}", alpha=0.8)
    ax.axhline(
        0.1, color="orange", linestyle="--", linewidth=0.8, label="10% threshold"
    )
    ax.set_title("Saturated neuron fraction  (want < 10%)")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save(fig, "neuron_health.png", output_dir)
    plt.close(fig)


def plot_weight_distributions(snapshots: dict, output_dir: Path) -> None:
    keys = sorted(snapshots.files, key=lambda k: int(k.split("_")[1]))
    if len(keys) < 2:
        print("[warn] fewer than 2 snapshots — skipping weight distribution plot")
        return

    indices = [0, len(keys) // 2, len(keys) - 1]
    selected_keys = [keys[i] for i in indices]
    labels = ["Start", "Middle", "End"]

    n_synapses = len(snapshots[selected_keys[0]])
    fig, axes = plt.subplots(n_synapses, 3, figsize=(13, 4 * n_synapses))
    fig.suptitle(
        "Weight Distributions: Start / Middle / End", fontsize=14, fontweight="bold"
    )

    if n_synapses == 1:
        axes = axes[np.newaxis, :]

    for syn_idx in range(n_synapses):
        for col, (key, label) in enumerate(zip(selected_keys, labels)):
            weights = snapshots[key][syn_idx].flatten()
            ax = axes[syn_idx, col]
            ax.hist(
                weights, bins=40, range=(0, 1), color="steelblue", edgecolor="white"
            )
            ax.set_title(f"Synapse {syn_idx} — {label}  (sample {key.split('_')[1]})")
            ax.set_xlabel("Weight value")
            ax.set_ylabel("Count")
            ax.set_xlim(0, 1)
            ax.grid(alpha=0.3)

    plt.tight_layout()
    save(fig, "weight_distributions.png", output_dir)
    plt.close(fig)


def plot_receptive_fields(
    snapshots: dict,
    output_dir: Path,
    synapse_idx: int = 0,
    n_neurons: int = 8,
    input_shape: tuple = (28, 28),
) -> None:
    keys = sorted(snapshots.files, key=lambda k: int(k.split("_")[1]))
    if not keys:
        print("[warn] no snapshots available — skipping receptive field plot")
        return

    n_snapshots = len(keys)
    fig, axes = plt.subplots(
        n_snapshots, n_neurons, figsize=(n_neurons * 1.8, n_snapshots * 1.8)
    )
    fig.suptitle(
        f"Receptive Fields — Synapse {synapse_idx}  ({n_neurons} hidden neurons over training)",
        fontsize=13,
        fontweight="bold",
    )

    if n_snapshots == 1:
        axes = axes[np.newaxis, :]

    for row, key in enumerate(keys):
        weights = snapshots[key][synapse_idx]
        sample_num = key.split("_")[1]

        n_available = weights.shape[0]
        n_to_show = min(n_neurons, n_available)

        for col in range(n_neurons):
            ax = axes[row, col]
            if col < n_to_show:
                rf = weights[col].reshape(input_shape)
                ax.imshow(rf, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
            ax.axis("off")

        axes[row, 0].set_ylabel(f"s={sample_num}", fontsize=7)
        axes[row, 0].axis("on")
        axes[row, 0].tick_params(
            left=False, bottom=False, labelleft=True, labelbottom=False
        )
        for spine in axes[row, 0].spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    save(fig, "receptive_fields.png", output_dir)
    plt.close(fig)


def print_summary(data: dict) -> None:
    firing_rates = data["firing_rates"]
    weight_stats = data["weight_stats"]
    silent = data["silent_fraction"]
    saturated = data["saturated_fraction"]

    n_samples = firing_rates.shape[0]
    n_layers = firing_rates.shape[1]
    n_synapses = weight_stats.shape[1]
    elapsed = float(data["elapsed_seconds"])

    print("\n" + "=" * 55)
    print("  RUN SUMMARY")
    print("=" * 55)
    print(f"  Samples trained  : {n_samples}")
    print(
        f"  Elapsed          : {elapsed:.1f}s  ({elapsed/n_samples*1000:.1f} ms/sample)"
    )
    print(f"  Layers           : {n_layers}")
    print(f"  Synapses         : {n_synapses}")

    print("\n  FINAL FIRING RATES (last 100 samples)")
    for layer_idx in range(n_layers):
        mean = firing_rates[-100:, layer_idx].mean()
        std = firing_rates[-100:, layer_idx].std()
        print(f"    Layer {layer_idx}: {mean:.4f} ± {std:.4f}")

    print("\n  FINAL WEIGHT STATS (last 100 samples)")
    for s in range(n_synapses):
        mean = weight_stats[-100:, s, 0].mean()
        std = weight_stats[-100:, s, 1].mean()
        clip_hi = weight_stats[-100:, s, 3].mean()
        clip_lo = weight_stats[-100:, s, 2].mean()
        print(
            f"    Synapse {s}: mean={mean:.3f}  std={std:.3f}  "
            f"clipped_hi={clip_hi:.1%}  clipped_lo={clip_lo:.1%}"
        )

    print("\n  NEURON HEALTH (last 100 samples)")
    for layer_idx in range(n_layers):
        sil = silent[-100:, layer_idx].mean()
        sat = saturated[-100:, layer_idx].mean()
        status = "✓" if sil < 0.1 and sat < 0.1 else "⚠"
        print(f"    Layer {layer_idx}: silent={sil:.1%}  saturated={sat:.1%}  {status}")

    print("=" * 55 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SNN training record analysis")
    parser.add_argument(
        "--record",
        type=str,
        default="run_record.npz",
        help="Path to run_record.npz (default: run_record.npz)",
    )
    parser.add_argument(
        "--snapshots",
        type=str,
        default="run_record_snapshots.npz",
        help="Path to snapshots .npz (default: run_record_snapshots.npz)",
    )
    parser.add_argument(
        "--no-snapshots", action="store_true", help="Skip all snapshot-based plots"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_output",
        help="Directory to save plots (default: analysis_output/)",
    )
    parser.add_argument(
        "--n-neurons",
        type=int,
        default=8,
        help="Number of hidden neurons to show in receptive field plot (default: 8)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    data = load_record(args.record)
    snapshots = None if args.no_snapshots else load_snapshots(args.snapshots)

    print_summary(data)

    print("[info] generating plots...")
    plot_learning_dynamics(data, output_dir)
    plot_neuron_health(data, output_dir)

    if snapshots is not None:
        plot_weight_distributions(snapshots, output_dir)
        plot_receptive_fields(snapshots, output_dir, n_neurons=args.n_neurons)

    print(f"\n[done] all plots saved to ./{output_dir}/")


if __name__ == "__main__":
    main()
