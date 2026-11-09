"""Paper figures: pipeline and architecture diagrams, plus charts drawn from the JSON reports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lstm_autoencoder.config import DatasetConfig

BUCKETS: tuple[str, ...] = ("1", "2-4", "5-9", ">=10")
PALETTE = {
    "data": "#dbe9f6",
    "model": "#fde8c8",
    "eval": "#dcefd9",
    "accent": "#f6d5d5",
    "edge": "#44546a",
    "text": "#1f2937",
}
SERIES_COLORS = ["#4c78a8", "#9ecae9", "#f58518", "#54a24b", "#e45756", "#72b7b2"]


def _pyplot():
    """Import matplotlib lazily with the Agg backend, so the rest of the package never needs it."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError("make-figures needs matplotlib: run `uv add matplotlib` (or `uv sync`).") from error
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    return plt


class FigureBuilder:
    """Draw the figures of the manuscript into one folder (PNG for the draft, SVG for submission).

    Diagrams are drawn from scratch; charts are read from the JSON reports under
    ``artifacts/report`` (nothing is trained or recomputed), so they always match the
    tables produced by `summarize-results`. A chart whose source report does not exist
    yet is skipped with a message instead of failing the whole run.
    """

    def __init__(self, dataset_config: DatasetConfig, output_dir: Path | None = None) -> None:
        """Bind the builder to a dataset's report folder and an output folder."""
        self._root = dataset_config.artifacts_dir / "report" / dataset_config.dataset_name
        self._output_dir = output_dir or dataset_config.project_root / "paper" / "figures"

    def build_all(self, window: int = 1, architecture: str = "attn") -> list[Path]:
        """Build every figure that can be built; return the written PNG paths."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        builders = [
            self.pipeline_diagram,
            self.architecture_diagram,
            lambda: self.bucket_performance(window, architecture),
            self.confound_decomposition,
            self.training_dynamics,
        ]
        written = []
        for build in builders:
            path = build()
            if path is not None:
                written.append(path)
        return written

    # ------------------------------------------------------------------ diagrams

    def pipeline_diagram(self) -> Path:
        """Figure 1: data -> features -> split -> windows -> LSTM-AE -> error -> standalone / hybrid evaluation."""
        plt = _pyplot()
        fig, ax = plt.subplots(figsize=(11, 5.6))
        ax.set_xlim(0, 11)
        ax.set_ylim(0, 5.6)
        ax.axis("off")
        width, height = 2.35, 1.15
        xs = [0.15, 2.95, 5.75, 8.55]
        rows = [4.2, 2.35, 0.5]
        lanes = [
            (
                "Data",
                "data",
                [
                    "PaySim\n6.36M transactions",
                    "Keep CASH_OUT + TRANSFER\n2.77M rows, fraud 0.30%",
                    "Causal features per receiver\nhistory_length, step_diff,\nlog_amount_ratio, is_new_receiver",
                    "Receiver-level split\ntrain / val / test\n(no receiver in two splits)",
                ],
            ),
            (
                "Model",
                "model",
                [
                    "Scaler fit on\ntrain-normal rows only",
                    "Windows of the last W\ntransactions of the receiver\n(left padding + mask)",
                    "LSTM-Autoencoder\ntrained on normal\nwindows only",
                    "Reconstruction error e\nmean squared error over\nreal steps",
                ],
            ),
            (
                "Evaluation",
                "eval",
                [
                    "Standalone: e as score\nAUC, AP, Recall@FPR",
                    "Hybrid: LR + log e\n(second stage fit on val)\n+ null control (shuffled e)",
                    "Receiver-cluster bootstrap\npaired deltas, 95% CI",
                    "Per-history-bucket analysis\nAP, fraud rate, 95% CI",
                ],
            ),
        ]
        for (label, color, texts), y in zip(lanes, rows):
            ax.text(0.0, y + height + 0.12, label, fontsize=10, fontweight="bold", color=PALETTE["edge"])
            for i, text in enumerate(texts):
                self._box(ax, xs[i], y, width, height, text, PALETTE[color])
                if i < len(texts) - 1:
                    self._arrow(ax, (xs[i] + width, y + height / 2), (xs[i + 1], y + height / 2))
        # lane hand-overs: the split feeds the scaler; the error feeds both evaluations
        for upper, lower in ((rows[0], rows[1]), (rows[1], rows[2])):
            self._elbow(ax, xs[3] + width / 2, upper, xs[0] + width / 2, lower + height, mid_y=lower + height + 0.3)
        ax.set_title(
            "Figure 1. Evaluation pipeline (checkpoint selection and all fits use training or validation data only)",
            fontsize=10,
            loc="left",
            pad=14,
        )
        return self._save(fig, "fig1_pipeline")

    def architecture_diagram(self) -> Path:
        """Figure 2: the four decoder-context variants of the LSTM-AE, with the optional latent code."""
        plt = _pyplot()
        fig, axes = plt.subplots(2, 2, figsize=(11, 5.4))
        variants = [
            ("plain", "Per-step encoder outputs\n(no summary vector)", "context_t = h_t", False),
            ("bottleneck", "Final hidden state,\nrepeated W times", "context = h_W", True),
            (
                "attn (masked self-attention)",
                "Masked self-attention: step t cannot\nattend to itself (unless the window\nhas a single real step)",
                "context_t = sum_s a_ts V_s  (s != t), V = value projection of h",
                False,
            ),
            (
                "attn_pool (attention pooling)",
                "One attention-pooled vector,\nrepeated W times",
                "context = sum_t a_t V_t, query = learned vector",
                True,
            ),
        ]
        for ax, (title, middle, formula, latent) in zip(axes.ravel(), variants):
            ax.set_xlim(0, 9.6)
            ax.set_ylim(0.4, 4.0)
            ax.axis("off")
            ax.set_title(title, fontsize=10, loc="left", fontweight="bold")
            y, h = 1.3, 1.1
            mid = y + h / 2
            self._box(ax, 0.05, y, 1.2, h, "Window\nx_1..x_W\n(W x 5)", PALETTE["data"])
            self._box(ax, 1.65, y, 1.4, h, "Encoder\nLSTM x2", PALETTE["model"])
            self._box(ax, 3.45, y, 2.8, h, middle, PALETTE["accent"], fontsize=7.5)
            self._box(ax, 6.65, y, 1.4, h, "Decoder\nLSTM + Linear", PALETTE["model"])
            self._box(ax, 8.45, y, 1.0, h, "x_hat", PALETTE["eval"])
            for x1, x2 in ((1.25, 1.65), (3.05, 3.45), (6.25, 6.65), (8.05, 8.45)):
                self._arrow(ax, (x1, mid), (x2, mid))
            ax.text(4.85, y - 0.3, formula, ha="center", fontsize=7.5, color=PALETTE["edge"])
            if latent:
                self._box(
                    ax,
                    3.45,
                    2.85,
                    2.8,
                    0.75,
                    "optional latent code z in R^k\nz = tanh(W c + b), then back up",
                    "white",
                    fontsize=7,
                )
                ax.annotate(
                    "",
                    xy=(4.85, 2.85),
                    xytext=(4.85, y + h),
                    arrowprops={"arrowstyle": "<->", "color": PALETTE["edge"]},
                )
        fig.suptitle(
            "Figure 2. LSTM-AE variants. Anomaly score = mean squared error between x and x_hat over real (unpadded) steps",
            fontsize=10,
            x=0.01,
            ha="left",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        return self._save(fig, "fig2_architectures")

    # ------------------------------------------------------------------ charts from reports

    def bucket_performance(self, window: int = 1, architecture: str = "attn") -> Path | None:
        """Figure 3: AUC, AP and fraud rate per history bucket for the baselines, the hybrid and the AE alone."""
        baseline = self._latest("baseline")
        hybrid = self._latest_hybrid(window, architecture)
        if baseline is None:
            print("[figures] no baseline report; skipping the per-bucket chart")
            return None
        ablation = baseline["ablation"]
        series: dict[str, dict[str, dict]] = {}
        for name, label in (
            ("full", "LR full"),
            ("no_history", "LR no history"),
            ("full_gbm", "GBM full"),
            ("full_gbm_no_step", "GBM no step"),
            ("ae_features", "LR (AE's 5 features)"),
            ("ae_features_gbm", "GBM (AE's 5 features)"),
        ):
            if name in ablation and ablation[name].get("by_bucket"):
                series[label] = ablation[name]["by_bucket"]
        if hybrid is not None and hybrid.get("per_seed"):
            for variant, label in (
                ("hybrid_val", f"Hybrid (W={window}, {architecture})"),
                ("ae_error_only", f"AE error only (W={window}, {architecture})"),
            ):
                merged = self._mean_over_seeds(hybrid["per_seed"], variant)
                if merged:
                    series[label] = merged

        plt = _pyplot()
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
        x = np.arange(len(BUCKETS))
        bar = 0.8 / max(len(series), 1)
        for k, (label, by_bucket) in enumerate(series.items()):
            for ax, key in ((axes[0], "auc"), (axes[1], "ap")):
                values = [by_bucket.get(bucket, {}).get(key, np.nan) for bucket in BUCKETS]
                ax.bar(
                    x + k * bar - 0.4 + bar / 2, values, bar, label=label, color=SERIES_COLORS[k % len(SERIES_COLORS)]
                )
        axes[0].set_ylabel("AUC (test)")
        axes[0].set_ylim(0.4, 1.0)
        axes[0].axhline(0.5, color="grey", lw=0.8, ls="--")
        axes[1].set_ylabel("Average precision (test, log scale)")
        axes[1].set_yscale("log")
        # Older baseline reports carry no `n_fraud`; take the counts from the first series that has them.
        counted = next((s for s in series.values() if all("n_fraud" in s.get(bucket, {}) for bucket in BUCKETS)), {})
        rates = [self._fraud_rate(counted.get(bucket, {})) for bucket in BUCKETS]
        counts = [counted.get(bucket, {}).get("n_fraud", 0) for bucket in BUCKETS]
        axes[2].bar(x, np.array(rates) * 100, color=SERIES_COLORS[4])
        for xi, (rate, count) in enumerate(zip(rates, counts)):
            axes[2].text(xi, rate * 100, f"{count} fraud", ha="center", va="bottom", fontsize=7)
        axes[2].set_ylabel("Fraud rate in bucket (%)")
        for ax in axes:
            ax.set_xticks(x, [f"bucket {b}" for b in BUCKETS])
            ax.set_xlabel("History bucket (transactions of the receiver so far)")
        axes[0].legend(fontsize=6.5, loc="upper right", frameon=False)
        fig.suptitle(
            "Figure 3. Performance by history bucket (AUC falls with history length; fraud concentrates in bucket 1)",
            fontsize=10,
            x=0.01,
            ha="left",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        return self._save(fig, "fig3_bucket_performance")

    def confound_decomposition(self) -> Path | None:
        """Figure 4: pooled AUC vs the AUC inside buckets and across buckets, for every diagnosed configuration."""
        latest: dict[tuple[int, str], dict] = {}
        for _, report in self._reports("score_diagnostics"):
            summary = report.get("score_diagnostics")
            if summary and summary.get("per_seed"):
                latest[(summary["window_length"], summary["architecture"])] = summary
        if not latest:
            print("[figures] no score_diagnostics reports; skipping the confound chart")
            return None
        labels, pooled, within, cross = [], [], [], []
        for (window, architecture), summary in sorted(latest.items()):
            parts = self.decompose_pooled_auc(summary["per_seed"][0]["per_bucket"])
            labels.append(f"{architecture}\nW={window}")
            pooled.append(parts["pooled"])
            within.append(parts["within"])
            cross.append(parts["cross"])

        plt = _pyplot()
        fig, ax = plt.subplots(figsize=(max(6.5, 1.3 * len(labels) + 2), 3.9))
        x = np.arange(len(labels))
        for k, (name, values) in enumerate(
            (("pooled (reported)", pooled), ("within buckets", within), ("across buckets", cross))
        ):
            ax.bar(x + (k - 1) * 0.26, values, 0.26, label=name, color=SERIES_COLORS[[4, 0, 2][k]])
        ax.axhline(0.5, color="grey", lw=0.8, ls="--")
        ax.set_xticks(x, labels, fontsize=7)
        ax.set_ylabel("AUC of the reconstruction error (validation)")
        ax.set_ylim(0.2, 0.85)
        ax.legend(frameon=False, fontsize=8)
        ax.set_title(
            "Figure 4. The pooled AUC mixes within-bucket signal (about 0.70) with across-bucket pairs (below 0.5)",
            fontsize=10,
            loc="left",
        )
        fig.tight_layout()
        return self._save(fig, "fig4_confound")

    def training_dynamics(self) -> Path | None:
        """Figure 5: validation AUC and validation loss per epoch for the most recent run of each capacity setting."""
        runs: dict[str, dict] = {}
        for _, report in self._reports("model_training"):
            run_config, multi = report.get("run_config"), report.get("multi_seed_training")
            if not run_config or not multi:
                continue
            first = multi["runs"][0]
            if first.get("history"):
                runs[f"{run_config['architecture']} W={run_config['window_length']}"] = first
        if not runs:
            print("[figures] no training reports; skipping the dynamics chart")
            return None
        chosen = self._pick_dynamics_runs(runs)
        plt = _pyplot()
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
        for k, label in enumerate(chosen):
            history = runs[label]["history"]
            epochs = [h["epoch"] for h in history]
            axes[0].plot(
                epochs, [h["val_auc"] for h in history], marker="o", ms=2.5, label=label, color=SERIES_COLORS[k % 6]
            )
            axes[1].plot(
                epochs,
                [float(h["val_loss"]) for h in history],
                marker="o",
                ms=2.5,
                label=label,
                color=SERIES_COLORS[k % 6],
            )
        axes[0].axhline(0.5, color="grey", lw=0.8, ls="--")
        axes[0].set_ylabel("Validation AUC of the reconstruction error")
        axes[1].set_ylabel("Validation loss (normal rows, log scale)")
        axes[1].set_yscale("log")
        for ax in axes:
            ax.set_xlabel("Epoch")
        axes[0].legend(fontsize=7, frameon=False)
        fig.suptitle(
            "Figure 5. As validation loss falls, AUC falls (capacity-limited runs) or oscillates (near-zero loss)",
            fontsize=10,
            x=0.01,
            ha="left",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        return self._save(fig, "fig5_training_dynamics")

    # ------------------------------------------------------------------ pure helpers (unit-tested)

    @staticmethod
    def decompose_pooled_auc(per_bucket: dict[str, dict[str, float]]) -> dict[str, float]:
        """Split a pooled AUC into its within-bucket and across-bucket pair components.

        AUC_pooled = (sum_b n+_b n-_b AUC_b + sum_{b != b'} n+_b n-_b' AUC_bb') / (N+ N-), so the AUC
        over all across-bucket pairs follows from the pooled AUC and the per-bucket AUCs and counts.
        """
        positives = np.array([per_bucket[b]["n_fraud"] for b in BUCKETS], dtype="float64")
        negatives = np.array([per_bucket[b]["n"] - per_bucket[b]["n_fraud"] for b in BUCKETS], dtype="float64")
        auc = np.array([per_bucket[b]["auc_mean"] for b in BUCKETS], dtype="float64")
        pooled = float(per_bucket["pooled"]["auc_mean"])
        all_pairs = positives.sum() * negatives.sum()
        within_pairs = float((positives * negatives).sum())
        within = float((positives * negatives * auc).sum() / within_pairs)
        cross = (pooled * all_pairs - float((positives * negatives * auc).sum())) / (all_pairs - within_pairs)
        return {
            "pooled": pooled,
            "within": within,
            "cross": float(cross),
            "within_pair_share": within_pairs / all_pairs,
        }

    @staticmethod
    def _pick_dynamics_runs(runs: dict[str, dict]) -> list[str]:
        """Prefer capacity-limited runs (names with `_z`), plus one unconstrained example for contrast."""
        limited = [label for label in runs if "_z" in label]
        plain = [label for label in runs if "_z" not in label and "W=1" in label]
        return (plain[:1] + limited)[:5] or list(runs)[:5]

    @staticmethod
    def _fraud_rate(metrics: dict) -> float:
        """Fraud rate of a bucket entry, from `fraud_rate` or from the counts."""
        if "fraud_rate" in metrics:
            return float(metrics["fraud_rate"])
        n = metrics.get("n", 0)
        return float(metrics.get("n_fraud", 0) / n) if n else 0.0

    @staticmethod
    def _mean_over_seeds(per_seed: list[dict], variant: str) -> dict[str, dict[str, float]]:
        """Average the per-bucket metrics of one hybrid variant over seeds (counts are identical across seeds)."""
        merged: dict[str, dict[str, float]] = {}
        for bucket in BUCKETS:
            entries = [
                seed["by_bucket"][variant][bucket]
                for seed in per_seed
                if bucket in seed.get("by_bucket", {}).get(variant, {})
            ]
            if entries:
                merged[bucket] = {
                    "n": entries[0].get("n", 0),
                    "n_fraud": entries[0].get("n_fraud", 0),
                    "auc": float(np.mean([e["auc"] for e in entries])),
                    "ap": float(np.mean([e["ap"] for e in entries])),
                }
        return merged

    # ------------------------------------------------------------------ drawing and I/O helpers

    @staticmethod
    def _box(ax, x: float, y: float, w: float, h: float, text: str, color: str, fontsize: float = 8) -> None:
        """Rounded labelled rectangle."""
        from matplotlib.patches import FancyBboxPatch

        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec=PALETTE["edge"], lw=1.0
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=PALETTE["text"])

    @staticmethod
    def _arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
        """Straight arrow between two points."""
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "-|>", "color": PALETTE["edge"], "lw": 1.1})

    @staticmethod
    def _elbow(ax, x1: float, y1: float, x2: float, y2: float, mid_y: float) -> None:
        """Down - across - down connector that runs through the gap between two lanes."""
        ax.plot([x1, x1, x2], [y1, mid_y, mid_y], color=PALETTE["edge"], lw=1.1, solid_capstyle="round")
        ax.annotate(
            "", xy=(x2, y2), xytext=(x2, mid_y), arrowprops={"arrowstyle": "-|>", "color": PALETTE["edge"], "lw": 1.1}
        )

    def _save(self, fig, name: str) -> Path:
        """Write PNG (for the draft) and SVG (for submission) and return the PNG path."""
        plt = _pyplot()
        png = self._output_dir / f"{name}.png"
        fig.savefig(png, dpi=200, bbox_inches="tight")
        fig.savefig(self._output_dir / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)
        print(f"[figures] wrote {png}")
        return png

    def _reports(self, pipeline: str) -> list[tuple[str, dict]]:
        """(timestamp, report) of every JSON report of `pipeline`, oldest first."""
        found = [
            (path.parent.name, json.loads(path.read_text(encoding="utf-8")))
            for path in (self._root / pipeline).glob("*/*/*.json")
        ]
        return sorted(found, key=lambda item: item[0])

    def _latest(self, pipeline: str) -> dict | None:
        """The newest report of `pipeline`, or None."""
        reports = self._reports(pipeline)
        return reports[-1][1] if reports else None

    def _latest_hybrid(self, window: int, architecture: str) -> dict | None:
        """The newest hybrid report for one (window, architecture), or None."""
        match = None
        for _, report in self._reports("hybrid_classifier"):
            hybrid = report.get("hybrid_classifier")
            if hybrid and hybrid.get("window_length") == window and hybrid.get("architecture") == architecture:
                match = hybrid
        return match
