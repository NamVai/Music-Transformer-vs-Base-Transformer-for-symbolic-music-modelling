import json
import os
import csv
import shutil
from typing import Dict, List, Tuple

from utils.seed import set_seed


def _generate_tokens(
    model,
    vocab: Dict[str, int],
    device,
    block_size: int,
    max_time_steps: int = 128,
    temperature: float = 1.0,
) -> List[int]:
    """Autoregressively sample tokens from a model given a BOS token."""
    import torch

    model.eval()
    tokens = [vocab["bos_id"]]
    time_steps = 0
    with torch.no_grad():
        while time_steps < max_time_steps:
            x = torch.tensor(tokens[-block_size:], dtype=torch.long, device=device)
            x = x.unsqueeze(0)
            logits = model(x)[:, -1, :] / max(1e-5, temperature)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            tokens.append(int(next_token))
            if next_token == vocab["time_shift_id"]:
                time_steps += 1
    return tokens


def _reset_outputs(base_dir: str) -> None:
    """Ensure the standard output directory structure exists (no deletion)."""
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, "jsb", "baseline"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "jsb", "relative"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "piano_e", "baseline"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "piano_e", "relative"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "comparison", "metrics"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "comparison", "plots"), exist_ok=True)


def _save_model_outputs(
    model,
    history: Dict[str, List[float]],
    splits,
    vocab: Dict[str, int],
    device,
    config: Dict,
    output_dir: str,
    model_tag: str,
) -> Dict[str, Dict[str, float]]:
    """Save metrics, plots, similarity heatmap, and one sample for a model."""
    # Save loss curves, metrics, and one sample for this model.
    from utils.eval import evaluate_splits
    from utils.plotting import (
        pianoroll_from_tokens,
        plot_loss_curves,
        save_pianoroll_plot,
        save_similarity_heatmap,
    )
    from utils.similarity import token_self_similarity
    from utils.midi import tokens_to_midi

    curves = {model_tag: history}
    plot_loss_curves(curves, os.path.join(output_dir, f"{model_tag}_val_nll.png"))
    with open(os.path.join(output_dir, f"{model_tag}_loss.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    metrics = evaluate_splits(model, splits, vocab, device, config["batch_size"])
    with open(os.path.join(output_dir, f"{model_tag}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    tokens = _generate_tokens(
        model,
        vocab,
        device,
        block_size=config["block_size"],
        max_time_steps=256,
        temperature=1.0,
    )
    roll = pianoroll_from_tokens(
        tokens,
        note_on_base=vocab["note_on_base"],
        time_shift_id=vocab["time_shift_id"],
        min_midi=vocab["min_midi"],
        rest_pitch=vocab["rest_pitch"],
        max_steps=256,
    )
    save_pianoroll_plot(roll, os.path.join(output_dir, f"{model_tag}_sample.png"))

    sim = token_self_similarity(tokens, max_len=min(2048, len(tokens)))
    save_similarity_heatmap(
        sim,
        os.path.join(output_dir, f"{model_tag}_similarity.png"),
        title=f"{model_tag} self-similarity",
    )

    try:
        tokens_to_midi(
            tokens,
            os.path.join(output_dir, f"{model_tag}_sample.mid"),
            note_on_base=vocab["note_on_base"],
            time_shift_id=vocab["time_shift_id"],
            min_midi=vocab["min_midi"],
            rest_pitch=vocab["rest_pitch"],
            ticks_per_step=120,
        )
    except RuntimeError:
        pass

    return metrics


def _save_stability_curve(
    history: Dict[str, List[float]],
    output_path: str,
    title: str,
) -> None:
    """Plot train vs validation NLL over epochs."""
    # Save train vs validation NLL over epochs.
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    plt.figure(figsize=(7, 4))
    if "train" in history:
        plt.plot(history["train"], label="train")
    if "val" in history:
        plt.plot(history["val"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("NLL")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _temperature_sweep(
    model,
    vocab: Dict[str, int],
    device,
    config: Dict,
    output_dir: str,
    model_tag: str,
    temperatures: List[float],
) -> None:
    """Generate temperature-sweep pianorolls and numeric summary metrics (JSB only)."""
    # Generate samples for multiple temperatures (JSB only).
    from utils.plotting import pianoroll_from_tokens, save_pianoroll_plot

    metrics_rows = []
    for temp in temperatures:
        tokens = _generate_tokens(
            model,
            vocab,
            device,
            block_size=config["block_size"],
            max_time_steps=256,
            temperature=temp,
        )
        roll = pianoroll_from_tokens(
            tokens,
            note_on_base=vocab["note_on_base"],
            time_shift_id=vocab["time_shift_id"],
            min_midi=vocab["min_midi"],
            rest_pitch=vocab["rest_pitch"],
            max_steps=256,
        )
        stem = f"{model_tag}_temp_{temp:.1f}"
        save_pianoroll_plot(roll, os.path.join(output_dir, f"{stem}.png"))

        # Simple quantitative metrics for comparison.
        note_density = float(roll.sum()) / max(1, roll.shape[0])
        pitch_range = 0
        if roll.sum() > 0:
            active_pitches = roll.sum(axis=0) > 0
            idx = active_pitches.nonzero()[0]
            if idx.size > 0:
                pitch_range = int(idx.max() - idx.min())
        time_steps = roll.shape[0]
        if time_steps < 2:
            change_rate = 0.0
        else:
            change_rate = float((roll[1:] != roll[:-1]).mean())
        metrics_rows.append(
            {
                "model": model_tag,
                "temperature": temp,
                "note_density": note_density,
                "pitch_range": pitch_range,
                "change_rate": change_rate,
                "time_steps": time_steps,
            }
        )

    # Write per-temperature metrics CSV.
    if metrics_rows:
        csv_path = os.path.join(output_dir, f"{model_tag}_temperature_metrics.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "model",
                    "temperature",
                    "note_density",
                    "pitch_range",
                    "change_rate",
                    "time_steps",
                ],
            )
            writer.writeheader()
            writer.writerows(metrics_rows)


def _run_dataset(
    dataset_name: str,
    loader_fn,
    config: Dict,
    device,
    output_root: str,
) -> Tuple[
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, float]],
    Dict[str, List[float]],
    Dict[str, List[float]],
    object,
    object,
    Dict[str, int],
]:
    """Train baseline and relative models for one dataset and return metrics and artifacts."""
    # Execution order: dataset -> baseline -> relative.
    local_config = dict(config)
    local_config["run_label"] = dataset_name
    splits, vocab = loader_fn("data", block_size=config["block_size"])

    from train.train_baseline import train_baseline
    from train.train_music_transformer import train_music_transformer

    baseline_model, baseline_hist = train_baseline(splits, vocab, device, local_config)
    baseline_metrics = _save_model_outputs(
        baseline_model,
        baseline_hist,
        splits,
        vocab,
        device,
        local_config,
        os.path.join(output_root, dataset_name, "baseline"),
        "baseline",
    )

    relative_model, relative_hist = train_music_transformer(
        splits, vocab, device, local_config
    )
    relative_metrics = _save_model_outputs(
        relative_model,
        relative_hist,
        splits,
        vocab,
        device,
        local_config,
        os.path.join(output_root, dataset_name, "relative"),
        "relative",
    )

    return (
        baseline_metrics,
        relative_metrics,
        baseline_hist,
        relative_hist,
        baseline_model,
        relative_model,
        vocab,
    )


def _write_comparison(
    output_root: str,
    summary: Dict[str, Dict[str, Dict[str, float]]],
) -> None:
    """Write comparison metrics and plots across datasets and models."""
    # Forecasting-style summary across datasets and models.
    metrics_dir = os.path.join(output_root, "comparison", "metrics")
    plots_dir = os.path.join(output_root, "comparison", "plots")

    with open(os.path.join(metrics_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(metrics_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "model", "split", "nll", "ppl"])
        for dataset_name, models in summary.items():
            for model_name, splits in models.items():
                for split_name, vals in splits.items():
                    writer.writerow([dataset_name, model_name, split_name, vals["nll"], vals["ppl"]])

    # Per-split metrics table (Markdown + CSV).
    with open(os.path.join(metrics_dir, "per_split.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "model", "train_nll", "valid_nll", "test_nll", "train_ppl", "valid_ppl", "test_ppl"])
        for dataset_name, models in summary.items():
            for model_name, splits in models.items():
                writer.writerow(
                    [
                        dataset_name,
                        model_name,
                        splits["train"]["nll"],
                        splits["valid"]["nll"],
                        splits["test"]["nll"],
                        splits["train"]["ppl"],
                        splits["valid"]["ppl"],
                        splits["test"]["ppl"],
                    ]
                )

    with open(os.path.join(metrics_dir, "per_split.md"), "w", encoding="utf-8") as f:
        f.write("# Per-split Metrics (NLL / PPL)\n\n")
        f.write("| Dataset | Model | Train NLL | Valid NLL | Test NLL | Train PPL | Valid PPL | Test PPL |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for dataset_name, models in summary.items():
            for model_name, splits in models.items():
                f.write(
                    f"| {dataset_name} | {model_name} | "
                    f"{splits['train']['nll']:.4f} | {splits['valid']['nll']:.4f} | {splits['test']['nll']:.4f} | "
                    f"{splits['train']['ppl']:.2f} | {splits['valid']['ppl']:.2f} | {splits['test']['ppl']:.2f} |\n"
                )

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    # One comparison plot per metric (NLL, PPL) on validation split.
    for metric in ["nll", "ppl"]:
        labels = []
        values = []
        for dataset_name, models in summary.items():
            for model_name, splits in models.items():
                labels.append(f"{dataset_name}-{model_name}")
                values.append(splits["valid"][metric])
        plt.figure(figsize=(6, 3.5))
        plt.bar(labels, values)
        plt.ylabel(metric.upper())
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"comparison_{metric}.png"), dpi=150)
        plt.close()


def main() -> None:
    """Run the full training and evaluation pipeline in a fixed order."""
    # Deterministic behavior.
    set_seed(42)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available. Install a CUDA-enabled PyTorch build.")
    device = torch.device("cuda")

    # Late imports to avoid failing before preflight.
    from data.jsb_chorales import load_jsb_chorales
    from data.maestro import load_maestro_piano

    # Fixed config (no ablations, no nested experiment loops).
    config = {
        "d_model": 128,
        "num_layers": 4,
        "num_heads": 4,
        "d_ff": 256,
        "dropout": 0.1,
        "batch_size": 32,
        "lr": 3e-4,
        "grad_clip": 1.0,
        "max_relative_position": 64,
        "block_size": 512,
        "epochs": 30,
    }

    output_root = "outputs"
    _reset_outputs(output_root)

    # 1) JSB Chorales experiments.
    (
        jsb_baseline,
        jsb_relative,
        jsb_baseline_hist,
        jsb_relative_hist,
        jsb_baseline_model,
        jsb_relative_model,
        jsb_vocab,
    ) = _run_dataset(
        "jsb",
        load_jsb_chorales,
        config,
        device,
        output_root,
    )

    # Training stability curves (train vs val) for JSB.
    _save_stability_curve(
        jsb_baseline_hist,
        os.path.join(output_root, "jsb", "baseline", "baseline_stability.png"),
        "JSB Baseline: Train vs Val NLL",
    )
    _save_stability_curve(
        jsb_relative_hist,
        os.path.join(output_root, "jsb", "relative", "relative_stability.png"),
        "JSB Relative: Train vs Val NLL",
    )

    # Temperature sweep for JSB only (baseline + relative).
    temps = [0.8, 1.0, 1.2]
    _temperature_sweep(
        model=jsb_baseline_model,
        vocab=jsb_vocab,
        device=device,
        config=config,
        output_dir=os.path.join(output_root, "jsb", "baseline"),
        model_tag="baseline",
        temperatures=temps,
    )
    _temperature_sweep(
        model=jsb_relative_model,
        vocab=jsb_vocab,
        device=device,
        config=config,
        output_dir=os.path.join(output_root, "jsb", "relative"),
        model_tag="relative",
        temperatures=temps,
    )

    # 2) Piano-e (MAESTRO subset) experiments.
    piano_config = dict(config)
    piano_config["epochs"] = 10
    (
        piano_baseline,
        piano_relative,
        piano_baseline_hist,
        piano_relative_hist,
        _piano_baseline_model,
        _piano_relative_model,
        _piano_vocab,
    ) = _run_dataset(
        "piano_e",
        load_maestro_piano,
        piano_config,
        device,
        output_root,
    )

    # Training stability curves (train vs val) for Piano-e.
    _save_stability_curve(
        piano_baseline_hist,
        os.path.join(output_root, "piano_e", "baseline", "baseline_stability.png"),
        "Piano-e Baseline: Train vs Val NLL",
    )
    _save_stability_curve(
        piano_relative_hist,
        os.path.join(output_root, "piano_e", "relative", "relative_stability.png"),
        "Piano-e Relative: Train vs Val NLL",
    )

    # 3) Relative positional attention comparison (metrics only).
    comparison = {
        "jsb": {
            "baseline": jsb_baseline,
            "relative": jsb_relative,
        },
        "piano_e": {
            "baseline": piano_baseline,
            "relative": piano_relative,
        },
    }

    # 4) Forecasting-style comparison summary (baseline vs relative).
    _write_comparison(output_root, comparison)


if __name__ == "__main__":
    main()
