"""Evaluate registered backbones on CPU with params/FLOPs/MACs/latency/memory metrics.

Usage examples:
  python -m pruning.eval_backbones_cpu --backbones ncsnpp_v2_mobile --batch_size 1 --warmup_frames 4 --frames 20
  python -m pruning.eval_backbones_cpu --all_backbones --frames 10 --output_json pruning/results/backbone_eval_cpu.json
"""

import argparse
import inspect
import json
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from sgmse.backbones import BackboneRegistry


def _parse_json_kwargs(raw: Optional[str]) -> Dict[str, Any]:
    if raw is None or raw.strip() == "":
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for --model_kwargs_json: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("--model_kwargs_json must be a JSON object, e.g. '{\"nf\":64}'")
    return parsed


def _parse_backbones(raw: Optional[str], use_all: bool) -> List[str]:
    all_names = sorted(BackboneRegistry.get_all_names())
    if use_all:
        return all_names
    if raw is None or raw.strip() == "":
        raise ValueError("Please provide --backbones or set --all_backbones.")
    names = [name.strip() for name in raw.split(",") if name.strip()]
    if not names:
        raise ValueError("No valid backbone names found in --backbones.")
    invalid = [name for name in names if name not in all_names]
    if invalid:
        raise ValueError(
            f"Unknown backbone(s): {invalid}. Registered backbones: {all_names}"
        )
    return names


def _build_complex_tensor(batch: int, channels: int, freq_bins: int, frames: int) -> torch.Tensor:
    real = torch.randn(batch, channels, freq_bins, frames, dtype=torch.float32)
    imag = torch.randn(batch, channels, freq_bins, frames, dtype=torch.float32)
    return torch.complex(real, imag)


def _build_call_inputs(
    model: nn.Module,
    batch_size: int,
    freq_bins: int,
    frames: int,
) -> Tuple[List[Any], Dict[str, Any]]:
    """Build forward inputs for backbone variants using forward signature heuristics.

    Supported common signatures in this repo:
    - forward(x, y, t)
    - forward(x, t) / forward(spec, t)
    - forward(x)
    """
    sig = inspect.signature(model.forward)
    params = [
        p for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    names = [p.name for p in params]

    x_single = _build_complex_tensor(batch_size, 1, freq_bins, frames)
    x_pair = _build_complex_tensor(batch_size, 2, freq_bins, frames)
    t = torch.rand(batch_size, dtype=torch.float32)

    has_y = "y" in names
    args: List[Any] = []
    kwargs: Dict[str, Any] = {}

    for p in params:
        name = p.name
        if name == "self":
            continue

        if name == "y":
            args.append(x_single)
            continue

        if name in {"t", "time_cond", "timesteps", "sigma", "noise_level"}:
            args.append(t)
            continue

        if name in {"x", "x_t", "spec", "input", "noisy"}:
            # If y exists in signature, x is usually single-channel complex.
            args.append(x_single if has_y else x_pair)
            continue

        # Fallback: fill with t if name suggests time; otherwise tensor input.
        if any(token in name.lower() for token in ["time", "sigma", "step"]):
            args.append(t)
        else:
            args.append(x_single if has_y else x_pair)

    return args, kwargs


def _count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _estimate_flops_with_hooks(model: nn.Module, sample_args: Sequence[Any], sample_kwargs: Dict[str, Any]) -> float:
    """Fallback FLOPs estimator (Conv/Linear only) when profiler FLOPs is unavailable."""
    handles = []
    flops = 0.0

    def hook(mod: nn.Module, _inputs: Tuple[Any, ...], output: Any):
        nonlocal flops
        tensor = output if isinstance(output, torch.Tensor) else None
        if tensor is None:
            return

        if isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if tensor.ndim < 3:
                return
            batch = tensor.shape[0]
            out_channels = tensor.shape[1]
            spatial = 1
            for dim in tensor.shape[2:]:
                spatial *= int(dim)
            kernel = 1
            for k in mod.kernel_size:
                kernel *= int(k)
            in_per_group = mod.in_channels // mod.groups
            flops += 2.0 * batch * out_channels * spatial * in_per_group * kernel
        elif isinstance(mod, nn.Linear):
            if tensor.ndim == 1:
                batch = 1
            else:
                batch = tensor.shape[0]
            flops += 2.0 * batch * mod.in_features * mod.out_features

    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            handles.append(module.register_forward_hook(hook))

    with torch.no_grad():
        _ = model(*sample_args, **sample_kwargs)

    for h in handles:
        h.remove()
    return flops


def _estimate_flops(model: nn.Module, sample_args: Sequence[Any], sample_kwargs: Dict[str, Any]) -> Tuple[float, str]:
    """Return FLOPs per forward and source tag."""
    # Try torch.profiler first (captures lower-level ops beyond simple module hooks).
    try:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
            with torch.no_grad():
                _ = model(*sample_args, **sample_kwargs)
        total_flops = 0.0
        for evt in prof.key_averages():
            value = getattr(evt, "flops", 0)
            if value is not None:
                total_flops += float(value)
        if total_flops > 0:
            return total_flops, "torch.profiler"
    except Exception:
        pass

    return _estimate_flops_with_hooks(model, sample_args, sample_kwargs), "hook_conv_linear"


def _read_rss_bytes() -> int:
    """Read current process RSS from /proc/self/status (Linux)."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    # VmRSS is reported in kB.
                    return int(parts[1]) * 1024
    except Exception:
        return 0
    return 0


@dataclass
class BackboneEvalResult:
    backbone: str
    params_total: int
    params_trainable: int
    flops_per_frame: float
    macs_per_frame: float
    flops_source: str
    warmup_frames: int
    measured_frames: int
    latency_ms_per_frame: List[float]
    latency_ms_avg: float
    latency_ms_std: float
    latency_ms_p50: float
    latency_ms_p90: float
    latency_ms_p95: float
    peak_rss_bytes: int
    peak_rss_mib: float
    peak_rss_delta_bytes: int
    peak_rss_delta_mib: float
    status: str
    error: str = ""


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if q <= 0:
        return float(sorted_values[0])
    if q >= 1:
        return float(sorted_values[-1])
    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    w = pos - lo
    return float(sorted_values[lo] * (1 - w) + sorted_values[hi] * w)


def _result_from_dict(item: Dict[str, Any]) -> BackboneEvalResult:
    return BackboneEvalResult(
        backbone=item.get("backbone", "unknown"),
        params_total=int(item.get("params_total", 0)),
        params_trainable=int(item.get("params_trainable", 0)),
        flops_per_frame=float(item.get("flops_per_frame", 0.0)),
        macs_per_frame=float(item.get("macs_per_frame", 0.0)),
        flops_source=str(item.get("flops_source", "n/a")),
        warmup_frames=int(item.get("warmup_frames", 0)),
        measured_frames=int(item.get("measured_frames", 0)),
        latency_ms_per_frame=list(item.get("latency_ms_per_frame", [])),
        latency_ms_avg=float(item.get("latency_ms_avg", 0.0)),
        latency_ms_std=float(item.get("latency_ms_std", 0.0)),
        latency_ms_p50=float(item.get("latency_ms_p50", 0.0)),
        latency_ms_p90=float(item.get("latency_ms_p90", 0.0)),
        latency_ms_p95=float(item.get("latency_ms_p95", 0.0)),
        peak_rss_bytes=int(item.get("peak_rss_bytes", 0)),
        peak_rss_mib=float(item.get("peak_rss_mib", 0.0)),
        peak_rss_delta_bytes=int(item.get("peak_rss_delta_bytes", 0)),
        peak_rss_delta_mib=float(item.get("peak_rss_delta_mib", 0.0)),
        status=str(item.get("status", "error")),
        error=str(item.get("error", "")),
    )


def _evaluate_one_backbone_in_subprocess(
    backbone_name: str,
    batch_size: int,
    freq_bins: int,
    time_frames: int,
    measured_frames: int,
    warmup_frames: int,
    model_kwargs: Dict[str, Any],
) -> BackboneEvalResult:
    """Run one backbone in an isolated Python process to avoid memory cross-contamination."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_json_path = Path(tmp.name)

    cmd = [
        sys.executable,
        "-m",
        "eval_backbones_cpu",
        "--backbones",
        backbone_name,
        "--batch_size",
        str(batch_size),
        "--freq_bins",
        str(freq_bins),
        "--time_frames",
        str(time_frames),
        "--warmup_frames",
        str(warmup_frames),
        "--frames",
        str(measured_frames),
        "--no_isolated_process",
        "--output_json",
        str(tmp_json_path),
    ]
    if model_kwargs:
        cmd.extend(["--model_kwargs_json", json.dumps(model_kwargs)])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            err_msg = (proc.stderr or proc.stdout or "subprocess evaluation failed").strip()
            return BackboneEvalResult(
                backbone=backbone_name,
                params_total=0,
                params_trainable=0,
                flops_per_frame=0.0,
                macs_per_frame=0.0,
                flops_source="n/a",
                warmup_frames=warmup_frames,
                measured_frames=measured_frames,
                latency_ms_per_frame=[],
                latency_ms_avg=0.0,
                latency_ms_std=0.0,
                latency_ms_p50=0.0,
                latency_ms_p90=0.0,
                latency_ms_p95=0.0,
                peak_rss_bytes=0,
                peak_rss_mib=0.0,
                peak_rss_delta_bytes=0,
                peak_rss_delta_mib=0.0,
                status="error",
                error=f"isolated subprocess failed: {err_msg}",
            )

        payload = json.loads(tmp_json_path.read_text(encoding="utf-8"))
        result_items = payload.get("results", [])
        if not result_items:
            raise ValueError("isolated subprocess produced empty results")
        return _result_from_dict(result_items[0])
    except Exception as exc:
        return BackboneEvalResult(
            backbone=backbone_name,
            params_total=0,
            params_trainable=0,
            flops_per_frame=0.0,
            macs_per_frame=0.0,
            flops_source="n/a",
            warmup_frames=warmup_frames,
            measured_frames=measured_frames,
            latency_ms_per_frame=[],
            latency_ms_avg=0.0,
            latency_ms_std=0.0,
            latency_ms_p50=0.0,
            latency_ms_p90=0.0,
            latency_ms_p95=0.0,
            peak_rss_bytes=0,
            peak_rss_mib=0.0,
            peak_rss_delta_bytes=0,
            peak_rss_delta_mib=0.0,
            status="error",
            error=f"isolated subprocess exception: {exc}",
        )
    finally:
        try:
            tmp_json_path.unlink(missing_ok=True)
        except Exception:
            pass


def evaluate_one_backbone(
    backbone_name: str,
    model_kwargs: Dict[str, Any],
    batch_size: int,
    freq_bins: int,
    time_frames: int,
    measured_frames: int,
    warmup_frames: int,
) -> BackboneEvalResult:
    torch.set_num_threads(max(torch.get_num_threads(), 1))
    torch.set_grad_enabled(False)

    baseline_rss = _read_rss_bytes()

    try:
        backbone_cls = BackboneRegistry.get_by_name(backbone_name)
        model = backbone_cls(**model_kwargs).to("cpu").eval()

        params_total, params_trainable = _count_parameters(model)
        args, kwargs = _build_call_inputs(
            model=model,
            batch_size=batch_size,
            freq_bins=freq_bins,
            frames=time_frames,
        )

        flops_per_frame, flops_source = _estimate_flops(model, args, kwargs)
        macs_per_frame = flops_per_frame / 2.0

        with torch.no_grad():
            for _ in range(warmup_frames):
                _ = model(*args, **kwargs)

        latencies_ms: List[float] = []
        peak_rss = _read_rss_bytes()

        with torch.no_grad():
            for _ in range(measured_frames):
                rss_before = _read_rss_bytes()
                start = time.perf_counter()
                _ = model(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                latencies_ms.append(elapsed_ms)
                rss_after = _read_rss_bytes()
                peak_rss = max(peak_rss, rss_before, rss_after)

        sorted_lat = sorted(latencies_ms)
        avg = statistics.mean(latencies_ms) if latencies_ms else 0.0
        std = statistics.pstdev(latencies_ms) if len(latencies_ms) > 1 else 0.0

        peak_delta = max(0, peak_rss - baseline_rss)
        return BackboneEvalResult(
            backbone=backbone_name,
            params_total=params_total,
            params_trainable=params_trainable,
            flops_per_frame=flops_per_frame,
            macs_per_frame=macs_per_frame,
            flops_source=flops_source,
            warmup_frames=warmup_frames,
            measured_frames=measured_frames,
            latency_ms_per_frame=latencies_ms,
            latency_ms_avg=avg,
            latency_ms_std=std,
            latency_ms_p50=_percentile(sorted_lat, 0.5),
            latency_ms_p90=_percentile(sorted_lat, 0.9),
            latency_ms_p95=_percentile(sorted_lat, 0.95),
            peak_rss_bytes=peak_rss,
            peak_rss_mib=peak_rss / (1024.0 * 1024.0),
            peak_rss_delta_bytes=peak_delta,
            peak_rss_delta_mib=peak_delta / (1024.0 * 1024.0),
            status="ok",
        )

    except Exception as exc:
        return BackboneEvalResult(
            backbone=backbone_name,
            params_total=0,
            params_trainable=0,
            flops_per_frame=0.0,
            macs_per_frame=0.0,
            flops_source="n/a",
            warmup_frames=warmup_frames,
            measured_frames=measured_frames,
            latency_ms_per_frame=[],
            latency_ms_avg=0.0,
            latency_ms_std=0.0,
            latency_ms_p50=0.0,
            latency_ms_p90=0.0,
            latency_ms_p95=0.0,
            peak_rss_bytes=0,
            peak_rss_mib=0.0,
            peak_rss_delta_bytes=0,
            peak_rss_delta_mib=0.0,
            status="error",
            error=str(exc),
        )


def _human_number(value: float) -> str:
    if abs(value) >= 1e9:
        return f"{value / 1e9:.3f}G"
    if abs(value) >= 1e6:
        return f"{value / 1e6:.3f}M"
    if abs(value) >= 1e3:
        return f"{value / 1e3:.3f}K"
    return f"{value:.3f}"


def _print_summary(results: Iterable[BackboneEvalResult]):
    rows = list(results)
    print("\n===== CPU Backbone Evaluation Summary =====")
    print(
        "| backbone | status | params | FLOPs/frame | MACs/frame | latency_avg(ms) | p95(ms) | peak RSS delta(MiB) |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for item in rows:
        print(
            "| {backbone} | {status} | {params} | {flops} | {macs} | {lat_avg:.3f} | {p95:.3f} | {peak_delta:.3f} |".format(
                backbone=item.backbone,
                status=item.status,
                params=_human_number(float(item.params_total)),
                flops=_human_number(item.flops_per_frame),
                macs=_human_number(item.macs_per_frame),
                lat_avg=item.latency_ms_avg,
                p95=item.latency_ms_p95,
                peak_delta=item.peak_rss_delta_mib,
            )
        )

    failed = [r for r in rows if r.status != "ok"]
    if failed:
        print("\nFailed backbones:")
        for item in failed:
            print(f"- {item.backbone}: {item.error}")


def _build_markdown_report(
    results: Sequence[BackboneEvalResult],
    batch_size: int,
    freq_bins: int,
    time_frames: int,
    warmup_frames: int,
    measured_frames: int,
    model_kwargs: Dict[str, Any],
) -> str:
    lines: List[str] = []
    lines.append("# CPU Backbone Evaluation Report")
    lines.append("")
    lines.append("## Run config")
    lines.append("")
    lines.append("| key | value |")
    lines.append("|---|---|")
    lines.append(f"| batch_size | {batch_size} |")
    lines.append(f"| freq_bins | {freq_bins} |")
    lines.append(f"| time_frames | {time_frames} |")
    lines.append(f"| warmup_frames | {warmup_frames} |")
    lines.append(f"| measured_frames | {measured_frames} |")
    lines.append(f"| model_kwargs | {json.dumps(model_kwargs, ensure_ascii=False)} |")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| backbone | status | params_total | params_trainable | FLOPs/frame | MACs/frame | latency_avg_ms | latency_p95_ms | peak_rss_delta_mib | flops_source |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for item in results:
        lines.append(
            "| {backbone} | {status} | {params_total} | {params_trainable} | {flops} | {macs} | {lat_avg:.3f} | {lat_p95:.3f} | {rss_delta:.3f} | {flops_source} |".format(
                backbone=item.backbone,
                status=item.status,
                params_total=item.params_total,
                params_trainable=item.params_trainable,
                flops=item.flops_per_frame,
                macs=item.macs_per_frame,
                lat_avg=item.latency_ms_avg,
                lat_p95=item.latency_ms_p95,
                rss_delta=item.peak_rss_delta_mib,
                flops_source=item.flops_source,
            )
        )

    lines.append("")
    lines.append("## Failures")
    lines.append("")
    failed = [x for x in results if x.status != "ok"]
    if not failed:
        lines.append("None")
    else:
        for item in failed:
            lines.append(f"- {item.backbone}: {item.error}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate registered backbones on CPU: params/FLOPs/MACs/latency/peak memory"
    )
    parser.add_argument("--backbones", type=str, default=None, help="Comma-separated backbone names")
    parser.add_argument("--all_backbones", action="store_true", help="Evaluate all registered backbones")
    parser.add_argument("--list_backbones", action="store_true", help="Only print registered backbones and exit")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--freq_bins", type=int, default=257, help="Input frequency bins")
    parser.add_argument("--time_frames", type=int, default=256, help="Input time frames")
    parser.add_argument("--warmup_frames", type=int, default=4)
    parser.add_argument("--frames", type=int, default=20, help="Measured frames after warmup")
    parser.add_argument(
        "--model_kwargs_json",
        type=str,
        default=None,
        help='JSON object passed to each backbone constructor, e.g. {"nf":64,"num_res_blocks":1}',
    )
    parser.add_argument("--output_json", type=str, default=None, help="Optional path to save detailed JSON report")
    parser.add_argument("--output_md", type=str, default=None, help="Optional path to save markdown report")
    parser.add_argument(
        "--no_isolated_process",
        action="store_true",
        help="Disable per-backbone isolated subprocess evaluation (for debugging only)",
    )
    args = parser.parse_args()

    if args.list_backbones:
        names = sorted(BackboneRegistry.get_all_names())
        print("Registered backbones:")
        for name in names:
            print(name)
        return

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be > 0")
    if args.freq_bins <= 0:
        raise ValueError("--freq_bins must be > 0")
    if args.time_frames <= 0:
        raise ValueError("--time_frames must be > 0")
    if args.warmup_frames < 0:
        raise ValueError("--warmup_frames must be >= 0")
    if args.frames <= 0:
        raise ValueError("--frames must be > 0")

    backbones = _parse_backbones(args.backbones, args.all_backbones)
    model_kwargs = _parse_json_kwargs(args.model_kwargs_json)

    print("Running CPU backbone evaluation with settings:")
    print(
        f"- backbones={backbones}, batch_size={args.batch_size}, freq_bins={args.freq_bins}, time_frames={args.time_frames}, warmup_frames={args.warmup_frames}, frames={args.frames}"
    )
    if model_kwargs:
        print(f"- model_kwargs={model_kwargs}")

    results: List[BackboneEvalResult] = []
    for name in backbones:
        print(f"\n[Eval] {name}")
        if args.no_isolated_process:
            res = evaluate_one_backbone(
                backbone_name=name,
                model_kwargs=model_kwargs,
                batch_size=args.batch_size,
                freq_bins=args.freq_bins,
                time_frames=args.time_frames,
                measured_frames=args.frames,
                warmup_frames=args.warmup_frames,
            )
        else:
            res = _evaluate_one_backbone_in_subprocess(
                backbone_name=name,
                batch_size=args.batch_size,
                freq_bins=args.freq_bins,
                time_frames=args.time_frames,
                measured_frames=args.frames,
                warmup_frames=args.warmup_frames,
                model_kwargs=model_kwargs,
            )
        print(
            f"  status={res.status}, params={res.params_total}, avg_latency_ms={res.latency_ms_avg:.3f}, peak_rss_delta_mib={res.peak_rss_delta_mib:.3f}"
        )
        results.append(res)

    _print_summary(results)

    if args.output_json:
        out_path = Path(args.output_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "device": "cpu",
            "batch_size": args.batch_size,
            "freq_bins": args.freq_bins,
            "time_frames": args.time_frames,
            "warmup_frames": args.warmup_frames,
            "frames": args.frames,
            "model_kwargs": model_kwargs,
            "results": [asdict(item) for item in results],
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nDetailed JSON written to: {out_path}")

    md_path: Optional[Path] = None
    if args.output_md:
        md_path = Path(args.output_md).expanduser().resolve()
    elif args.all_backbones:
        # In full-registry mode, generate a markdown file by default for quick comparison.
        md_path = Path("./backbone_eval_cpu_all.md").expanduser().resolve()

    if md_path is not None:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_text = _build_markdown_report(
            results=results,
            batch_size=args.batch_size,
            freq_bins=args.freq_bins,
            time_frames=args.time_frames,
            warmup_frames=args.warmup_frames,
            measured_frames=args.frames,
            model_kwargs=model_kwargs,
        )
        md_path.write_text(md_text, encoding="utf-8")
        print(f"Markdown report written to: {md_path}")


if __name__ == "__main__":
    main()
