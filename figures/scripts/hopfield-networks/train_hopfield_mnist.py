from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torchvision.datasets import MNIST

from hopfield import HopfieldNetwork, binarize


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Hopfield network on binarized MNIST.")
    script_dir = Path(__file__).resolve().parent
    p.add_argument("--data-dir", type=Path, default=script_dir / "data/mnist", help="MNIST root directory.")
    p.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download MNIST if missing.",
    )
    p.add_argument("--split", choices=("train", "test"), default="train", help="Which MNIST split to sample from.")

    p.add_argument("--n-patterns", type=int, default=100, help="Number of patterns to store.")
    p.add_argument(
        "--balanced",
        action="store_true",
        help="Sample roughly equal patterns per digit (0-9).",
    )
    p.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold in [0, 1].")
    p.add_argument("--seed", type=int, default=0, help="Sampling RNG seed.")

    p.add_argument(
        "--method",
        choices=("pinv", "hebbian"),
        default="pinv",
        help="Training rule: pseudoinverse ('pinv') is recommended for binarized MNIST.",
    )
    p.add_argument(
        "--pinv-ridge",
        type=float,
        default=1e-4,
        help="Ridge added to Gram matrix for pinv stability (method=pinv).",
    )

    p.add_argument(
        "--out",
        type=Path,
        default=script_dir / "artifacts/hopfield_mnist.pt",
        help="Output .pt file path.",
    )
    return p.parse_args()


def _sample_indices(targets: torch.Tensor, n_patterns: int, *, balanced: bool, generator: torch.Generator) -> torch.Tensor:
    if n_patterns <= 0:
        raise ValueError(f"n_patterns must be > 0, got {n_patterns}")
    if not balanced:
        return torch.randperm(int(targets.numel()), generator=generator)[:n_patterns]

    per_class = [n_patterns // 10 for _ in range(10)]
    for digit in range(n_patterns % 10):
        per_class[digit] += 1

    out: list[torch.Tensor] = []
    for digit, count in enumerate(per_class):
        if count == 0:
            continue
        idx = (targets == digit).nonzero(as_tuple=False).view(-1)
        if idx.numel() < count:
            raise ValueError(f"Not enough samples for digit={digit}: requested {count}, found {idx.numel()}")
        perm = torch.randperm(int(idx.numel()), generator=generator)[:count]
        out.append(idx[perm])

    return torch.cat(out, dim=0)


def _configure_ssl_certificates() -> None:
    """
    Work around common macOS Python SSL issues when downloading MNIST.

    Some Python distributions (notably python.org builds on macOS) can fail with
    CERTIFICATE_VERIFY_FAILED unless a CA bundle is configured.
    """
    if os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE"):
        return
    try:
        import certifi  # type: ignore
    except Exception:
        return

    cafile = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", cafile)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", cafile)

    try:
        import ssl

        ctx = ssl.create_default_context(cafile=cafile)
        ssl._create_default_https_context = lambda: ctx  # type: ignore[attr-defined]
    except Exception:
        # Env vars above may still be enough for some setups.
        pass


def main() -> None:
    args = _parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    train = args.split == "train"
    _configure_ssl_certificates()
    try:
        ds = MNIST(root=str(args.data_dir), train=train, download=bool(args.download))
    except RuntimeError as e:
        msg = str(e)
        if "CERTIFICATE_VERIFY_FAILED" in msg:
            raise RuntimeError(
                "MNIST download failed due to SSL certificate verification.\n"
                "Fix options:\n"
                "  1) On macOS (python.org install), run the bundled 'Install Certificates.command'.\n"
                "  2) Ensure certifi is installed: python -m pip install certifi\n"
                "  3) Or set SSL_CERT_FILE to certifi's CA bundle, e.g.:\n"
                "       export SSL_CERT_FILE=\"$(python -c 'import certifi; print(certifi.where())')\"\n"
                "  4) Or download MNIST manually and re-run with --no-download.\n"
            ) from e
        raise

    indices = _sample_indices(ds.targets, args.n_patterns, balanced=bool(args.balanced), generator=generator)
    # Shuffle indices so "balanced" doesn't group digits together.
    indices = indices[torch.randperm(int(indices.numel()), generator=generator)]

    images01 = (ds.data[indices].unsqueeze(1).to(torch.float32) / 255.0).contiguous()
    labels = ds.targets[indices].to(torch.int64).contiguous()

    patterns = binarize(images01, threshold=float(args.threshold), low=-1.0, high=1.0).view(images01.shape[0], -1)
    net = HopfieldNetwork(n_units=int(patterns.shape[1]))
    if args.method == "hebbian":
        net.fit_hebbian(patterns)
    elif args.method == "pinv":
        net.fit_pseudoinverse(patterns, ridge=float(args.pinv_ridge))
    else:
        raise ValueError(f"Unknown method: {args.method}")

    ckpt = {
        "version": 1,
        "method": str(args.method),
        "pinv_ridge": float(args.pinv_ridge) if args.method == "pinv" else None,
        "split": args.split,
        "threshold": float(args.threshold),
        "seed": int(args.seed),
        "balanced": bool(args.balanced),
        "image_shape": (28, 28),
        "indices": indices.to(torch.int64),
        "labels": labels,
        "patterns": patterns.to(torch.int8),
        "W": net.W.cpu(),
        "theta": net.theta.cpu(),
    }
    torch.save(ckpt, args.out)

    print(f"Saved: {args.out}")
    print(
        f"Method: {args.method} | Patterns: {int(patterns.shape[0])} | Units: {int(patterns.shape[1])} | W: {tuple(net.W.shape)}"
    )


if __name__ == "__main__":
    main()
