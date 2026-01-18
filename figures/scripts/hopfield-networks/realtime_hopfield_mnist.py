from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch

from hopfield import HopfieldNetwork, add_flip_noise, to_pm_one


def _ensure_mpl_cache_dir() -> None:
    if os.environ.get("MPLCONFIGDIR"):
        return
    script_dir = Path(__file__).resolve().parent
    cache_dir = script_dir / ".cache/matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir.resolve())


_ensure_mpl_cache_dir()

import matplotlib.pyplot as plt  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-time Hopfield recall on binarized MNIST patterns.")
    script_dir = Path(__file__).resolve().parent
    p.add_argument(
        "--weights",
        type=Path,
        default=script_dir / "artifacts/hopfield_mnist.pt",
        help="Weights .pt file.",
    )

    p.add_argument("--rule", choices=("sync", "async"), default="async", help="Update rule.")
    p.add_argument("--async-updates", type=int, default=20, help="Async updates per step (rule=async).")
    p.add_argument("--steps", type=int, default=1_000, help="Max update steps per demo.")
    p.add_argument(
        "--stop-when-stable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop early when the state stops changing.",
    )

    p.add_argument("--noise", type=float, default=0.20, help="Bit-flip probability applied to the clean pattern.")
    p.add_argument("--pattern-idx", type=int, default=-1, help="Which stored pattern to recall (-1 for random).")
    p.add_argument("--demos", type=int, default=1, help="How many demos to run (0 = infinite).")

    p.add_argument("--interval-ms", type=int, default=125, help="Delay between frames in milliseconds.")
    p.add_argument("--pause-ms", type=int, default=500, help="Pause after a demo completes (ms).")
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for sampling and noise (omit for random).",
    )
    return p.parse_args()


def _pmone_to_img01(state_pm_one: torch.Tensor, image_shape: tuple[int, int]) -> torch.Tensor:
    img = to_pm_one(state_pm_one).view(*image_shape)
    return (img + 1.0) / 2.0


def main() -> None:
    args = _parse_args()
    ckpt = torch.load(args.weights, map_location="cpu")

    W = ckpt["W"].to(torch.float32)
    theta = ckpt.get("theta", torch.zeros((W.shape[0],), dtype=torch.float32)).to(torch.float32)
    patterns = ckpt.get("patterns")
    labels = ckpt.get("labels")
    image_shape = tuple(int(x) for x in ckpt.get("image_shape", (28, 28)))
    method = str(ckpt.get("method", "unknown"))
    if method == "hebbian":
        print(
            "Note: weights were trained with Hebbian learning; for binarized MNIST, this often recalls a blank/spurious attractor.\n"
            "Re-train with: python train_hopfield_mnist.py --method pinv\n"
        )

    if patterns is None:
        raise ValueError(f"{args.weights} is missing 'patterns'. Re-train with train_hopfield_mnist.py.")
    patterns = patterns.to(torch.float32)

    net = HopfieldNetwork(n_units=int(W.shape[0]))
    net.W = W
    net.theta = theta

    seed = int.from_bytes(os.urandom(8), "little") if args.seed is None else int(args.seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    plt.ion()
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    im_target = axes[0].imshow(torch.zeros(image_shape), cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Stored")
    im_noisy = axes[1].imshow(torch.zeros(image_shape), cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("Noisy input")
    im_state = axes[2].imshow(torch.zeros(image_shape), cmap="gray", vmin=0.0, vmax=1.0)
    axes[2].set_title("Reconstruction")

    demo = 0
    while args.demos == 0 or demo < args.demos:
        if not plt.fignum_exists(fig.number):
            break

        if args.pattern_idx >= 0:
            idx = int(args.pattern_idx)
        else:
            idx = int(torch.randint(0, int(patterns.shape[0]), (1,), generator=generator).item())

        target = patterns[idx]
        noisy = add_flip_noise(target, float(args.noise), generator=generator)
        state = noisy.clone()

        im_target.set_data(_pmone_to_img01(target, image_shape).cpu().numpy())
        im_noisy.set_data(_pmone_to_img01(noisy, image_shape).cpu().numpy())
        im_state.set_data(_pmone_to_img01(state, image_shape).cpu().numpy())

        label_str = ""
        if labels is not None:
            label_str = f" | label={int(labels[idx])}"
        fig.suptitle(
            f"pattern={idx}{label_str} | method={method} | rule={args.rule} | noise={args.noise:.2f} | seed={seed}"
        )
        fig.canvas.draw_idle()
        plt.pause(max(0.001, args.interval_ms / 1000.0))

        for step in range(1, int(args.steps) + 1):
            if not plt.fignum_exists(fig.number):
                break

            next_state = net.step(
                state,
                rule=args.rule,
                async_updates=int(args.async_updates),
                generator=generator,
            )
            state_changed = not torch.equal(next_state, state)
            state = next_state

            energy = float(net.energy(state).item())
            hamming = float((state != target).to(torch.float32).mean().item())
            axes[2].set_title(f"Step {step} | E={energy:.1f} | err={hamming:.3f}")
            im_state.set_data(_pmone_to_img01(state, image_shape).cpu().numpy())
            fig.canvas.draw_idle()
            plt.pause(max(0.001, args.interval_ms / 1000.0))

            if args.stop_when_stable and (not state_changed):
                if args.rule == "async":
                    # For async updates over a small random subset, "no change" does not necessarily
                    # mean we're at a fixed point. Verify by checking the full synchronous update.
                    if torch.equal(net.step(state, rule="sync"), state):
                        break
                else:
                    break

        time.sleep(max(0.0, args.pause_ms / 1000.0))
        demo += 1

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
