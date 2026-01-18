# Hopfield Network on Binarized MNIST

## Train

```bash
python train_hopfield_mnist.py --n-patterns 100 --balanced --method pinv --out artifacts/hopfield_mnist.pt
```

Notes:
- Stores patterns in `{-1, +1}`.
- For binarized MNIST, `--method pinv` (pseudoinverse) is recommended; `--method hebbian` often collapses to a blank attractor.
- MNIST is downloaded to `data/mnist` by default (use `--no-download` if you already have it).
- If you see `CERTIFICATE_VERIFY_FAILED` on macOS, run the bundled `Install Certificates.command` for your Python install (or set `SSL_CERT_FILE` to `certifi.where()`).

## Real-time recall (interactive window)

```bash
python realtime_hopfield_mnist.py --weights artifacts/hopfield_mnist.pt --rule async --async-updates 20 --noise 0.20 --steps 100
```

Try:
- `--rule async --async-updates 200` for asynchronous dynamics
- `--pattern-idx 0` to recall a specific stored pattern
- `--demos 0` to loop forever (close the window to stop)
- `--seed 0` for reproducible randomness
