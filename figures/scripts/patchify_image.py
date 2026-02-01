from PIL import Image, ImageDraw, ImageFont

def visualize_patch_tokens_separated(
    in_path: str,
    out_path: str,
    n: int = 4,
    gap: int = 24,                   # ↑ increase this for more separation
    bg=(245, 245, 245),
    add_border: bool = True,
    border_width: int = 2,
    border_color=(30, 30, 30),
    label: bool = True
):
    img = Image.open(in_path).convert("RGB")
    W, H = img.size

    xs = [round(i * W / n) for i in range(n + 1)]
    ys = [round(i * H / n) for i in range(n + 1)]

    canvas = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    def split_gap(i: int, last: int):
        # distribute the full `gap` between neighboring patches; none on exterior edges
        if i == 0:
            return 0, gap // 2
        if i == last:
            return gap - gap // 2, 0
        return gap - gap // 2, gap // 2

    token_id = 0
    for r in range(n):
        for c in range(n):
            x0, x1 = xs[c], xs[c + 1]
            y0, y1 = ys[r], ys[r + 1]

            gl, gr = split_gap(c, n - 1)
            gt, gb = split_gap(r, n - 1)

            ix0, ix1 = x0 + gl, x1 - gr
            iy0, iy1 = y0 + gt, y1 - gb

            if ix1 <= ix0 or iy1 <= iy0:
                raise ValueError(
                    f"gap={gap} too large for image {W}x{H} with n={n}. "
                    f"Try smaller gap (e.g., {max(2, min(W, H)//(2*n))})."
                )

            patch = img.crop((x0, y0, x1, y1)).resize(
                (ix1 - ix0, iy1 - iy0),
                resample=Image.BICUBIC
            )
            canvas.paste(patch, (ix0, iy0))

            if add_border:
                for k in range(border_width):
                    draw.rectangle([ix0 + k, iy0 + k, ix1 - 1 - k, iy1 - 1 - k], outline=border_color)

            if label:
                text = str(token_id)
                token_id += 1
                cx, cy = (ix0 + ix1) // 2, (iy0 + iy1) // 2
                bbox = draw.textbbox((0, 0), text, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                pad = 3
                draw.rectangle(
                    [cx - tw // 2 - pad, cy - th // 2 - pad, cx + (tw + 1) // 2 + pad, cy + (th + 1) // 2 + pad],
                    fill=(255, 255, 255)
                )
                draw.text((cx - tw // 2, cy - th // 2), text, fill=(0, 0, 0), font=font)

    canvas.save(out_path)


if __name__ == "__main__":
    # Increase gap further (e.g., 32, 40) for even more separation.
    visualize_patch_tokens_separated("augusto-sleeping.jpg", "tokenized_4x4.png", n=4, gap=32, bg=(245, 245, 245))
