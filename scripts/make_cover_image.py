"""Generates the project cover image (docs/cover.png), 600x300 as OpenUPM expects.

Kept as a script rather than a one-off export so the image can be regenerated when the numbers
or wording change, instead of drifting out of date in a binary nobody can edit.

Rendered at 4x and downsampled, which is the cheapest way to get clean antialiasing out of PIL's
integer-coordinate drawing.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, SS = 600, 300, 4          # final size, supersample factor
FONT_DIR = Path("C:/Windows/Fonts")

BG_TOP = (13, 17, 23)
BG_BOTTOM = (22, 29, 43)
TRACKED = (56, 208, 248)        # measured: head and both hands
INFERRED = (118, 138, 168)      # everything the model has to work out
BONE = (72, 88, 112)
TEXT = (236, 242, 250)
MUTED = (139, 155, 178)

# A relaxed standing pose in metres, one arm raised -- a T-pose reads as a rig screenshot rather
# than something in use. Values follow the same proportions as the package's demo avatar.
JOINTS = {
    "hips": (0.00, 0.95), "spine": (0.00, 1.08), "chest": (0.00, 1.24),
    "neck": (0.00, 1.42), "head": (0.00, 1.57),
    "l_shoulder": (-0.09, 1.36), "l_elbow": (-0.24, 1.11), "l_hand": (-0.31, 0.84),
    "r_shoulder": (0.09, 1.36), "r_elbow": (0.27, 1.19), "r_hand": (0.43, 1.44),
    "l_hip": (-0.09, 0.90), "l_knee": (-0.12, 0.50), "l_foot": (-0.12, 0.06),
    "r_hip": (0.09, 0.90), "r_knee": (0.10, 0.50), "r_foot": (0.10, 0.06),
}
BONES = [
    ("hips", "spine"), ("spine", "chest"), ("chest", "neck"), ("neck", "head"),
    ("chest", "l_shoulder"), ("l_shoulder", "l_elbow"), ("l_elbow", "l_hand"),
    ("chest", "r_shoulder"), ("r_shoulder", "r_elbow"), ("r_elbow", "r_hand"),
    ("hips", "l_hip"), ("l_hip", "l_knee"), ("l_knee", "l_foot"),
    ("hips", "r_hip"), ("r_hip", "r_knee"), ("r_knee", "r_foot"),
]
TRACKED_JOINTS = {"head", "l_hand", "r_hand"}


def font(name, size):
    for candidate in (name, "arialbd.ttf", "arial.ttf"):
        path = FONT_DIR / candidate
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def background(size):
    w, h = size
    img = Image.new("RGB", (w, h), BG_TOP)
    draw = ImageDraw.Draw(img)
    for y in range(h):
        t = y / max(1, h - 1)
        draw.line([(0, y), (w, y)],
                  fill=tuple(round(a + (b - a) * t) for a, b in zip(BG_TOP, BG_BOTTOM)))
    return img


def draw_figure(draw, cx, base_y, scale):
    """Maps figure-space metres to pixels, y flipped so the feet sit at base_y."""
    def px(joint):
        x, y = JOINTS[joint]
        return cx + x * scale, base_y - y * scale

    for a, b in BONES:
        draw.line([px(a), px(b)], fill=BONE, width=max(1, round(scale * 0.020)))

    for name in JOINTS:
        x, y = px(name)
        if name in TRACKED_JOINTS:
            r = scale * 0.055
            # Ring plus core, so the measured inputs read as instrumented rather than just bigger.
            draw.ellipse([x - r * 1.9, y - r * 1.9, x + r * 1.9, y + r * 1.9], outline=TRACKED,
                         width=max(1, round(scale * 0.010)))
            draw.ellipse([x - r, y - r, x + r, y + r], fill=TRACKED)
        else:
            r = scale * 0.026
            draw.ellipse([x - r, y - r, x + r, y + r], fill=INFERRED)


def main():
    img = background((W * SS, H * SS))
    glow = Image.new("RGB", img.size, (0, 0, 0))

    s = SS  # every literal below is in final-image pixels
    draw_figure(ImageDraw.Draw(glow), cx=112 * s, base_y=272 * s, scale=145 * s)
    img = Image.blend(img, img.point(lambda v: v), 0)  # keep type stable
    img.paste(Image.blend(img, glow.filter(ImageFilter.GaussianBlur(9 * s)), 0.55), (0, 0))
    draw_figure(ImageDraw.Draw(img), cx=112 * s, base_y=272 * s, scale=145 * s)

    d = ImageDraw.Draw(img)
    x = 232 * s

    d.text((x, 74 * s), "QUDMI", font=font("arialbd.ttf", 54 * s), fill=TEXT)
    d.text((x + 208 * s, 92 * s), "FULL BODY", font=font("arialbd.ttf", 21 * s), fill=TRACKED)

    d.text((x, 140 * s), "Full-body VR avatars from a headset", font=font("arial.ttf", 19 * s), fill=MUTED)
    d.text((x, 164 * s), "and two controllers. No extra trackers.", font=font("arial.ttf", 19 * s), fill=MUTED)

    d.line([(x, 200 * s), (x + 44 * s, 200 * s)], fill=TRACKED, width=2 * s)

    # Laid out from measured text widths rather than guessed offsets, which is what previously
    # made the arrow overlap the label next to it.
    label_font = font("arialbd.ttf", 17 * s)
    value_font = font("arial.ttf", 17 * s)
    left_label = "3 tracked points"
    gap = 12 * s

    d.text((x, 218 * s), left_label, font=label_font, fill=TRACKED)
    ax = x + d.textlength(left_label, font=label_font) + gap
    ay = 227 * s
    d.line([(ax, ay), (ax + 14 * s, ay)], fill=MUTED, width=round(1.5 * s))
    d.polygon([(ax + 14 * s, ay - 3.5 * s), (ax + 21 * s, ay), (ax + 14 * s, ay + 3.5 * s)], fill=MUTED)
    d.text((ax + 21 * s + gap, 218 * s), "22 body joints", font=value_font, fill=TEXT)

    d.text((x, 246 * s), "on-device  ·  real time  ·  open source", font=font("arial.ttf", 15 * s), fill=MUTED)

    out = Path("docs/cover.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    img.resize((W, H), Image.LANCZOS).save(out, optimize=True)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
