#!/usr/bin/env python3
"""Generate arifhaxn-style animated profile banner SVGs (dark.svg / light.svg)."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from scipy.cluster.vq import kmeans2
from scipy.ndimage import binary_closing, binary_fill_holes, label
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent
PHOTO = ROOT / "assets" / "portrait.png"
LOGO_CODE = ROOT / "assets" / "logos" / "code.png"
LOGO_NEXT = ROOT / "assets" / "logos" / "nextjs.png"
LOGO_NODE = ROOT / "assets" / "logos" / "nodejs.png"
OUT_DARK = ROOT / "dark.svg"
OUT_LIGHT = ROOT / "light.svg"
DATA_DIR = ROOT / "banner_data"

GRID_W, GRID_H = 300, 340
NUM_TRAVELLERS = 900
NUM_DRIFT_BANDS = 94
INTRO_GROUPS = 60
NOISE_SIGMA = 4.0

KEY_TIMES = "0.000;0.194;0.288;0.432;0.525;0.669;0.763;0.906;1.000"
DUR = "13.9s"
BEGIN = "3.2s"
OP_PORTRAIT = "1;1;0;0;0;0;0;0;1"
OP_PARTICLE = "0;0;1;1;1;1;1;1;0"
VISUAL_TRANSFORM = "translate(50,86) scale(1.2400,1.4471)"

PROFILE = {
    "subject": "Nicarlo Junior",
    "role": "Full-Stack Developer",
    "origin": "João Pessoa, PB",
    "education": "UNIPÊ · ADS",
    "status": "Building + Learning + Shipping",
    "toolchain": "VS Code · Git",
    "core_lang": "JavaScript · TypeScript · Python",
    "core_frontend": "React · Next.js",
    "core_backend": "Node.js",
    "core_database": "MongoDB · Firebase",
    "core_infra": "Docker · Git · Vercel",
    "mail": "nicarlo.sdlj@gmail.com",
    "portfolio": "coming soon",
    "linkedin": "nicarlo-junior-85b496200",
    "github": "@nicarlojunior",
}

THEMES = {
    "dark": {
        "portrait": "#A78BFA",
        "tv_id": "tvdark",
        "aria": "Nicarlo Junior — profile.sh --live",
        "outer_bg": "#070B16",
        "title_bar": "#0B1222",
        "title_text": "#94A3B8",
        "visual_label": "#475569",
        "frame_stroke": "#22D3EE",
        "frame_fill": "#0A101F",
        "frame_stroke_alpha": "rgba(34,211,238,0.35)",
        "corner": "#22D3EE",
        "info_title": "#22D3EE",
        "info_line": "rgba(255,255,255,0.10)",
        "live": "#F87171",
        "email_bg": "#4C1D95",
        "email_text": "#E9D5FF",
        "label": "#22D3EE",
        "dots": "rgba(148,163,184,0.35)",
        "value": "#F8FAFC",
        "contact": "#94A3B8",
        "footer": "#94A3B8",
        "cursor": "#22D3EE",
        "panel_grad": ('<linearGradient id="panelGrad" x1="0" y1="0" x2="0" y2="1">'
                       '<stop offset="0" stop-color="#0A101F"/><stop offset="1" stop-color="#0C1426"/></linearGradient>'),
        "accent_grad": ('<linearGradient id="accent" x1="0" y1="0" x2="1" y2="0">'
                        '<stop offset="0" stop-color="#7C3AED"><animate attributeName="stop-color" '
                        'values="#7C3AED;#22D3EE;#10B981;#7C3AED" dur="10s" repeatCount="indefinite"/></stop>'
                        '<stop offset="0.5" stop-color="#22D3EE"><animate attributeName="stop-color" '
                        'values="#22D3EE;#10B981;#7C3AED;#22D3EE" dur="10s" repeatCount="indefinite"/></stop>'
                        '<stop offset="1" stop-color="#10B981"><animate attributeName="stop-color" '
                        'values="#10B981;#7C3AED;#22D3EE;#10B981" dur="10s" repeatCount="indefinite"/></stop>'
                        '</linearGradient>'),
        "ascii_grad": ('<linearGradient id="asciiGrad" x1="0" y1="0" x2="0" y2="520" gradientUnits="userSpaceOnUse">'
                       '<stop offset="0" stop-color="#60A5FA"/><stop offset="0.45" stop-color="#A78BFA"/>'
                       '<stop offset="1" stop-color="#22D3EE"/>'
                       '<animateTransform attributeName="gradientTransform" type="translate" '
                       'values="0 -120; 0 120; 0 -120" dur="9s" repeatCount="indefinite"/></linearGradient>'),
    },
    "light": {
        "portrait": "#7C3AED",
        "tv_id": "tvlight",
        "aria": "Nicarlo Junior — profile.sh --live",
        "outer_bg": "#FFFFFF",
        "title_bar": "#F1F5F9",
        "title_text": "#475569",
        "visual_label": "#94A3B8",
        "frame_stroke": "#06B6D4",
        "frame_fill": "#F8FAFC",
        "frame_stroke_alpha": "rgba(8,145,178,0.40)",
        "corner": "#06B6D4",
        "info_title": "#0891B2",
        "info_line": "rgba(15,23,42,0.10)",
        "live": "#DC2626",
        "email_bg": "#DBEAFE",
        "email_text": "#1D4ED8",
        "label": "#0891B2",
        "dots": "rgba(15,23,42,0.25)",
        "value": "#0F172A",
        "contact": "#475569",
        "footer": "#475569",
        "cursor": "#06B6D4",
        "panel_grad": ('<linearGradient id="panelGrad" x1="0" y1="0" x2="0" y2="1">'
                       '<stop offset="0" stop-color="#F8FAFC"/><stop offset="1" stop-color="#EEF2F7"/></linearGradient>'),
        "accent_grad": ('<linearGradient id="accent" x1="0" y1="0" x2="1" y2="0">'
                        '<stop offset="0" stop-color="#2563EB"><animate attributeName="stop-color" '
                        'values="#2563EB;#06B6D4;#10B981;#2563EB" dur="10s" repeatCount="indefinite"/></stop>'
                        '<stop offset="0.5" stop-color="#06B6D4"><animate attributeName="stop-color" '
                        'values="#06B6D4;#10B981;#2563EB;#06B6D4" dur="10s" repeatCount="indefinite"/></stop>'
                        '<stop offset="1" stop-color="#10B981"><animate attributeName="stop-color" '
                        'values="#10B981;#2563EB;#06B6D4;#10B981" dur="10s" repeatCount="indefinite"/></stop>'
                        '</linearGradient>'),
        "ascii_grad": ('<linearGradient id="asciiGrad" x1="0" y1="0" x2="0" y2="520" gradientUnits="userSpaceOnUse">'
                       '<stop offset="0" stop-color="#1D4ED8"/><stop offset="0.45" stop-color="#7C3AED"/>'
                       '<stop offset="1" stop-color="#0891B2"/>'
                       '<animateTransform attributeName="gradientTransform" type="translate" '
                       'values="0 -120; 0 120; 0 -120" dur="9s" repeatCount="indefinite"/></linearGradient>'),
    },
}

INFO_SPEC = [
    (0.90, 162, "Subject", "subject", True),
    (1.02, 185, "Role", "role", True),
    (1.14, 208, "Origin", "origin", True),
    (1.26, 231, "Education", "education", True),
    (1.38, 254, "Status", "status", True),
    (1.50, 277, "ToolChain", "toolchain", True),
    (1.72, 308, "Core.Lang", "core_lang", True),
    (1.84, 331, "Core.Frontend", "core_frontend", True),
    (1.96, 354, "Core.Backend", "core_backend", True),
    (2.08, 377, "Core.Database", "core_database", True),
    (2.20, 400, "Core.Infra", "core_infra", True),
    (2.42, 431, "- Contact", None, False),
    (2.54, 454, "Grid.Mail", "mail", True),
    (2.66, 477, "Grid.Portfolio", "portfolio", True),
    (2.78, 500, "Grid.LinkedIn", "linkedin", True),
    (2.90, 523, "Grid.GitHub", "github", True),
]

LOGO_SLOTS = [
    (LOGO_CODE, 150, 168, 224),
    (LOGO_NEXT, 150, 168, 240),
    (LOGO_NODE, 150, 168, 236),
]


@dataclass
class DotSet:
    xs: np.ndarray
    ys: np.ndarray


def load_and_crop(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGBA")
    w, h = img.size
    cropped = img.crop((int(w * 0.08), 0, int(w * 0.92), int(h * 0.72)))
    cw, ch = cropped.size
    return cropped.crop((0, int(ch * 0.04), cw, ch))


def preprocess(img: Image.Image) -> Image.Image:
    img = ImageEnhance.Contrast(img).enhance(1.3)
    rgb = ImageOps.autocontrast(img.convert("RGB"), cutoff=1)
    rgb = rgb.filter(ImageFilter.UnsharpMask(radius=3, percent=140, threshold=2))
    out = rgb.convert("RGBA")
    out.putalpha(img.split()[3])
    return out


def subject_mask(img: Image.Image) -> np.ndarray:
    arr = np.array(img)
    rgb = arr[..., :3].astype(np.float32)
    alpha = arr[..., 3]
    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    mask = (lum > 28) & (alpha > 16)
    mask = binary_closing(mask, iterations=2)
    mask = binary_fill_holes(mask)
    labeled, n = label(mask)
    if n > 0:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        mask = labeled == sizes.argmax()
    return mask


def floyd_steinberg_serpentine(gray: np.ndarray, draw_mask: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    work = gray.astype(np.float32).copy()
    out = np.zeros((h, w), dtype=bool)
    for y in range(h):
        xs = range(w) if y % 2 == 0 else range(w - 1, -1, -1)
        for x in xs:
            if not draw_mask[y, x]:
                continue
            old = work[y, x]
            new = 0 if old < 128 else 255
            out[y, x] = new == 0
            err = old - new
            if y % 2 == 0:
                neighbors = [(1, 0, 7 / 16), (-1, 1, 3 / 16), (0, 1, 5 / 16), (1, 1, 1 / 16)]
            else:
                neighbors = [(-1, 0, 7 / 16), (1, 1, 3 / 16), (0, 1, 5 / 16), (-1, 1, 1 / 16)]
            for dx, dy, weight in neighbors:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    work[ny, nx] += err * weight
    return out


def center_in_grid(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(xs) == 0:
        return xs, ys
    min_x, max_x = int(xs.min()), int(xs.max())
    min_y, max_y = int(ys.min()), int(ys.max())
    bw, bh = max_x - min_x, max_y - min_y
    ox = (GRID_W - bw) // 2 - min_x
    oy = (GRID_H - bh) // 2 - min_y - 12
    return xs + ox, ys + oy


def portrait_dots(img: Image.Image, dark_mode: bool) -> DotSet:
    resized = img.resize((GRID_W, GRID_H), Image.Resampling.LANCZOS)
    mask = subject_mask(resized)
    gray = np.array(resized.convert("L"))
    bits = floyd_steinberg_serpentine(gray, mask)
    active = bits & mask if dark_mode else bits & mask
    ys, xs = np.where(active)
    xs, ys = center_in_grid(xs.astype(np.float32), ys.astype(np.float32))
    return DotSet(xs, ys)


def _foreground_mask(png_file: Path) -> np.ndarray:
    """Boolean mask of the logo artwork at native resolution.

    Transparency wins when present; otherwise decide by background luminance.
    """
    arr = np.array(Image.open(png_file).convert("RGBA"))
    r, g, b, a = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]

    if a.min() < 240:
        return a > 96

    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    corners = np.concatenate([lum[0, :], lum[-1, :], lum[:, 0], lum[:, -1]])
    if np.median(corners) < 128:
        return lum > 60
    return lum < 235


def logo_mask(cx: int, cy: int, size: int, png_file: Path) -> np.ndarray:
    mask = np.zeros((GRID_H, GRID_W), dtype=bool)
    icon = _foreground_mask(png_file)
    ys, xs = np.where(icon)
    if len(xs) == 0:
        return mask

    # Resize a clean grayscale so transparent RGB never bleeds into the edges.
    flat = Image.fromarray((icon * 255).astype(np.uint8), "L")
    flat = flat.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    cw, ch = flat.size
    scale = (size - 6) / max(cw, ch)
    nw, nh = max(1, round(cw * scale)), max(1, round(ch * scale))
    small = np.array(flat.resize((nw, nh), Image.Resampling.LANCZOS)) >= 110

    x0, y0 = cx - nw // 2, cy - nh // 2
    y1, y2 = max(0, y0), min(GRID_H, y0 + nh)
    x1, x2 = max(0, x0), min(GRID_W, x0 + nw)
    mask[y1:y2, x1:x2] = small[y1 - y0 : y1 - y0 + (y2 - y1), x1 - x0 : x1 - x0 + (x2 - x1)]
    return mask


def spread_sample(dots: DotSet, count: int, seed: int) -> DotSet:
    """Pick `count` points spread evenly over the shape instead of clumped."""
    n = len(dots.xs)
    if n == 0:
        return dots
    rng = np.random.default_rng(seed)
    if n <= count:
        extra = rng.choice(n, count - n, replace=True)
        idx = np.concatenate([np.arange(n), extra])
        jitter = rng.normal(0, 0.6, count)
        return DotSet(dots.xs[idx] + jitter, dots.ys[idx] + rng.normal(0, 0.6, count))

    data = np.column_stack([dots.xs, dots.ys]).astype(np.float64)
    try:
        centroids, _ = kmeans2(data, count, iter=12, minit="points", seed=seed, missing="raise")
        return DotSet(centroids[:, 0].astype(np.float32), centroids[:, 1].astype(np.float32))
    except Exception:
        idx = rng.choice(n, count, replace=False)
        return DotSet(dots.xs[idx], dots.ys[idx])


def logo_dots(png_file: Path, cx: int, cy: int, size: int) -> DotSet:
    mask = logo_mask(cx, cy, size, png_file)
    ys, xs = np.where(mask)
    return DotSet(xs.astype(np.float32), ys.astype(np.float32))


def paths_from_dots(xs: np.ndarray, ys: np.ndarray) -> str:
    rows: dict[int, list[int]] = defaultdict(list)
    for x, y in zip(xs, ys, strict=False):
        rows[int(y)].append(int(x))
    parts: list[str] = []
    for y in sorted(rows):
        xs_row = sorted(set(rows[y]))
        start = prev = xs_row[0]
        for x in xs_row[1:] + [None]:
            if x is not None and x == prev + 1:
                prev = x
                continue
            width = prev - start + 1
            parts.append(f"M{start} {y}h{width}v1h-{width}z" if width > 1 else f"M{start} {y}h1v1h-1z")
            if x is not None:
                start = prev = x
    return "".join(parts)


def hungarian_match(src_x: np.ndarray, src_y: np.ndarray, dst: DotSet, count: int) -> tuple[np.ndarray, np.ndarray]:
    m = len(src_x)
    if m == 0:
        return np.array([]), np.array([])
    dx, dy = dst.xs, dst.ys
    if len(dx) == 0:
        return src_x.copy(), src_y.copy()
    if len(dx) < m:
        rep = int(math.ceil(m / len(dx)))
        dx = np.tile(dx, rep)[:m]
        dy = np.tile(dy, rep)[:m]
    cost = np.sqrt((src_x[:, None] - dx[None, :]) ** 2 + (src_y[:, None] - dy[None, :]) ** 2)
    _, col = linear_sum_assignment(cost)
    return dx[col].astype(np.float32), dy[col].astype(np.float32)


def assign_bands(xs: np.ndarray, ys: np.ndarray, n_bands: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    keys = xs + rng.normal(0, NOISE_SIGMA, len(xs)) + (ys + rng.normal(0, NOISE_SIGMA, len(ys))) * 0.37
    order = np.argsort(keys)
    bands = np.empty(len(xs), dtype=int)
    size = max(1, len(xs) // n_bands)
    for i, idx in enumerate(order):
        bands[idx] = min(i // size, n_bands - 1)
    return bands


def assign_intro_groups(n: int, groups: int) -> np.ndarray:
    rng = np.random.default_rng(7)
    keys = rng.random(n)
    order = np.argsort(keys)
    out = np.empty(n, dtype=int)
    for i, idx in enumerate(order):
        out[idx] = i % groups
    return out


def dot_leaders(label: str, value: str | None) -> str:
    if value is None:
        return "-" * 69
    target = 58 - len(label) - len(value)
    return "." * max(12, target)


def info_panel(theme: str) -> str:
    t = THEMES[theme]
    mail = PROFILE["mail"]
    bar_w = max(220, len(mail) * 9 + 24)
    lines = [
        f'<text x="470" y="106" font-size="13" letter-spacing="2" fill="{t["info_title"]}" filter="url(#txtGlow)">SYSTEM.INFO</text>',
        f'<line x1="566" y1="102" x2="1061" y2="102" stroke="{t["info_line"]}"/>',
        f'<text x="1125" y="106" text-anchor="end" font-size="12" fill="{t["live"]}" font-weight="700">'
        f'<tspan>&#9679;</tspan> LIVE<animate attributeName="opacity" values="1;0.25;1" dur="1.6s" repeatCount="indefinite"/></text>',
        f'<g opacity="0"><animate attributeName="opacity" from="0" to="1" dur="0.5s" begin="0.6s" fill="freeze"/>',
        f'<rect x="470" y="122" width="{bar_w}" height="20" rx="4" fill="{t["email_bg"]}"/>',
        f'<text x="479" y="136" font-size="14" font-weight="700" fill="{t["email_text"]}">{mail}</text>',
        f'<line x1="{470 + bar_w}" y1="130" x2="1125" y2="130" stroke="{t["info_line"]}"/>',
        f"</g>",
    ]
    for begin, y, label, key, is_value in INFO_SPEC:
        dots = dot_leaders(label, PROFILE.get(key) if key else None)
        label_color = t["contact"] if label.startswith("-") else t["label"]
        anim_in = (
            f'<g opacity="0"><animate attributeName="opacity" from="0" to="1" dur="0.4s" begin="{begin:.2f}s" fill="freeze"/>'
        )
        slide = ""
        if is_value:
            slide = (
                f'<animateTransform attributeName="transform" type="translate" values="-8 0;0 0" '
                f'dur="0.4s" begin="{begin:.2f}s" fill="freeze"/>'
            )
        if key:
            value = PROFILE[key]
            text = (
                f'<text x="470" y="{y}" font-size="14" textLength="655" lengthAdjust="spacingAndGlyphs" xml:space="preserve">'
                f'<tspan fill="{label_color}">{label} </tspan>'
                f'<tspan fill="{t["dots"]}">{dots}</tspan>'
                f'<tspan fill="{t["value"]}" font-weight="600"> {value}</tspan></text>'
            )
        else:
            text = (
                f'<text x="470" y="{y}" font-size="14" textLength="655" lengthAdjust="spacingAndGlyphs" xml:space="preserve">'
                f'<tspan fill="{label_color}">{label} </tspan>'
                f'<tspan fill="{t["dots"]}">{dots}</tspan></text>'
            )
        lines.append(f"{anim_in}{slide}{text}</g>")
    lines.append(
        f'<g opacity="0"><animate attributeName="opacity" from="0" to="1" dur="0.5s" begin="3.14s" fill="freeze"/>'
        f'<text x="470" y="577" font-size="14" fill="{t["footer"]}">'
        f"&#9656; More about me &amp; projects below in README &#8595; "
        f'<tspan fill="{t["cursor"]}">&#9608;<animate attributeName="fill-opacity" values="1;0;1" dur="1s" repeatCount="indefinite"/></tspan></text></g>'
    )
    return "\n".join(lines)


def chrome_head(theme: str) -> str:
    t = THEMES[theme]
    title = f"{PROFILE['mail']} - % ./profile.sh --live"
    title_line = "rgba(255,255,255,0.10)" if theme == "dark" else "rgba(15,23,42,0.10)"
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1180" height="610" viewBox="0 0 1180 610" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,'Liberation Mono',monospace" role="img" aria-label="{t['aria']}">
<defs>
{t['accent_grad']}
{t['ascii_grad']}
{t['panel_grad']}
<filter id="glow8" x="-60%" y="-60%" width="220%" height="220%"><feGaussianBlur stdDeviation="8"/></filter>
<filter id="glow3" x="-60%" y="-60%" width="220%" height="220%"><feGaussianBlur stdDeviation="3"/></filter>
<filter id="txtGlow" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="0.9" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
<clipPath id="winClip"><rect x="2" y="2" width="1176" height="606" rx="18"/></clipPath>
</defs>
<rect x="2" y="2" width="1176" height="606" rx="18" fill="{t['outer_bg']}"/>
<g clip-path="url(#winClip)">
<rect x="2" y="2" width="1176" height="606" fill="url(#panelGrad)"/>
<rect x="2" y="2" width="1176" height="46" fill="{t['title_bar']}"/>
<line x1="2" y1="48" x2="1178" y2="48" stroke="{title_line}"/>
<circle cx="30" cy="25.0" r="5.5" fill="#ff5f56"/>
<circle cx="50" cy="25.0" r="5.5" fill="#ffbd2e"/>
<circle cx="70" cy="25.0" r="5.5" fill="#27c93f"/>
<text x="590.0" y="29.0" text-anchor="middle" font-size="12" fill="{t['title_text']}">{title}</text>
<text x="38" y="74" font-size="10" letter-spacing="3" fill="{t['visual_label']}">VISUAL.MAP</text>
<rect x="36" y="84" width="400" height="492" rx="10" fill="none" stroke="{t['frame_stroke']}" stroke-width="2" opacity="0.45" filter="url(#glow3)"/>
<rect x="36" y="84" width="400" height="492" rx="10" fill="{t['frame_fill']}" stroke="{t['frame_stroke_alpha']}"/>"""


def build_visual_layers(portrait: DotSet, theme: str) -> str:
    t = THEMES[theme]
    xs, ys = portrait.xs, portrait.ys
    n = len(xs)
    if n == 0:
        raise RuntimeError("Portrait produced zero dots")

    logos = [
        spread_sample(logo_dots(png, cx, cy, size), NUM_TRAVELLERS, seed=101 + i)
        for i, (png, cx, cy, size) in enumerate(LOGO_SLOTS)
    ]

    rng = np.random.default_rng(99)
    if n >= NUM_TRAVELLERS:
        t_idx = rng.choice(n, NUM_TRAVELLERS, replace=False)
    else:
        t_idx = np.tile(np.arange(n), int(math.ceil(NUM_TRAVELLERS / n)))[:NUM_TRAVELLERS]

    px = xs[t_idx]
    py = ys[t_idx]

    phases_x = [px.copy()]
    phases_y = [py.copy()]
    for logo in logos:
        lx, ly = hungarian_match(phases_x[-1], phases_y[-1], logo, len(phases_x[-1]))
        phases_x.append(lx)
        phases_y.append(ly)
    phases_x.append(px.copy())
    phases_y.append(py.copy())

    intro_ids = assign_intro_groups(n, INTRO_GROUPS)
    bands = assign_bands(xs, ys, NUM_DRIFT_BANDS)

    def band_offset(phase: int, mask: np.ndarray) -> tuple[int, int]:
        bx = float(xs[mask].mean())
        by = float(ys[mask].mean())
        txs, tys = [], []
        for ti, pi in enumerate(t_idx):
            if mask[pi]:
                txs.append(float(phases_x[phase][ti]))
                tys.append(float(phases_y[phase][ti]))
        if not txs:
            return 0, 0
        return int(round(np.mean(txs) - bx)), int(round(np.mean(tys) - by))

    parts = [
        f'<g transform="{VISUAL_TRANSFORM}" fill="{t["portrait"]}" shape-rendering="crispEdges">',
        f'<set attributeName="opacity" to="0" begin="{BEGIN}"/>',
    ]
    for g in range(INTRO_GROUPS):
        mask = intro_ids == g
        if not mask.any():
            continue
        d = paths_from_dots(xs[mask], ys[mask])
        begin = 0.20 + g * 0.03
        parts.append(
            f'<g opacity="0"><animate attributeName="opacity" values="0;1" dur="0.9s" begin="{begin:.2f}s" '
            f'fill="freeze" calcMode="spline" keyTimes="0;1" keySplines=".4 0 .2 1"/><path d="{d}"/></g>'
        )
    parts.append("</g>")

    parts.append(
        f'<g transform="{VISUAL_TRANSFORM}" fill="{t["portrait"]}" shape-rendering="crispEdges" opacity="0">'
        f'<set attributeName="opacity" to="1" begin="{BEGIN}"/>'
    )
    for b in range(NUM_DRIFT_BANDS):
        mask = bands == b
        if not mask.any():
            continue
        d = paths_from_dots(xs[mask], ys[mask])

        offsets = ["0 0", "0 0"]
        for phase in (1, 2, 3):
            ox, oy = band_offset(phase, mask)
            offsets.extend([f"{ox} {oy}", f"{ox} {oy}"])
        offsets.append("0 0")
        translate_vals = ";".join(offsets)
        parts.append(
            f'<g opacity="1"><animate attributeName="opacity" values="{OP_PORTRAIT}" keyTimes="{KEY_TIMES}" '
            f'dur="{DUR}" begin="{BEGIN}" repeatCount="indefinite"/>'
            f'<animateTransform attributeName="transform" type="translate" values="{translate_vals}" '
            f'keyTimes="{KEY_TIMES}" dur="{DUR}" begin="{BEGIN}" repeatCount="indefinite"/>'
            f'<path d="{d}"/></g>'
        )
    parts.append("</g>")

    parts.append(f'<defs><rect id="{t["tv_id"]}" width="2.4" height="1.7" fill="{t["portrait"]}"/></defs>')
    parts.append(f'<g transform="{VISUAL_TRANSFORM}">')
    for i in range(NUM_TRAVELLERS):
        p0 = f"{int(round(px[i]))} {int(round(py[i]))}"
        pos = [
            p0, p0,
            f"{int(round(phases_x[1][i]))} {int(round(phases_y[1][i]))}",
            f"{int(round(phases_x[1][i]))} {int(round(phases_y[1][i]))}",
            f"{int(round(phases_x[2][i]))} {int(round(phases_y[2][i]))}",
            f"{int(round(phases_x[2][i]))} {int(round(phases_y[2][i]))}",
            f"{int(round(phases_x[3][i]))} {int(round(phases_y[3][i]))}",
            f"{int(round(phases_x[3][i]))} {int(round(phases_y[3][i]))}",
            p0,
        ]
        vals = ";".join(pos)
        parts.append(
            f'<use href="#{t["tv_id"]}" opacity="0">'
            f'<animate attributeName="opacity" values="{OP_PARTICLE}" keyTimes="{KEY_TIMES}" dur="{DUR}" begin="{BEGIN}" repeatCount="indefinite"/>'
            f'<animateTransform attributeName="transform" type="translate" values="{vals}" '
            f'keyTimes="{KEY_TIMES}" dur="{DUR}" begin="{BEGIN}" repeatCount="indefinite"/></use>'
        )
    parts.append("</g>")
    return "\n".join(parts)


def generate_svg(theme: str, portrait: DotSet) -> str:
    t = THEMES[theme]
    chunks = [
        chrome_head(theme),
        build_visual_layers(portrait, theme),
        f'<path d="M 50 84 L 36 84 L 36 98" fill="none" stroke="{t["corner"]}" stroke-width="2" opacity="0.8"/>',
        f'<path d="M 422 84 L 436 84 L 436 98" fill="none" stroke="{t["corner"]}" stroke-width="2" opacity="0.8"/>',
        f'<path d="M 50 576 L 36 576 L 36 562" fill="none" stroke="{t["corner"]}" stroke-width="2" opacity="0.8"/>',
        f'<path d="M 422 576 L 436 576 L 436 562" fill="none" stroke="{t["corner"]}" stroke-width="2" opacity="0.8"/>',
        info_panel(theme),
        "</g>",
        '<rect x="3" y="3" width="1174" height="604" rx="17" fill="none" stroke="url(#accent)" stroke-width="3" opacity="0.55" filter="url(#glow8)"/>',
        '<rect x="3" y="3" width="1174" height="604" rx="17" fill="none" stroke="url(#accent)" stroke-width="1.6"/>',
        "</svg>",
    ]
    return "\n".join(chunks)


def main() -> None:
    if not PHOTO.exists():
        raise FileNotFoundError(f"Portrait not found: {PHOTO}")

    img = preprocess(load_and_crop(PHOTO))
    portrait_dark = portrait_dots(img, dark_mode=True)
    portrait_light = portrait_dots(img, dark_mode=False)
    print(f"Dark dots: {len(portrait_dark.xs)} | Light dots: {len(portrait_light.xs)}")

    DATA_DIR.mkdir(exist_ok=True)
    np.save(DATA_DIR / "portrait_dark.npy", np.column_stack([portrait_dark.xs, portrait_dark.ys]))
    np.save(DATA_DIR / "portrait_light.npy", np.column_stack([portrait_light.xs, portrait_light.ys]))

    OUT_DARK.write_text(generate_svg("dark", portrait_dark), encoding="utf-8")
    OUT_LIGHT.write_text(generate_svg("light", portrait_light), encoding="utf-8")
    print(f"Wrote {OUT_DARK} ({OUT_DARK.stat().st_size // 1024} KB)")
    print(f"Wrote {OUT_LIGHT} ({OUT_LIGHT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
