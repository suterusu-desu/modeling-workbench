"""Matched visual review sheets and landmark-aligned overlays of actual images. No rendering, selection or judgment.

A review compares actual renders at matched views, poses and states. This module only arranges the supplied
images, pins their exact bytes for review evidence and reports anything that would make the comparison
unmatched (missing images, differing image sizes). The owner still looks at every image.
"""
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image, ImageDraw


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compose_review_sheet(rows, columns, output, *, cell=(400, 300), title=None, background=(30, 30, 36)):
    """Write a PNG grid: one labeled row per entry, one headed column per state or view.

    rows: [{'label': str, 'images': [path or None, ...]}], one image per column (None = intentionally absent).
    columns: column headings. cell: (width, height) each image is fitted into, aspect preserved.
    Returns the sheet path/hash, every placed source with its hash, missing sources and size mismatches.
    """
    columns = [str(c) for c in columns]
    if not columns or not rows or not isinstance(rows, list):
        raise ValueError('At least one column heading and one row are required')
    width, height = (int(v) for v in cell)
    if not (16 <= width <= 4096 and 16 <= height <= 4096):
        raise ValueError('Cell size must be 16..4096 pixels on each side')
    output = Path(output)
    if output.suffix.lower() != '.png':
        raise ValueError('Review sheets are written as .png')
    if output.exists():
        raise FileExistsError('Refusing to overwrite an existing review sheet: ' + str(output))
    for row in rows:
        if not isinstance(row, dict) or not str(row.get('label', '')).strip() or len(row.get('images', [])) != len(columns):
            raise ValueError('Each row needs a label and exactly one image entry per column')
    gutter, header = 150, 28 + (24 if title else 0)
    sheet = Image.new('RGB', (gutter + width * len(columns), header + height * len(rows)), tuple(background))
    draw = ImageDraw.Draw(sheet)
    if title:
        draw.text((8, 6), str(title), fill=(255, 255, 255))
    for c, heading in enumerate(columns):
        draw.text((gutter + c * width + 6, header - 20), heading, fill=(255, 255, 255))
    sources, missing, sizes = [], [], {}
    for r, row in enumerate(rows):
        top = header + r * height
        draw.text((6, top + height // 2 - 6), str(row['label'])[:24], fill=(255, 255, 255))
        for c, image in enumerate(row['images']):
            left = gutter + c * width
            if image is None:
                continue
            path = Path(image)
            if not path.is_file():
                missing.append({'row': row['label'], 'column': columns[c], 'path': str(path)})
                draw.rectangle([left + 4, top + 4, left + width - 5, top + height - 5], outline=(200, 60, 60))
                draw.text((left + 10, top + 10), 'missing', fill=(200, 60, 60))
                continue
            with Image.open(path) as opened:
                sizes.setdefault(opened.size, []).append({'row': row['label'], 'column': columns[c]})
                picture = opened.convert('RGB')
                picture.thumbnail((width, height))
                sheet.paste(picture, (left + (width - picture.width) // 2, top + (height - picture.height) // 2))
            sources.append({'row': row['label'], 'column': columns[c], 'path': str(path.resolve()), 'sha256': _sha(path)})
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return {'path': str(output.resolve()), 'sha256': _sha(output), 'sources': sources, 'missing': missing,
            'size_groups': [{'size': list(size), 'count': len(entries)} for size, entries in sizes.items()],
            'matched_sizes': len(sizes) <= 1,
            'interpretation': 'Arrangement of supplied actual images only; matching views, poses and appearance judgment remain the reviewer\'s'}


def rows_from_frames(frames, *, views, poses, states, tolerance=1e-6):
    """Build review rows from recorded frames ({'view', 'pose', 'state', 'path'}): one row per view and pose,
    one column per state. Absent combinations become None entries rather than being dropped silently."""
    rows = []
    for view in views:
        for pose in poses:
            images = []
            for state in states:
                match = [f['path'] for f in frames if f.get('view') == view and f.get('state') == state
                         and abs(float(f.get('pose', float('nan'))) - float(pose)) <= tolerance]
                if len(match) > 1:
                    raise ValueError(f'Ambiguous frames for view {view}, pose {pose}, state {state}')
                images.append(match[0] if match else None)
            rows.append({'label': f'{view} {pose}', 'images': images})
    return rows


def similarity_from_landmarks(source, target):
    """The 2D similarity (scale, rotation, translation) taking two source points onto two target points exactly:
    a function mapping (N, 2) points and its parameters. For a render and a drawing, use two landmarks both show at
    the same place, such as the two corners of an eye."""
    s, t = (np.asarray(p, float) for p in (source, target))
    if s.shape != (2, 2) or t.shape != (2, 2) or not np.isfinite([s, t]).all() or np.allclose(s[0], s[1]):
        raise ValueError('Two distinct source points and two target points are required')
    zs, zt = s[:, 0] + 1j * s[:, 1], t[:, 0] + 1j * t[:, 1]
    a = (zt[1] - zt[0]) / (zs[1] - zs[0]); b = zt[0] - a * zs[0]

    def apply(points):
        z = np.asarray(points, float); z = z[:, 0] + 1j * z[:, 1]; w = a * z + b
        return np.c_[w.real, w.imag]
    return apply, {'scale': float(abs(a)), 'rotation_degrees': float(np.degrees(np.angle(a))),
                   'translation': [float(b.real), float(b.imag)]}


def aligned_overlay(render, drawing, render_points, drawing_points, output, *, crop=None, alpha=.25, stroke_alpha=.85,
                    stroke_below=95., lines=()):
    """Overlap a render and a drawing in the drawing's frame, aligned on two landmarks, never side by side.

    The render is resampled into the drawing's pixel frame by the similarity taking `render_points` onto
    `drawing_points` (pixel x, y of the same two landmarks in each image), so the drawing keeps its own pixels. The
    drawing is laid over the render at `alpha`, its dark strokes (luminance below `stroke_below`) rising to
    `stroke_alpha`. `crop` (x0, y0, x1, y1 in drawing pixels) limits the output. `lines` are polylines drawn on top:
    {'points': [[x, y], ...], 'frame': 'drawing' | 'render', 'color': [r, g, b], 'width': 2}. Returns the output
    path and hash, the similarity and the sources' hashes.
    """
    output = Path(output)
    if output.suffix.lower() != '.png':
        raise ValueError('Overlays are written as .png')
    if output.exists():
        raise FileExistsError('Refusing to overwrite an existing overlay: ' + str(output))
    if not (0 <= alpha <= 1 and 0 <= stroke_alpha <= 1):
        raise ValueError('Alphas must lie in [0, 1]')
    to_drawing, parameters = similarity_from_landmarks(render_points, drawing_points)
    to_render, _ = similarity_from_landmarks(drawing_points, render_points)
    with Image.open(drawing) as d:
        art = np.asarray(d.convert('RGB'), float)
    with Image.open(render) as r:
        shot = r.convert('RGB')
    height, width = art.shape[:2]
    # Inverse mapping: every drawing pixel looks up its render pixel (an affine map for PIL).
    origin = to_render(np.array([[0., 0.]]))[0]; ex = to_render(np.array([[1., 0.]]))[0] - origin
    ey = to_render(np.array([[0., 1.]]))[0] - origin
    moved = np.asarray(shot.transform((width, height), Image.Transform.AFFINE,
                                      (ex[0], ey[0], origin[0], ex[1], ey[1], origin[1]), Image.Resampling.BICUBIC), float)
    darkness = np.clip((stroke_below - art.mean(axis=2)) / max(stroke_below, 1e-9), 0, 1)
    weight = (alpha + (stroke_alpha - alpha) * darkness)[..., None]
    picture = Image.fromarray(np.clip(moved * (1 - weight) + art * weight, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(picture)
    for line in lines:
        points = np.asarray(line['points'], float)
        if line.get('frame', 'drawing') == 'render':
            points = to_drawing(points)
        draw.line([tuple(p) for p in points], fill=tuple(int(c) for c in line.get('color', (255, 0, 0))),
                  width=int(line.get('width', 2)))
    if crop is not None:
        picture = picture.crop(tuple(int(v) for v in crop))
    output.parent.mkdir(parents=True, exist_ok=True)
    picture.save(output)
    return {'path': str(output.resolve()), 'sha256': _sha(output), 'similarity_render_to_drawing': parameters,
            'sources': {'render': {'path': str(Path(render).resolve()), 'sha256': _sha(render)},
                        'drawing': {'path': str(Path(drawing).resolve()), 'sha256': _sha(drawing)}},
            'interpretation': 'Two-landmark alignment only: the landmarks coincide exactly, everything else shows how '
                              'the render and the drawing differ; the comparison and its judgment are the reviewer\'s'}
