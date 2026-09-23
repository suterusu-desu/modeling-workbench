"""Matched visual review sheets: labeled grids of actual images. No rendering, selection or judgment.

A review compares actual renders at matched views, poses and states. This module only arranges the supplied
images, pins their exact bytes for review evidence and reports anything that would make the comparison
unmatched (missing images, differing image sizes). The owner still looks at every image.
"""
from pathlib import Path
import hashlib
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
