"""Ship agent guidance with the package, including pinned upstream dependencies."""
import hashlib
import json
from pathlib import Path
import shutil


def verify_dependencies(root=None):
    root = Path(root or Path(__file__).parent)
    lock = json.loads((root/'skill-dependencies.json').read_text(encoding='utf-8'))
    checked, problems = [], []
    for name, dependency in lock['skills'].items():
        for relative, expected in dependency['files'].items():
            path = (root/relative).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError('Skill dependency escapes package')
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected['sha256']:
                problems.append(relative)
            checked.append(relative)
    return {'status': 'verified' if not problems else 'mismatch', 'files': checked, 'problems': problems}


def install_skills(destination):
    """Copy complete skills; refuse conflicting files instead of erasing local edits."""
    source = Path(__file__).parent/'plugin/skills'
    destination = Path(destination).resolve()
    verification = verify_dependencies()
    if verification['problems']:
        raise ValueError('Bundled skill dependency failed integrity verification')
    files = [p for p in source.rglob('*') if p.is_file()]
    for path in files:
        target = destination/path.relative_to(source)
        if target.exists() and (not target.is_file() or target.read_bytes() != path.read_bytes()):
            raise FileExistsError('Existing skill differs; preserve it and select a new destination')
        if any(parent.exists() and not parent.is_dir() for parent in target.parents):
            raise FileExistsError('Skill destination contains a conflicting file')
    for path in files:
        target = destination/path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    return [p.name for p in sorted(source.iterdir()) if (p/'SKILL.md').is_file()]
