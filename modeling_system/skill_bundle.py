"""Ship agent guidance with the package, without external skill dependencies."""
from pathlib import Path
import shutil


def verify_dependencies(root=None):
    """Verify complete workbench guidance, without an external skill dependency."""
    root = Path(root or Path(__file__).parent)
    files = ['plugin/skills/modeling-workbench/SKILL.md',
             'plugin/skills/modeling-workbench/references/method.md',
             'plugin/skills/modeling-workbench/references/operating-session.md']
    problems = [name for name in files if not (root/name).is_file() or not (root/name).stat().st_size]
    return {'status': 'verified' if not problems else 'mismatch', 'files': files, 'problems': problems,
            'external_dependencies': []}


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
