"""Run one reversible Blender construction experiment with isolated user files."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', type=Path, help='Qualified job JSON; see the packaged isolated-Blender reference')
    args = parser.parse_args()
    job = json.loads(args.job.read_text(encoding='utf-8-sig'))
    base = args.job.resolve().parent
    script = (base/job['script']).resolve(strict=True)
    blender = (base/job['blender']).resolve(strict=True)
    source = (base/job['input']).resolve(strict=True) if job.get('input') else None
    out_root = (base/job['output_root']).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S-%fZ')
    output = out_root / stamp
    output.mkdir()
    job.update(script=str(script), output=str(output), started_utc=stamp)
    before = sha256(source) if source else None
    if source:
        if source.suffix.lower() != '.blend':
            raise ValueError('input must be a .blend; import other formats in your script')
        copied = output / 'input.blend'
        shutil.copy2(source, copied)
        job['copied_input'] = str(copied)
    config_path = output / 'job.json'
    config_path.write_text(json.dumps(job, indent=2), encoding='utf-8')
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    env['PYTHONNOUSERSITE'] = '1'
    profile = output / 'profile'
    for key, folder in [('RESOURCES', ''), ('CONFIG', 'config'), ('SCRIPTS', 'scripts'),
                        ('EXTENSIONS', 'extensions'), ('DATAFILES', 'datafiles')]:
        path = profile / folder
        path.mkdir(parents=True, exist_ok=True)
        env['BLENDER_USER_' + key] = str(path)
    command = [str(blender), '--background', '--factory-startup', '--disable-autoexec',
               '--threads', str(job.get('threads', 2)), '--python-exit-code', '7',
               '--python', str(Path(__file__).with_name('blender_worker.py')), '--', str(config_path)]
    start = time.monotonic()
    receipt = {'source': str(source) if source else None, 'source_sha256_before': before,
               'script_sha256': sha256(script), 'output': str(output), 'command': command,
               'profile': str(profile), 'status': 'running', 'started_utc': stamp}
    receipt_path = output / 'receipt.json'
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'running', 'output': str(output)}), flush=True)
    try:
        with (output / 'blender.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=output, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            receipt['pid'] = process.pid
            receipt_path.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
            try:
                returncode = process.wait(timeout=job.get('timeout_seconds', 240))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                receipt['error'] = 'Blender exceeded the bounded job timeout'
                returncode = -1
        after = sha256(source) if source else None
        receipt.update(returncode=returncode, source_sha256_after=after,
                       source_unchanged=(before == after), elapsed_seconds=round(time.monotonic() - start, 2))
        passed = (returncode == 0 and before == after and
                  (output / 'inventory.json').is_file() and (output / 'candidate.blend').is_file())
        receipt['status'] = 'completed' if passed else 'failed'
    except Exception as exc:
        receipt.update(status='failed', error=str(exc), elapsed_seconds=round(time.monotonic() - start, 2))
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    (out_root / 'latest.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt), flush=True)
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    sys.exit(main())
