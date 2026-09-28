"""Trial, reopen and retain: three plain calls for changing the native file.

    lane = Trials(service, owner, 'runtime/trials', blender=BLENDER, reference=reference, objects=['Face'],
                  poses={'0.0': {'blink': 0.}, '0.5': {'blink': .5}, '1.0': {'blink': 1.}},
                  declaration='eye-declaration.json')
    lane.trial('lift03', source='checkpoints/working.blend', construction='build_lift.py')
    lane.reopen('lift03')
    lane.retain('lift03', target='checkpoints/lifted-lower-lid.blend', label='lifted lower lid',
                review={'by': 'owner', 'judgment': 'reads as one lid', 'watched': ['videos/lift03.mp4']})

`trial` runs an isolated copy of the live source (it must be open and clean under the owner): the construction script
installs the change between two pose exports (`trial_worker.py`), or a workspace's own trial script runs as it is. The
declaration's standard checks measure the result. `reopen` re-evaluates the saved candidate in a fresh Blender and
compares every pose. `retain` saves the candidate as a new checkpoint in the live Blender, only after a passed reopen,
passing (or explicitly waived) standard checks and a recorded review of the watched motion. Each call uses NativeJob or
RetainCheckpoint with their freshness guards and receipts; every call is appended to `journal.jsonl`. A tag is used
once: a failed step is inspected from its receipts, never replayed.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading

from .checks import _array_refs, run_checks, validate_declaration
from .native_recipes import NativeJob, RetainCheckpoint
from .preservation import file_ref, stale_references
from .store import append_line

HERE = Path(__file__).resolve().parent


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Trials:
    def __init__(self, service, owner, directory, *, blender, reference, objects, poses, declaration=None,
                 workspace=None, python=None, threads=2, timeout=1500, resolution=320):
        if not owner or not objects or not poses:
            raise ValueError('A trial lane needs the native owner, the objects to save and the poses to evaluate')
        first = next(iter(poses))
        if float(first) != 0.:
            raise ValueError('The first pose must be the rest pose, phase 0')
        self.service, self.owner, self.directory = service, owner, Path(directory).resolve()
        self.blender, self.reference, self.objects = str(blender), dict(reference), list(objects)
        self.poses = {str(k): dict(v) for k, v in poses.items()}
        self.declaration = Path(declaration).resolve(strict=True) if declaration else None
        self.workspace = Path(workspace or service.workspace).resolve()
        self.python, self.threads, self.timeout, self.resolution = python, threads, timeout, resolution
        self.directory.mkdir(parents=True, exist_ok=True)

    # -- records -----------------------------------------------------------------------------------------------------
    def _log(self, verb, tag, record, *, reason=None):
        line = {'at': datetime.now(timezone.utc).isoformat(), 'verb': verb, 'tag': tag, 'status': record.get('status'),
                'record': str(self.directory / tag / f'{verb}.json') if reason is None else None}
        if reason is not None:
            line['reason'] = reason
        append_line(self.directory / 'journal.jsonl', json.dumps(line))    # helpers may run lanes side by side

    def _write(self, verb, tag, record):
        path = self.directory / tag / f'{verb}.json'
        path.write_text(json.dumps(record, indent=1, default=str), encoding='utf-8')
        self._log(verb, tag, record)
        return record

    def record(self, verb, tag):
        path = self.directory / tag / f'{verb}.json'
        return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None

    def journal(self):
        path = self.directory / 'journal.jsonl'
        return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.is_file() else []

    # -- native runs ---------------------------------------------------------------------------------------------------
    def _context(self, tag, verb):
        return {'owner': self.owner, 'attempt_key': f'{tag}:{verb}', 'cancelled': threading.Event(),
                'progress': lambda **_: None}

    def _run(self, tag, verb, *, input_ref, live_source, script, job, pinned):
        hashes = {str(script): _sha(script), **{str(Path(p).resolve()): _sha(p) for p in pinned}}
        spec = {'blender': self.blender, 'input': input_ref['path'], 'script': str(script),
                'output_root': str(self.directory / tag / f'{verb}-runs'), 'source_sha256': input_ref['sha256'],
                'dependency_hashes': hashes, 'threads': self.threads, 'timeout_seconds': self.timeout,
                'resolution': self.resolution, 'workspace': str(self.workspace), 'reference': self.reference,
                'objects': self.objects, 'poses': self.poses, **job}
        runner = file_ref(HERE / 'isolated_blender.py')
        refs = [runner, live_source, input_ref, *({'path': p, 'sha256': h} for p, h in hashes.items())]
        item = {'id': verb, 'description': f'{verb} {tag}', 'reads': {str(i): r['sha256'] for i, r in enumerate(refs)},
                'workbench': {'native': True}, 'payload': {'job': spec, 'runner': runner, 'live_source': live_source}}
        return NativeJob(self.service, self.directory / tag, python=self.python)(item, self._context(tag, verb))

    def _check_live(self, tag, source_ref):
        """The live owner must hold `source`, saved and clean; refuse (journaled) otherwise, naming what is live."""
        from .native_recipes import returned_live
        live = returned_live(self.service, self.service.execute('native_inspect_live',
                                                                {'owner': self.owner, 'refresh_scene': False}))
        saved = (live.get('saved_file') or {}).get('sha256')
        problems = [p for p, bad in (
            (f"the live owner is {live.get('owner')!r}", live.get('owner') != self.owner),
            (f"the live file is {live.get('file')}, not {source_ref['path']}",
             Path(live['file']).resolve() != Path(source_ref['path']).resolve()),
            ('the live file has unsaved changes', live.get('dirty') is not False),
            ('the saved live file differs from the source', saved != source_ref['sha256'])) if bad]
        if problems:
            reason = '; '.join(problems)
            self._log('trial', tag, {'status': 'refused'}, reason=reason)
            raise ValueError(f'Trial {tag!r} refused before it started ({reason}); the tag stays unused')

    def trial(self, tag, *, source, construction=None, script=None, job=None, dependencies=()):
        """Run the change on an isolated copy of `source` and measure it with the declaration's standard checks."""
        if (construction is None) == (script is None):
            raise ValueError('Give a construction script (run between pose exports) or a complete trial script')
        if (self.directory / tag).exists():
            raise ValueError(f'Tag {tag!r} was already used; inspect its receipts and choose a new tag')
        source_ref = file_ref(source)
        self._check_live(tag, source_ref)                  # before the tag folder: a refusal leaves the tag unused
        (self.directory / tag).mkdir(parents=True)
        extra = dict(job or {}); pinned = [Path(p) for p in dependencies]
        declared = json.loads(self.declaration.read_text(encoding='utf-8')) if self.declaration else None
        preserved_inputs = None
        if declared is not None and 'preserved' in declared:
            # Pinned before launch (and into the job's dependency hashes), rechecked before measuring and at retention.
            preserved_inputs = {'declaration': file_ref(self.declaration),
                                **_array_refs(validate_declaration(declared), self.declaration.parent)}
            pinned += [Path(ref['path']) for ref in preserved_inputs.values()]
        if construction is not None:
            extra['construction'] = str(Path(construction).resolve(strict=True)); pinned.append(Path(construction))
            pinned.append(HERE / 'native_poses.py'); script = HERE / 'trial_worker.py'
        result = self._run(tag, 'trial', input_ref=source_ref, live_source=source_ref,
                           script=Path(script).resolve(strict=True), job=extra, pinned=pinned)
        record = {'status': result['status'], 'source': source_ref, 'output': result.get('output'), 'native': result}
        if result['status'] == 'completed':
            out = Path(result['output']); record['candidate'] = file_ref(out / 'candidate.blend')
            if self.declaration:
                baseline = out / 'source-evaluated.npz'
                changed = stale_references(list(preserved_inputs.values())) if preserved_inputs else []
                if changed:
                    record['checks'] = {'declaration': preserved_inputs['declaration'], 'status': 'unknown',
                                        'failing': ['preserved_regions'], 'preserved_inputs': preserved_inputs,
                                        'inputs_changed_during_trial': [c['path'] for c in changed]}
                    return self._write('trial', tag, record)
                if preserved_inputs is None:                   # other declarations are read after the run, as before
                    declared = json.loads(self.declaration.read_text(encoding='utf-8'))
                    if 'preserved' in declared:                # a preserved scope added during the run is unbound
                        record['checks'] = {'declaration': file_ref(self.declaration), 'status': 'unknown',
                                            'failing': ['preserved_regions'], 'preserved_inputs': {},
                                            'inputs_changed_during_trial': [str(self.declaration)]}
                        return self._write('trial', tag, record)
                report = run_checks(declared, out / 'evaluated.npz', baseline if baseline.exists() else None,
                                    base=self.declaration.parent)
                record['checks'] = {'declaration': file_ref(self.declaration), 'status': report['status'],
                                    'failing': [c['id'] for c in report['checks'] if c['status'] != 'pass'],
                                    'report': report}
                if preserved_inputs is not None:
                    # The exact files the preserved verdict was measured from; retention re-checks every one.
                    record['checks'].update(preserved_inputs=preserved_inputs, inputs={
                        **preserved_inputs, 'candidate_poses': file_ref(out / 'evaluated.npz'),
                        **({'baseline_poses': file_ref(baseline)} if baseline.exists() else {})})
        return self._write('trial', tag, record)

    def reopen(self, tag, *, script=None, job=None, tolerance=1e-6):
        """Re-evaluate the trial's saved candidate in a fresh Blender and compare every saved pose."""
        trial = self.record('trial', tag)
        if not trial or trial['status'] != 'completed':
            raise ValueError(f'Tag {tag!r} has no completed trial to reopen')
        if self.record('reopen', tag):
            raise ValueError(f'Tag {tag!r} was already reopened; inspect that record')
        out = Path(trial['output']); evaluated = out / 'evaluated.npz'
        script = Path(script).resolve(strict=True) if script else HERE / 'reopen_poses.py'
        extra = {'candidate_evaluated': str(evaluated), 'candidate_sha256': trial['candidate']['sha256'],
                 'tolerance': tolerance, **(job or {})}
        result = self._run(tag, 'reopen', input_ref=trial['candidate'], live_source=trial['source'], script=script,
                           job=extra, pinned=[evaluated, HERE / 'native_poses.py'])
        record = {'status': result['status'], 'native': result, 'output': result.get('output')}
        if result['status'] == 'completed':
            verification = Path(result['output']) / 'verification.json'
            record['verification'] = file_ref(verification)
            record['verdict'] = json.loads(verification.read_text(encoding='utf-8'))
            record['status'] = 'passed' if record['verdict']['status'] == 'passed' else 'failed'
        return self._write('reopen', tag, record)

    def retain(self, tag, *, target, label, review, pose=None, display=None, waive_checks=None, transaction=False):
        """Save the reviewed candidate as a new checkpoint in the live Blender."""
        trial, reopened = self.record('trial', tag), self.record('reopen', tag)
        if not trial or trial['status'] != 'completed' or not reopened or reopened['status'] != 'passed':
            raise ValueError('Retention needs a completed trial and a passed independent reopen')
        if not isinstance(review, dict) or not review.get('judgment') or not review.get('watched'):
            raise ValueError('Retention needs a review: the judgment and the motion watched (videos or frames)')
        checks = trial.get('checks')
        if checks and checks.get('preserved_inputs') is not None:
            # A preserved-region verdict holds only for the exact declaration, selections and poses it measured. A
            # waiver can waive a metric's outcome; it does not cover changed inputs.
            if checks.get('inputs_changed_during_trial'):
                raise ValueError('Inputs of the preserved-region check changed during the trial ('
                                 + '; '.join(checks['inputs_changed_during_trial']) + '); run a new trial')
            stale = stale_references(list(checks['inputs'].values()))
            if stale:
                raise ValueError('Inputs of the preserved-region check changed or are missing since the trial ('
                                 + '; '.join(s['path'] for s in stale) + '); run a new trial')
        if checks and checks['status'] != 'pass' and not waive_checks:
            raise ValueError(f"Standard checks did not pass ({', '.join(checks['failing'])}); change the candidate, "
                             'or record why with waive_checks')
        if self.record('retain', tag):
            raise ValueError(f'Tag {tag!r} was already retained')
        refs = [trial['source'], trial['candidate'], reopened['verification']]
        item = {'id': 'retain', 'description': f'retain {tag}', 'reads': {str(i): r['sha256'] for i, r in enumerate(refs)},
                'workbench': {'profile': 'retention', 'native': True, 'visual_review': 'review'},
                'payload': {'source': trial['source'], 'candidate': trial['candidate'], 'reopen': reopened['verification'],
                            'target': str(Path(target).resolve()), 'label': label,
                            'display': display or {'mode': 'GUIDE_WIRE'}, **({'pose': pose} if pose else {})}}
        verify = lambda verdict, candidate: (verdict.get('status') == 'passed'
                                             and verdict.get('candidate', {}).get('sha256') == candidate['sha256'])
        result = RetainCheckpoint(self.service, self.directory / tag, verify_reopen=verify,
                                  transaction=transaction)(item, self._context(tag, 'retain'))
        record = {'status': result['status'], 'checkpoint': result.get('subject'), 'label': label, 'review': review,
                  'waived_checks': waive_checks, 'native': result, 'user_appearance_accepted': review.get('by') == 'owner'}
        return self._write('retain', tag, record)
