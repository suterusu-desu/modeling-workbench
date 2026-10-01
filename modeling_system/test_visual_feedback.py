"""Focused durable-workflow and false-authority risks, using anonymous images."""
import base64
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from PIL import Image
from .service import ModelingService
from .visual_feedback_server import make_server


class VisualFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.service = ModelingService(self.root)
        created = self.call('create', title='Surface appearance', idempotency_key='synthetic-board')
        self.board, self.revision = created['board'], created['revision']
        self.image = self.root/'inspection.png'
        Image.new('RGB', (120, 90), '#c4ab81').save(self.image)
        self.inspection = self.add_image('Inspection', 'source-v1')
        self.baseline = self.add_image('Approved rest', 'baseline-v1')
        self.trial = self.add_image('Isolated trial', 'trial-v1')
        self.target = self.mutate('target', image=self.inspection, label='Surface band',
            wording='  Smooth the skin; preserve its outline.\nDo not change the curve.  ',
            date='2026-01-01', source='Synthetic user message', region=[.2,.3,.4,.2])['added']['targets'][0]
        self.plan = self.mutate('plan', target=self.target, interpretation='Reduce the outline bulge',
            preserved_features=['Rest likeness'], method='Move the contour inward',
            expected_appearance='Smaller silhouette')['added']['plans'][0]
        self.comparison = self.compare()

    def call(self, name, **args):
        result = self.service.execute('visual_feedback_'+name, args)
        self.assertNotIn(result['status'], ('failed', 'conflicting'), result)
        return result

    def mutate(self, name, **args):
        result = self.call(name, board=self.board, expected_revision=self.revision, **args)
        self.revision = result['revision']
        return result

    def fact(self, wording='Yes, that is the target.'):
        return dict(actor='user', wording=wording, date='2026-01-02', source='Synthetic owner message')

    def add_image(self, label, version, **metadata):
        return self.mutate('image', image_path=str(self.image), metadata=dict(label=label,
            source='Synthetic saved capture', source_version=version, captured_at='2026-01-01T12:00:00Z',
            camera='front-v1', display_state='bare-surface', pose='closed', **metadata))['added']['images'][0]

    def compare(self, **changes):
        args = dict(target=self.target, plan=self.plan, baseline=self.baseline, trial=self.trial,
            baseline_caption='Approved appearance before the trial', trial_caption='Isolated contour change; surface unresolved',
            changed='Outline moved inward', unchanged='Uneven skin remains', baseline_region=[.2,.3,.4,.2], trial_region=[.2,.3,.4,.2],
            baseline_approval=self.fact('Use this baseline.'))
        args.update(changes)
        return self.mutate('compare', **args)['added']['comparisons'][0]

    def feedback_args(self, comparison=None):
        r = self.service._visual_feedback().read(self.board)
        row = next(x for x in r['comparisons'] if x['record'] == (comparison or self.comparison))
        return dict(comparison=row['record'], **{k:row[k] for k in
            ('target','plan','inspection','trial','source_version','result_version')},
            wording='The curve was never the problem. Smooth the surface and preserve the curve.',
            date='2026-01-02', source='Synthetic exact correction')

    def test_correct_agree_reopen_keeps_target_history_and_local_rejected_reference(self):
        ref = self.mutate('reference', image=self.inspection, role='Relaxed surface appearance', overall_status='rejected',
            rejection_reason='Failed motion contact', supported='Central surface appearance; visibly relaxed',
            excluded='Contact boundary; failed motion', unknown='Depth; no registration',
            qualification='Synthetic local appearance inspection, no geometry transfer')['added']['reference_roles'][0]
        self.mutate('agreement', target=self.target, plan=self.plan, fact=self.fact())
        result = self.mutate('submit', **self.feedback_args(), correction=dict(
            interpretation='Smooth relaxed skin, keeping the macroscopic outline',
            preserved_features=['Macro curve', 'Corners', 'Rest likeness'], method='Diagnose roughness and correct locally',
            expected_appearance='Skin without bumps or tension; same outline', references=[ref]))
        corrected = result['added']['plans'][0]
        self.mutate('agreement', target=self.target, plan=corrected, fact=self.fact('Yes: skin smoothness, with the same outline.'))
        self.service = ModelingService(self.root)  # New process-equivalent service; no in-memory flow state.
        view = self.call('read', board=self.board)
        active = next(iter(view['current'].values()))
        self.assertEqual(active['target'], self.target)
        self.assertEqual(active['plan'], corrected)
        self.assertIsNone(active['comparison'])
        self.assertIsNotNone(active['agreement'])
        self.assertEqual(len(view['plans']), 2)
        self.assertEqual(view['plans'][1]['supersedes'], self.plan)
        self.assertEqual(view['targets'][0]['wording'], '  Smooth the skin; preserve its outline.\nDo not change the curve.  ')
        self.assertIn('historical only', view['feedback'][0]['applicability'])
        self.assertEqual(view['reference_roles'][0]['overall_status'], 'rejected')
        self.assertFalse(view['reference_roles'][0]['geometry_adopted'])
        self.assertFalse(view['state_facts'])
        output = self.root/'target.md'
        self.call('export', board=self.board, output=str(output))
        self.assertIn('Smooth relaxed skin', output.read_text())
        self.assertIn(self.plan, output.read_text())
        self.assertEqual(self.service.store.resolve_blob(view['images'][0]['asset']).read_bytes(), self.image.read_bytes())

    def test_stale_feedback_exact_result_binding_and_concurrent_revision(self):
        old_revision = self.revision
        newer = self.compare()
        result = self.service.execute('visual_feedback_submit', dict(board=self.board, expected_revision=self.revision, **self.feedback_args()))
        self.assertEqual(result['status'], 'conflicting')
        result = self.service.execute('visual_feedback_submit', dict(board=self.board, expected_revision=old_revision, **self.feedback_args(newer)))
        self.assertEqual(result['status'], 'conflicting')
        args = self.feedback_args(newer); args['result_version']='different-result'
        result = self.service.execute('visual_feedback_submit', dict(board=self.board, expected_revision=self.revision, **args))
        self.assertEqual(result['status'], 'conflicting')
        self.assertFalse(self.call('read', board=self.board)['feedback'])
        self.mutate('submit', **self.feedback_args(), historical=True)
        self.assertTrue(self.call('read', board=self.board)['feedback'][0]['historical'])

    def test_unknown_matching_and_user_facts_never_imply_result_approval(self):
        unknown = self.add_image('Missing camera', 'trial-v2')
        # Create a distinct unknown descriptor record without changing existing evidence.
        image = self.service.store.get(unknown, 'visual_images')
        metadata = dict(image['metadata'], camera=None)
        unknown = self.mutate('image', image_path=str(self.image), metadata=metadata)['added']['images'][0]
        c = self.compare(trial=unknown)
        different = self.compare(trial_region=[.1,.3,.4,.2])
        view = self.call('read', board=self.board)
        self.assertEqual(next(r for r in view['comparisons'] if r['record']==c)['matching']['status'], 'unknown')
        self.assertEqual(next(r for r in view['comparisons'] if r['record']==different)['matching']['status'], 'unmatched')
        self.assertEqual(view['comparisons'][0]['matching']['status'], 'matched declared inputs')
        self.mutate('state', comparison=self.comparison, facet='technical_verification', value='passed', evidence='Synthetic contract passed; no appearance claim')
        self.mutate('agreement', target=self.target, plan=self.plan, fact=self.fact())
        result = self.service.execute('visual_feedback_state', dict(board=self.board, expected_revision=self.revision,
            comparison=self.comparison, facet='owner_acceptance', value='accepted', evidence='Generic review flag', fact={'approved':True}))
        self.assertEqual(result['status'], 'failed')
        self.assertEqual([r['facet'] for r in self.call('read', board=self.board)['state_facts']], ['technical_verification'])
        self.assertFalse(self.call('read', board=self.board)['mandatory_approval_gate'])

    def test_matching_keeps_inspection_region_bound_to_its_view(self):
        # A mutually matched pair from another camera cannot inherit the marked
        # inspection coordinates. This is a concrete false-matching risk.
        metadata = dict(label='Other view', source='Synthetic saved capture', source_version='other-v1',
            captured_at='2026-01-01', camera='profile-v1', display_state='bare-surface', pose='closed')
        baseline = self.mutate('image', image_path=str(self.image), metadata=metadata)['added']['images'][0]
        trial = self.mutate('image', image_path=str(self.image), metadata=dict(metadata, source_version='other-v2'))['added']['images'][0]
        comparison = self.compare(baseline=baseline, trial=trial)
        row = next(r for r in self.call('read', board=self.board)['comparisons'] if r['record'] == comparison)
        self.assertEqual(row['matching']['status'], 'unmatched')
        self.assertIn('inspection/baseline camera', row['matching']['mismatched'])
        self.assertIn('inspection/trial camera', row['matching']['mismatched'])

    def test_archive_reopen_restore_preserves_bound_evidence_and_history(self):
        original = self.call('read', board=self.board)
        self.mutate('presentation', collection='images', record=self.trial, archived=True, reason='Incidental image in working view')
        self.mutate('presentation', collection='board', record=self.board, archived=True, reason='Completed demonstration')
        self.service = ModelingService(self.root)
        archived = self.call('read', board=self.board)
        self.assertEqual(archived['images'], original['images'])
        self.assertEqual(archived['comparisons'], original['comparisons'])
        self.assertEqual(archived['current'], original['current'])
        self.assertIn(self.trial, archived['presentation']['archived_images'])
        self.assertTrue(self.call('read')['boards'][0]['archived'])
        row = next(r for r in archived['images'] if r['record'] == self.trial)
        self.assertEqual(self.service.store.resolve_blob(row['asset']).read_bytes(), self.image.read_bytes())
        self.mutate('presentation', collection='images', record=self.trial, archived=False, reason='Restore useful reference')
        self.mutate('presentation', collection='board', record=self.board, archived=False, reason='Reopen work')
        restored = self.call('read', board=self.board)
        self.assertEqual(restored['presentation'], {'board_archived':False, 'archived_images':[]})
        self.assertEqual(len(restored['presentation_events']), 4)

    def test_local_ui_transport_reopens_same_store_and_refuses_native_route(self):
        server = make_server(self.service, 0)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = 'http://127.0.0.1:'+str(server.server_port)
        with urlopen(base+'/') as response:
            self.assertIn(b'Baseline &amp;', response.read().replace(b'Baseline &', b'Baseline &amp;'))
        with urlopen(base+'/app.js') as response:
            self.assertIn(b'onpointermove', response.read())
        with urlopen(base+'/api/board?board='+self.board) as response:
            self.assertEqual(json.load(response)['targets'][0]['record'], self.target)
        body = dict(board=self.board, expected_revision=self.revision, filename='upload.png',
                    data=base64.b64encode(self.image.read_bytes()).decode(), metadata=dict(
                    label='UI upload', source='Synthetic upload', source_version='v3', captured_at='2026-01-03'))
        with urlopen(Request(base+'/api/upload', data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})) as response:
            self.assertEqual(json.load(response)['status'], 'completed')
        for body, origin in [(dict(operation='native_set_controls', arguments={}),None),
                             (dict(operation='visual_feedback_create', arguments={}), 'http://example.invalid')]:
            headers={'Content-Type':'application/json'}
            if origin:headers['Origin']=origin
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(base+'/api/operation', data=json.dumps(body).encode(), headers=headers))
            self.assertEqual(error.exception.code,403 if origin else 400)
