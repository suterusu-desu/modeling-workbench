"""Regressions for imported evidence, displacement and revision-bound recovery."""
from copy import deepcopy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import imageio_ffmpeg
from . import test_workflow as workflow_fixtures
from . import test_execution_receipts as execution_fixtures
from .test_execution_receipts import ref
from .service import ModelingService
from .store import canonical, digest, native_path, atomic_write
from .episodes import coverage
from .interventions import assess, compare
from .recovery import select, inspect


class AuditContracts(unittest.TestCase):
    setUp=workflow_fixtures.WorkflowTests.setUp
    tearDown=workflow_fixtures.WorkflowTests.tearDown
    request=workflow_fixtures.WorkflowTests.request
    completed=workflow_fixtures.WorkflowTests.completed
    reviewed=workflow_fixtures.WorkflowTests.reviewed

    def test_imported_scope_count_is_not_object_inclusion(self):
        state=self.s.store.get(self.state,'state')
        state['coverage']={'included':'all captured mesh surfaces in this pose','objects':1}
        result=coverage(state)
        self.assertEqual(result['declaration']['schema'],'scope_count')
        self.assertEqual(result['declaration']['objects'],[])
        self.assertIn('unknown',result['declaration']['object_membership'])
        state['coverage']=[dict(name='Hidden',included=False,reason='uncaptured hidden dependency')]
        self.assertEqual(coverage(state)['excluded'][0]['reason'],'uncaptured hidden dependency')
        state['coverage']={'included':True,'objects':'everything'}
        with self.assertRaisesRegex(ValueError,'Unsupported coverage'):coverage(state)

    def test_close_episode_reads_legacy_scope_count_without_exception(self):
        state=self.s.store.get(self.state,'state');state['coverage']={'included':'captured meshes','objects':1}
        key=self.s.store.put('state',state)
        q=self.s.open_question('Coverage close',key,'surface','Recover imported context')['question']
        e=self.s.open_episode(q,'fixture','offline only',{},'coverage-close')
        closed=self.s.reconcile_episode(e['episode'],e['revision'],character={'status':'unresolved','reason':'No appearance review'},
            method={'status':'supported','reason':'Legacy scope can be read','integration':{'disposition':'not_generalizable','reason':'Fixture evidence only'}},
            evidence=[{'kind':'record','id':key,'role':'imported scene'}],applicability='synthetic imported scope',close=True)
        self.assertEqual(closed['status'],'closed')
        self.assertEqual(self.s.inspect_situation()['coverage']['declaration']['schema'],'scope_count')
        self.assertEqual(self.s.decision_workspace(e['episode'])['baseline']['coverage']['declaration']['schema'],'scope_count')

    def test_landmark_error_precedes_image_access_and_preserves_job(self):
        job=self.completed();before=self.s.ledger.read(job['handle'])
        with patch('modeling_system.references.Image.open',side_effect=AssertionError('should validate first')):
            with self.assertRaisesRegex(ValueError,'landmarks requires source and output'):
                self.s.review_reference(job['handle'],0,'useful','eye',0,'framed','disputed','matched','matched',[],{'output':[[1,2]]})
        self.assertEqual(before,self.s.ledger.read(job['handle']))

    def test_ancestor_rejection_invalidates_completed_descendants(self):
        first=self.reviewed()
        second=self.completed('image',first,'derived')
        review=self.s.review_reference(second['handle'],0,'useful','eye',0,'framed','matched','matched','matched',[])['review']
        third=self.completed('video',review,'motion')
        parent=self.s.store.get(first,'reference_review')['job']
        rejected=self.s.review_reference(parent,0,'rejected','eye',0,'Faulty eye position copied from mesh','failed','matched','matched',[],
                                          {'source':[[0,0]],'output':[[0,0]]})
        self.assertEqual(rejected['measured']['maximum_pixel_drift'],0)
        self.assertEqual(self.s.inspect_workflow(third['handle'])['source_validity']['status'],'stale')
        self.assertEqual(self.s.ledger.read(third['handle'])['status'],'completed')
        with self.assertRaises(ValueError):self.s.references.require_usable_review(review)

    def intervention(self):
        reviewed=self.reviewed()
        state=self.s.store.get(self.state,'state');state['objects'][0]['evaluation']='evaluated world surface'
        baseline=self.s.store.put('state',state)
        a=deepcopy(self.a);a['co'][:,2]=.01
        p=self.root/'predicted.npz';np.savez_compressed(p,**a)
        predicted=deepcopy(state);predicted['objects'][0]['asset']=self.s.store.blob(p)
        target=self.s.store.put('state',predicted)
        guide=self.root/'guide.npz';np.savez_compressed(guide,**a)
        qualification=self.s.store.put('guide_qualification',dict(review=reviewed,native_state=baseline,pose='neutral',
            registered=self.s.store.blob(guide),support=[dict(selection=None)]))
        evidence=self.s.store.put('construction_evidence',dict(reason='Synthetic solver evidence'))
        case=dict(mode='guide_fit',state=baseline,predicted_state=target,pose=state['controls'],guide_pose='neutral',mechanism='normal displacement',
            construction_evidence=[evidence],realization_tolerance=.0001,layers=[dict(object='Face',semantic_component='moving surface',support=[
                dict(kind='direct',indices=[0,1],qualification=qualification,tolerance=.001,basis='synthetic measured tolerance'),
                dict(kind='interpolated',indices=[2],neighbors=[0,1],weights=[.5,.5],tolerance=.001,basis='connected neighbor displacement')])])
        return case,baseline,target

    def test_actual_displacement_denominator_and_realized_violation(self):
        case,base,target=self.intervention()
        result=assess(self.s,case,base,{'allowed_objects':['Face']})
        self.assertTrue(result['numerically_supported']);self.assertEqual(result['layers'][0]['direct_fraction'],2/3)
        self.assertEqual(result['layers'][0]['support']['2']['kind'],'interpolated')
        self.assertTrue(compare(self.s,result['record'],target)['agrees'])
        a=self.s.wb.arrays(target,'Face');a['co'][2,2]+=.02
        p=self.root/'actual.npz';np.savez_compressed(p,**a)
        state=self.s.store.get(target,'state');state['objects'][0]['asset']=self.s.store.blob(p)
        actual=self.s.store.put('state',state)
        observed=compare(self.s,result['record'],actual)
        self.assertFalse(observed['agrees']);self.assertFalse(observed['realized_numerically_supported'])
        self.assertEqual(observed['layers'][0]['exceeds_tolerance'],[2])

    def test_unsupported_prediction_and_explicit_construction_repair(self):
        case,base,target=self.intervention();case['layers'][0]['support'].pop()
        result=assess(self.s,case,base,{'allowed_objects':['Face']})
        self.assertEqual(result['disposition'],'unsupported_prediction');self.assertEqual(result['layers'][0]['unsupported'],[2])
        case['mode']='construction_repair'
        with self.assertRaisesRegex(ValueError,'Construction repair needs'):assess(self.s,case,base,{'allowed_objects':['Face']})
        backup=self.root/'baseline.bin';backup.write_bytes(b'baseline')
        case['repair']=dict(reason='Existing correspondence is defective',recovery_checkpoint=ref(backup),
            revised_constraints='New supported correspondence pending',validation_plan='Check native geometry and affected layers')
        result=assess(self.s,case,base,{'allowed_objects':['Face']})
        self.assertEqual(result['disposition'],'construction_repair');self.assertFalse(result['numerically_supported'])

    def test_realized_guide_violation_below_prediction_tolerance_is_reported(self):
        case,base,target=self.intervention();case['realization_tolerance']=.1
        result=assess(self.s,case,base,{'allowed_objects':['Face']})
        a=self.s.wb.arrays(target,'Face');a['co'][:,2]+=.002
        p=self.root/'small-error.npz';np.savez_compressed(p,**a)
        state=self.s.store.get(target,'state');state['objects'][0]['asset']=self.s.store.blob(p)
        observed=compare(self.s,result['record'],self.s.store.put('state',state))
        self.assertEqual(observed['layers'][0]['exceeds_tolerance'],[])
        self.assertFalse(observed['realized_numerically_supported']);self.assertFalse(observed['agrees'])

    def test_dependent_attachment_uses_supported_driver_without_inventing_guide(self):
        case,base,target=self.intervention()
        states=[]
        for key in (base,target):
            state=self.s.store.get(key,'state');a=self.s.wb.arrays(key,'Face');a['co'][:,2]+=.05
            p=self.root/(key+'.npz');np.savez_compressed(p,**a)
            state['objects'].append(dict(name='Dependent',type='MESH',asset=self.s.store.blob(p),evaluation='evaluated world'))
            states.append(self.s.store.put('state',state))
        base,target=states;case.update(state=base,predicted_state=target)
        group=case['layers'][0]['support'][0];q=self.s.store.get(group['qualification'],'guide_qualification');q['native_state']=base
        group['qualification']=self.s.store.put('guide_qualification',q)
        case['layers'].append(dict(object='Dependent',semantic_component='attached trim',support=[dict(kind='attachment',indices=[0,1,2],
            driver_object='Face',driver_indices=[0,1,2],weights=[1,0,0],rest_offsets=[[0,0,.05],[2,0,.05],[0,2,.05]],
            tolerance=.00001,basis='independently reproduced baseline support frame')]))
        result=assess(self.s,case,base,{'allowed_objects':['Face','Dependent']})
        self.assertTrue(result['numerically_supported']);self.assertEqual(result['layers'][1]['direct_changed'],0)
        self.assertEqual(result['layers'][1]['support']['0']['kind'],'attachment')

    def context_case(self):
        case,base,target=self.intervention()
        contexts=[('Guide','registered target','registered guide'),
                  ('Preserved','base source geometry; outside active dependency graph, modifiers and pose not evaluated','source/reference'),
                  ('Diagnostic','evaluated active dependency graph','derived diagnostic/display')]
        states=[]
        for key in (base,target):
            state=self.s.store.get(key,'state')
            for name,evaluation,role in contexts:
                state['objects'].append(dict(name=name,type='MESH',evaluation=evaluation,geometry_role=role,
                    asset=self.s.store.get(base,'state')['objects'][0]['asset']))
            states.append(self.s.store.put('state',state))
        new_base,new_target=states
        self.assertEqual(self.s.store.get(new_base)['source_state_id'],self.s.store.get(base)['source_state_id'])
        case.update(state=new_base,predicted_state=new_target)
        group=case['layers'][0]['support'][0];q=self.s.store.get(group['qualification']);q['native_state']=new_base
        group['qualification']=self.s.store.put('guide_qualification',q)
        for name,_,role in contexts:
            case['layers'].append(dict(object=name,mode='unchanged_context',semantic_component=role,reason='Captured context, preserved through this trial'))
        return case,new_base,new_target

    def test_full_native_inventory_preserves_unchanged_context_without_relabeling(self):
        case,base,target=self.context_case()
        report=assess(self.s,case,base,{'allowed_objects':['Face']})
        self.assertTrue(report['numerically_supported'])
        self.assertEqual(report['case']['state'],base)
        self.assertEqual(report['layers'][1]['context']['evaluation'],'registered target')
        self.assertIn('not evaluated',report['layers'][2]['context']['evaluation'])
        actual=compare(self.s,report['record'],target)
        self.assertTrue(actual['agrees']);self.assertEqual(len(actual['layers']),4)
        case['layers'].pop()
        with self.assertRaisesRegex(ValueError,'every recorded surface'):assess(self.s,case,base,{'allowed_objects':['Face']})

    def test_context_geometry_attributes_and_interpretation_cannot_change(self):
        case,base,target=self.context_case();report=assess(self.s,case,base,{'allowed_objects':['Face']})
        for change in ('position','attribute','evaluation'):
            state=self.s.store.get(target,'state');entry=state['objects'][1]
            if change=='evaluation':entry['evaluation']='evaluated world'
            else:
                a=self.s.wb.arrays(target,'Guide')
                if change=='position':a['co'][0,2]+=.01
                else:a['POINT__meaning']=np.ones(len(a['co']))
                p=self.root/(change+'.npz');np.savez_compressed(p,**a);entry['asset']=self.s.store.blob(p)
            changed=self.s.store.put('state',state)
            with self.subTest(change=change):
                with self.assertRaisesRegex(ValueError,'unchanged_context changed'):
                    assess(self.s,dict(case,predicted_state=changed),base,{'allowed_objects':['Face']})
                with self.assertRaisesRegex(ValueError,'unchanged_context changed'):compare(self.s,report['record'],changed)

    def test_not_evaluated_is_never_positive_and_context_cannot_supply_support(self):
        case,base,target=self.context_case()
        case['layers'][2].pop('mode')
        with self.assertRaisesRegex(ValueError,'positively declared evaluated'):
            assess(self.s,case,base,{'allowed_objects':['Face']})
        case['layers'][2]['mode']='unchanged_context';case['layers'][2]['support']=[{'kind':'direct'}]
        with self.assertRaisesRegex(ValueError,'cannot declare displacement support'):
            assess(self.s,case,base,{'allowed_objects':['Face']})

    def diagnostic_case(self):
        case,base,target=self.context_case()
        row=case['layers'][-1];row.update(mode='recomputed_diagnostic',reason='Recomputed section of changed surface',derivation_evidence=case['construction_evidence'])
        arrays=dict(co=np.array([[0.,0,0],[1,0,0],[1,1,0],[0,1,0]]),tri=np.array([[0,1,2],[0,2,3]],dtype=np.int32))
        p=self.root/'regenerated.npz';np.savez_compressed(p,**arrays)
        state=self.s.store.get(target,'state');state['objects'][-1]['asset']=self.s.store.blob(p)
        target=self.s.store.put('state',state);case['predicted_state']=target
        return case,base,target

    def test_recomputed_diagnostic_changes_topology_without_material_correspondence(self):
        case,base,target=self.diagnostic_case();report=assess(self.s,case,base,{'allowed_objects':['Face']})
        diagnostic=report['layers'][-1]
        self.assertEqual(diagnostic['vertices_before'],3);self.assertEqual(diagnostic['vertices_predicted'],4)
        self.assertIsNone(diagnostic['changed_count']);self.assertFalse(diagnostic['support'])
        result=compare(self.s,report['record'],target)
        self.assertTrue(result['agrees']);self.assertTrue(result['realized_diagnostics_agree'])

    def test_realized_diagnostic_requires_exact_arrays_independent_of_material_tolerance(self):
        case,base,target=self.diagnostic_case();report=assess(self.s,case,base,{'allowed_objects':['Face']})
        arrays=self.s.wb.arrays(target,'Diagnostic');arrays['co'][0,0]+=1e-9
        p=self.root/'diagnostic-drift.npz';np.savez_compressed(p,**arrays)
        state=self.s.store.get(target,'state');state['objects'][-1]['asset']=self.s.store.blob(p)
        result=compare(self.s,report['record'],self.s.store.put('state',state))
        self.assertTrue(result['realized_numerically_supported']);self.assertFalse(result['realized_diagnostics_agree'])
        self.assertFalse(result['agrees']);self.assertIsNone(result['layers'][-1]['max_residual'])

    def test_character_or_guide_row_cannot_claim_diagnostic_mode(self):
        original,base,target=self.diagnostic_case()
        for index in (0,1):
            case=deepcopy(original)
            case['layers'][index].update(mode='recomputed_diagnostic',reason='Attempt to bypass correspondence',derivation_evidence=case['construction_evidence'])
            with self.subTest(index=index),self.assertRaisesRegex(ValueError,'recorder-declared evaluated derived'):
                assess(self.s,case,base,{'allowed_objects':['Face']})

    def test_recovery_hashes_over_labels_and_runtime_distinction(self):
        file=self.root/'base.bin';file.write_bytes(b'original')
        pinned=dict(file=str(file),sha256=digest(file.read_bytes()))
        runtime=self.s.runtime_status()
        manifest=dict(retained=pinned,baseline=pinned,active_experiment=None,native_owner='fixture owner',preview={'status':'unknown'},
            selected_runtime=dict(python=runtime['python'],loaded_revision=runtime['loaded']['revision']))
        key=select(self.s,manifest,None)
        self.assertTrue(inspect(self.s)['selected_runtime_matches_process'])
        file.write_bytes(b'overwritten under same filename')
        state=inspect(self.s);self.assertEqual(state['status'],'needs_reconciliation')
        self.assertEqual(state['checkpoints']['baseline']['status'],'changed')
        self.assertEqual(state['runtime']['native_status'],'not contacted')
        self.assertEqual(state['record'],key)

    def test_long_media_source_and_atomic_failure_recovery(self):
        long=native_path(self.root/('a'*90)/('b'*90)/('c'*90)/'video.mp4');long.parent.mkdir(parents=True)
        writer=imageio_ffmpeg.write_frames(str(long),(16,16),fps=2,codec='libx264');writer.send(None)
        for color in (0,100,200):writer.send(bytes([color]*16*16*3))
        writer.close()
        result=self.s.analyze_video(str(long),maximum_frames=2)
        self.assertEqual(result['frames'],2);self.assertTrue(result['truncated'])
        dest=long.parent/'receipt.json';atomic_write(dest,b'old')
        with patch('modeling_system.store.os.replace',side_effect=OSError('interrupted')):
            with self.assertRaises(OSError):atomic_write(dest,b'new')
        self.assertEqual(dest.read_bytes(),b'old')
        self.assertEqual(len(list(dest.parent.glob('*.tmp'))),0)
        # Ordinary Windows traversal can omit long children during temp cleanup.
        # Remove only these known fixture files through their extended paths.
        dest.unlink();long.unlink();long.parent.rmdir()


class DeliveryAudit(unittest.TestCase):
    setUp=execution_fixtures.ExecutionReceiptTests.setUp
    write=execution_fixtures.ExecutionReceiptTests.write
    run_case=execution_fixtures.ExecutionReceiptTests.run_case

    def test_recent_name_cannot_mask_stale_export_and_between_anchor_error(self):
        source=self.root/'source.bin';source.write_bytes(b'revision one')
        current=self.root/'current.bin';current.write_bytes(b'revision two')
        reference=lambda p:dict(file=str(p),sha256=digest(p.read_bytes()))
        comparison=self.root/'comparison.npz';np.savez_compressed(comparison,source=[[0.,0,0]],output=[[0.,0,.0015]])
        delivery=dict(source=reference(source),current_source=reference(current),native_mechanism=ref(self.receipt_path),scope=['blink'],deferred=['mouth'],
            objects=['Face'],materials=['skin'],exports=[reference(self.candidate)],tolerance=.001,units='model units',samples=[
                dict(object='Face',controls={'blink':.85},kind='between_anchor',source_sha256=digest(source.read_bytes()),
                     export_sha256=digest(self.candidate.read_bytes()),comparison=ref(comparison))])
        self.case['delivery']=delivery
        result,payload=self.run_case();self.assertEqual(result['summary']['delivery']['status'],'stale')
        delivery['current_source']=reference(source)
        result,payload=self.run_case();self.assertEqual(result['summary']['delivery']['status'],'residual_exceeds_tolerance')
        self.assertAlmostEqual(payload['delivery']['samples'][0]['maximum'],.0015)
        self.assertFalse(payload['appearance_accepted'])
