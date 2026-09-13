"""Tests for motion evidence that previously could be mistaken for demonstrated continuity."""
import copy
import unittest
from . import test_workflow


class ContinuityTests(unittest.TestCase):
    def setUp(self):
        self.fixture=test_workflow.WorkflowTests();self.fixture.setUp()
        self.s=self.fixture.s;self.root=self.fixture.root
        context=self.s.record_observation(self.fixture.state,str(self.fixture.image),dict(self.fixture.meta,crop=None),'context')['observation']
        phases=['open','closing','closing','contact','reopening','reopening','open_rest']
        self.frames=[];video=[];self.mapping=[]
        for i,(phase,value) in enumerate(zip(phases,[0,.2,.8,1,.8,.2,0])):
            receipt=self.root/(str(i)+'.json');receipt.write_text('{"independent_capture_index":'+str(i)+'}')
            self.frames.append(dict(index=i,controls={'blink':value},pose={'controls':{'blink':value}},state=self.fixture.state,
                native_receipt=self.s.store.blob(receipt),observations=[{'observation':self.fixture.ob},{'observation':context}]))
            video.append({'index':i,'time_seconds':i/24,'image':self.s.store.blob(self.fixture.image)})
            self.mapping.append(dict(native_index=i,reference_index=i,phase=phase,basis='Explicit fixture correspondence'))
        self.video=self.s.store.put('video_analysis',{'frames':video})

    def tearDown(self):self.fixture.tearDown()

    def native(self,frames=None):
        return self.s.store.put('native_motion',dict(frames=frames or self.frames,coverage={'complete_requested_capture':True},
            native_preservation={'keys_unchanged':True,'pose_restored':True}))

    def test_real_order_and_paired_views_have_finite_evidence_without_appearance_approval(self):
        motion=self.native()
        result=self.s.assess_motion_continuity(motion,self.video,self.mapping,baseline_motion=motion,object_name='Face')
        self.assertEqual(result['coverage_status'],'complete requested phase evidence')
        self.assertEqual(result['missing'],[]);self.assertIn('not established',result['appearance_acceptance'])
        self.assertTrue(all(r['comparison'] for r in result['rows']))

    def test_reversed_closing_cache_is_refused_even_when_pose_values_match(self):
        frames=copy.deepcopy(self.frames)
        frames[4]['native_receipt']=frames[2]['native_receipt']
        with self.assertRaisesRegex(ValueError,'repeated capture receipt'):
            self.s.assess_motion_continuity(self.native(frames),self.video,self.mapping)

    def test_out_of_range_or_reversed_video_correspondence_is_refused(self):
        for changes in ({'reference_index':99},{'reference_index':1}):
            mapping=copy.deepcopy(self.mapping);mapping[4].update(changes)
            with self.assertRaises(ValueError):self.s.assess_motion_continuity(self.native(),self.video,mapping)

    def test_closing_only_and_missing_context_are_explicit_gaps(self):
        result=self.s.assess_motion_continuity(self.native(),self.video,self.mapping[:4])
        self.assertEqual(result['coverage_status'],'partial');self.assertIn('open_rest phase',result['missing'])
        frames=copy.deepcopy(self.frames)
        for f in frames:f['observations']=f['observations'][:1]
        result=self.s.assess_motion_continuity(self.native(frames),self.video,self.mapping)
        self.assertTrue(any('context' in m for m in result['missing']))

    def test_changed_view_and_wrong_native_direction_are_refused(self):
        frames=copy.deepcopy(self.frames)
        other=self.s.record_observation(self.fixture.state,str(self.fixture.image),dict(self.fixture.meta,crop=[1,2,200,100]))['observation']
        frames[4]['observations'][0]={'observation':other}
        with self.assertRaisesRegex(ValueError,'viewing conditions changed'):
            self.s.assess_motion_continuity(self.native(frames),self.video,self.mapping)
        frames=copy.deepcopy(self.frames);frames[5]['controls']['blink']=.9
        with self.assertRaisesRegex(ValueError,'motion direction'):
            self.s.assess_motion_continuity(self.native(frames),self.video,self.mapping)

    def test_equal_pixel_crops_with_different_native_zoom_are_distinct_views(self):
        frames=copy.deepcopy(self.frames)
        meta=copy.deepcopy(self.fixture.meta)
        meta['view_projection_matrix'][0][0]*=2
        close=self.s.record_observation(self.fixture.state,str(self.fixture.image),meta,'actual close-up')['observation']
        for frame in frames:
            frame['observations']=[{'observation':self.fixture.ob},{'observation':close}]
        result=self.s.assess_motion_continuity(self.native(frames),self.video,self.mapping)
        self.assertEqual(result['missing'],[])
        for frame in frames:frame['observations'][1]={'observation':self.fixture.ob}
        result=self.s.assess_motion_continuity(self.native(frames),self.video,self.mapping)
        self.assertTrue(any('distinct close-up' in x for x in result['missing']))


if __name__=='__main__':unittest.main()
