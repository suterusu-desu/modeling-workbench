"""Motion feedback risks: frame identity, speed mapping, stale drafts and replay."""
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import imageio_ffmpeg
from .service import ModelingService
from . import test_visual_feedback as fixtures
from .visual_feedback_server import make_server


class VisualFeedbackVideoTests(unittest.TestCase):
    # Reuse the anonymous image/comparison setup, without inheriting its tests.
    setUp = fixtures.VisualFeedbackTests.setUp
    call = fixtures.VisualFeedbackTests.call
    mutate = fixtures.VisualFeedbackTests.mutate
    fact = fixtures.VisualFeedbackTests.fact
    add_image = fixtures.VisualFeedbackTests.add_image
    compare = fixtures.VisualFeedbackTests.compare
    feedback_args = fixtures.VisualFeedbackTests.feedback_args

    @classmethod
    def setUpClass(cls):
        cls.media = tempfile.TemporaryDirectory()
        cls.clips = []
        raw = b''.join(bytes([30+i*20, 110, 160]) * (240*90) for i in range(8))
        for fps in (8, 2):
            path = Path(cls.media.name) / f'blink-{fps}.mp4'
            result = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-v', 'error',
                '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '240x90', '-r', str(fps), '-i', 'pipe:0',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)],
                input=raw, capture_output=True)
            if result.returncode: raise RuntimeError('Synthetic MP4 creation failed')
            cls.clips.append(path)

    @classmethod
    def tearDownClass(cls):
        cls.media.cleanup()

    def add_video(self, slow=False, **overrides):
        metadata = dict(label='Slow blink' if slow else 'Normal blink', speed_label='slow' if slow else 'normal',
            source='Synthetic exact source frame sequence', captured_at='2026-01-01',
            baseline_version='baseline-v1', result_version='trial-v1', camera='front-v1',
            display_state='bare-surface', lighting='fixed soft studio', framing='full frame, orthographic 1.0',
            view_role='eye_context', layout='baseline_left_trial_right', sequence_id='synthetic-sequence-v1',
            frame_ids=[f'moment-{i}' for i in range(8)], related_image=self.inspection)
        metadata.update(overrides)
        return self.mutate('video', video_path=str(self.clips[int(slow)]), metadata=metadata)['added']['videos'][0]

    def link(self, *videos):
        return self.mutate('motion', comparison=self.comparison, videos=list(videos))['added']['motion_reviews'][0]

    def moment(self, review, video, frame=4):
        clip = self.service._visual_feedback()._record(self.service.ledger.read(self.board), 'videos', video)
        return dict(review=review, video=video, frame_index=frame, timestamp_seconds=clip['frame_times'][frame],
                    region=[.6,.3,.2,.2], panel='trial')

    def failed_submit(self, motion, expected='failed'):
        result = self.service.execute('visual_feedback_submit', dict(board=self.board,
            expected_revision=self.revision, **self.feedback_args(), motion=motion))
        self.assertEqual(result['status'], expected, result)

    def test_normal_slow_exact_frame_feedback_and_reopen(self):
        normal, slow = self.add_video(), self.add_video(True)
        motion_review = self.link(normal, slow)
        normal_moment, slow_moment = self.moment(motion_review, normal), self.moment(motion_review, slow)
        self.assertEqual(normal_moment['timestamp_seconds'], .5)
        self.assertEqual(slow_moment['timestamp_seconds'], 2.)
        self.mutate('agreement', target=self.target, plan=self.plan, fact=self.fact())
        self.mutate('submit', **self.feedback_args(), motion=normal_moment)
        self.mutate('submit', **self.feedback_args(), motion=slow_moment)
        self.service = ModelingService(self.root)
        view = self.call('read', board=self.board)
        self.assertEqual(view['motion_reviews'][0]['matching']['status'], 'matched declared inputs')
        self.assertEqual([f['motion']['moment_id'] for f in view['feedback']], ['moment-4', 'moment-4'])
        self.assertEqual([f['motion']['timestamp_seconds'] for f in view['feedback']], [.5, 2.])
        self.assertEqual(view['feedback'][0]['motion']['related_image'], self.inspection)
        self.assertEqual(view['targets'][0]['wording'], '  Smooth the skin; preserve its outline.\nDo not change the curve.  ')
        self.assertEqual(len(view['agreements']), 1)
        self.assertFalse(view['state_facts'])
        self.assertFalse(view['native_effects'])
        self.assertIn('moment-4', self.call('summary', board=self.board)['markdown'])

    def test_forged_time_frame_clip_and_panel_cannot_apply(self):
        normal = self.add_video()
        review = self.link(normal)
        moment = self.moment(review, normal)
        self.failed_submit(dict(moment, timestamp_seconds=2.), 'conflicting')
        self.failed_submit(dict(moment, frame_index=True))
        self.failed_submit(dict(moment, frame_index=8))
        self.failed_submit(dict(moment, region=[.4,.3,.2,.2]))
        other = self.add_video(True)
        self.failed_submit(dict(moment, video=other), 'conflicting')
        self.assertFalse(self.call('read', board=self.board)['feedback'])

    def test_stale_motion_and_comparison_remain_historical(self):
        normal = self.add_video()
        review = self.link(normal)
        moment = self.moment(review, normal)
        self.link(normal)  # A new explicit motion review supersedes only its link.
        self.failed_submit(moment, 'conflicting')
        self.mutate('submit', **self.feedback_args(), motion=moment, historical=True)
        self.assertIn('historical', self.call('read', board=self.board)['feedback'][0]['applicability'])
        self.comparison = self.compare()
        self.failed_submit(moment, 'conflicting')

    def test_mismatched_or_missing_mapping_is_not_false_agreement(self):
        normal = self.add_video()
        slow = self.add_video(True, frame_ids=[f'other-{i}' for i in range(8)], result_version='different-result')
        self.link(normal, slow)
        view = self.call('read', board=self.board)
        self.assertEqual(view['motion_reviews'][-1]['matching']['status'], 'unmatched')
        self.assertTrue(view['motion_reviews'][-1]['moment_mapping'].startswith('unmatched'))
        unknown = self.add_video(True, frame_ids=None, sequence_id=None, camera=None, layout=None)
        self.link(unknown)
        view = self.call('read', board=self.board)
        self.assertEqual(view['motion_reviews'][-1]['matching']['status'], 'unknown')
        self.assertFalse(view['agreements'])
        self.assertFalse(view['state_facts'])
        result = self.service.execute('visual_feedback_video', dict(board=self.board, expected_revision=self.revision,
            video_path=str(self.clips[0]), metadata=dict(view['videos'][0]['metadata'], frame_ids=['same']*8)))
        self.assertEqual(result['status'], 'failed')

    def test_pinned_http_video_ranges_and_origin_guards(self):
        normal = self.add_video()
        server = make_server(self.service, 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f'http://127.0.0.1:{server.server_port}/api/video/{self.board}/{normal}'
            with urlopen(Request(url, headers={'Range':'bytes=5-19'})) as response:
                self.assertEqual(response.status, 206)
                self.assertEqual(response.read(), self.clips[0].read_bytes()[5:20])
                self.assertTrue(response.headers['Content-Range'].startswith('bytes 5-19/'))
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(url, headers={'Range':'bytes=99999999-'}))
            self.assertEqual(error.exception.code, 416)
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(url, headers={'Origin':'http://foreign.invalid'}))
            self.assertEqual(error.exception.code, 403)
        finally:
            server.shutdown(); server.server_close(); thread.join()

    def test_partial_cycle_mapping_keeps_unshared_moments_unknown(self):
        normal = self.add_video()
        slow = self.add_video(True, frame_ids=[f'moment-{i}' for i in range(4)]+[f'extra-{i}' for i in range(4)])
        self.link(normal, slow)
        review = self.call('read', board=self.board)['motion_reviews'][-1]
        self.assertEqual(review['matching']['status'], 'unknown')
        self.assertTrue(review['moment_mapping'].startswith('same explicit shared source moment IDs (4)'))
        self.assertTrue(any('only one speed' in x for x in review['matching']['unknown']))
        self.assertEqual(review['speed_matching']['status'], 'matched declared inputs')
        self.assertTrue(review['speed_matching']['mapping_allowed'])

    def test_nonzero_source_timestamps_are_not_rebased_in_feedback(self):
        offset = self.root/'offset.mp4'
        result = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-i', str(self.clips[0]),
            '-c', 'copy', '-output_ts_offset', '5', str(offset)], capture_output=True)
        self.assertEqual(result.returncode, 0)
        normal = self.add_video()
        metadata = self.call('read', board=self.board)['videos'][0]['metadata']
        clip = self.mutate('video', video_path=str(offset), metadata=metadata)['added']['videos'][0]
        review = self.link(clip)
        frame = self.moment(review, clip, 0)
        self.assertEqual(frame['timestamp_seconds'], 5.)
        stored = self.call('read', board=self.board)['videos'][-1]
        self.assertEqual(stored['playback_times'][0], 0.)
        self.failed_submit(dict(frame, timestamp_seconds=0.), 'conflicting')
        self.mutate('submit', **self.feedback_args(), motion=frame)
        self.assertEqual(self.call('read', board=self.board)['feedback'][0]['motion']['timestamp_seconds'], 5.)
