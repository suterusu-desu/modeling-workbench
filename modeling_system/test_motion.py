"""Replay of matched captures: before/after and more than two columns."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import imageio_ffmpeg
from PIL import Image

from .motion import Motion
from .workbench import Workbench


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, self.root, True)
        self.motion = Motion(Workbench(self.root / 'store'), self.root / 'media')

    def manifest(self, columns, labels=None):
        frames = []
        for k, blink in enumerate((0., .5, 1.)):
            frame = {'blink': blink}
            for c, side in enumerate(columns):
                path = self.root / f'{side}-{k}.png'
                Image.new('RGB', (64, 48), (40 * c + 30, 80 * k, 120)).save(path)
                frame[side] = {'front': {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}
            frames.append(frame)
        m = {'frames': frames, 'timing': {'close': .15, 'hold': .06, 'reopen': .24}, 'cycle_s': .9,
             'labels': labels or {}, 'limits': 'synthetic'}
        if columns != ['before', 'after']:
            m['columns'] = columns
        path = self.root / f'manifest-{len(columns)}.json'; path.write_text(json.dumps(m))
        return path

    def width(self, video):
        probe = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-i', str(video)], capture_output=True, text=True)
        return int(probe.stderr.split('Video:')[1].split(',')[2].strip().split('x')[0].split()[-1])

    def test_three_variants_replay_side_by_side_with_their_labels(self):
        imported = self.motion.import_player(self.manifest(['A', 'B', 'C'], {'A': 'still', 'B': 'half', 'C': 'shallow'}))
        receipt = self.motion.encode(imported['motion'], self.root / 'replay', normal_cycles=1, slow_cycles=1,
                                     panel_size=[64, 48])
        video = Path(receipt['outputs'][0]['path'])
        self.assertTrue(receipt['outputs'][0]['decoded_without_errors'])
        self.assertEqual(self.width(video), 3 * 64)

    def test_before_and_after_stay_the_default(self):
        imported = self.motion.import_player(self.manifest(['before', 'after'], {'before': 'old', 'after': 'new'}))
        receipt = self.motion.encode(imported['motion'], self.root / 'pair', normal_cycles=1, slow_cycles=0,
                                     panel_size=[64, 48])
        self.assertEqual(self.width(Path(receipt['outputs'][0]['path'])), 2 * 64)

    def test_column_names_are_checked(self):
        for columns in (['A', 'A'], [], ['A'] * 7):
            with self.assertRaisesRegex(ValueError, 'Columns'):
                Motion.columns({'columns': columns} if columns else {'columns': [1]})


if __name__ == '__main__':
    unittest.main()
