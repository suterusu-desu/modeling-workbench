"""Review sheets arrange actual images and report unmatched evidence; no rendering or judgment."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from .review_sheets import compose_review_sheet, rows_from_frames


class ReviewSheetTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def image(self, name, size=(64, 48), color=(200, 120, 40)):
        path = self.root / name; Image.new('RGB', size, color).save(path); return path

    def test_grid_pins_sources_and_reports_a_matched_comparison(self):
        before, after = self.image('before.png'), self.image('after.png', color=(40, 120, 200))
        result = compose_review_sheet([{'label': 'corner .85', 'images': [before, after]}], ['baseline', 'candidate'],
                                      self.root / 'sheet.png', cell=(80, 60), title='Matched corner review')
        with Image.open(result['path']) as sheet:
            self.assertEqual(sheet.size, (150 + 2 * 80, 28 + 24 + 60))
        self.assertTrue(result['matched_sizes']); self.assertEqual(result['missing'], [])
        self.assertEqual([s['sha256'] for s in result['sources']],
                         [hashlib.sha256(p.read_bytes()).hexdigest() for p in (before, after)])

    def test_missing_images_and_size_mismatch_are_reported_not_hidden(self):
        before = self.image('before.png'); larger = self.image('larger.png', size=(128, 96))
        rows = [{'label': 'front 1.0', 'images': [before, self.root / 'absent.png']},
                {'label': 'oblique 1.0', 'images': [larger, None]}]
        result = compose_review_sheet(rows, ['baseline', 'candidate'], self.root / 'sheet.png')
        self.assertEqual(len(result['missing']), 1); self.assertEqual(result['missing'][0]['row'], 'front 1.0')
        self.assertFalse(result['matched_sizes']); self.assertEqual(len(result['sources']), 2)

    def test_existing_output_malformed_rows_and_other_formats_refuse(self):
        before = self.image('before.png')
        compose_review_sheet([{'label': 'a', 'images': [before]}], ['only'], self.root / 'sheet.png')
        with self.assertRaises(FileExistsError):
            compose_review_sheet([{'label': 'a', 'images': [before]}], ['only'], self.root / 'sheet.png')
        with self.assertRaisesRegex(ValueError, 'one image entry per column'):
            compose_review_sheet([{'label': 'a', 'images': [before]}], ['one', 'two'], self.root / 'other.png')
        with self.assertRaisesRegex(ValueError, 'png'):
            compose_review_sheet([{'label': 'a', 'images': [before]}], ['only'], self.root / 'sheet.jpg')

    def test_rows_from_frames_keep_absent_combinations_and_refuse_ambiguity(self):
        frames = [{'view': 'corner', 'pose': .5, 'state': 'before', 'path': 'b5.png'},
                  {'view': 'corner', 'pose': .5, 'state': 'after', 'path': 'a5.png'},
                  {'view': 'corner', 'pose': .7, 'state': 'after', 'path': 'a7.png'}]
        rows = rows_from_frames(frames, views=['corner'], poses=[.5, .7], states=['before', 'after'])
        self.assertEqual(rows[0]['images'], ['b5.png', 'a5.png']); self.assertEqual(rows[1]['images'], [None, 'a7.png'])
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            rows_from_frames(frames + [dict(frames[0], path='other.png')], views=['corner'], poses=[.5], states=['before'])


if __name__ == '__main__': unittest.main()
