"""Disconnected-component catalogs: exact index connectivity, lazy members, stale rejection."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from .bounded_reads import json_chars
from .service import ModelingService
from .store import canonical, digest
from . import component_catalog as cc


def integer_lists(value):
    """True when any list in the value carries integer identities (member IDs)."""
    if isinstance(value, dict): return any(integer_lists(v) for v in value.values())
    if isinstance(value, list): return any(type(v) is int for v in value) or any(integer_lists(v) for v in value)
    return False


def tetrahedra(count, spacing=5.):
    """count closed tetrahedral shells, four triangles each, no shared indices."""
    tet = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], np.int32)
    co = np.concatenate([tet + [spacing * i, 0, 0] for i in range(count)]) if count else np.zeros((0, 3))
    tri = np.concatenate([faces + 4 * i for i in range(count)]) if count else np.zeros((0, 3), np.int32)
    return dict(co=co, tri=tri.astype(np.int32))


class ComponentCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.s = ModelingService(workspace=self.root, store=self.root / 'store')

    def import_arrays(self, name, arrays, controls=None):
        path = self.root / (name + '.npz'); np.savez_compressed(path, **arrays)
        with np.load(path, allow_pickle=False) as z: loaded = {k: z[k] for k in z.files}
        ah = digest(canonical({k: digest(v.tobytes()) for k, v in loaded.items()}))
        state = {'objects': [{'name': 'Face', 'type': 'MESH', 'arrays': str(path), 'geometry_hash': ah, 'source': {}}],
                 'controls': controls or {}, 'selected_guide': 'Neutral', 'references': {}}
        record = {'state': state, 'state_id': digest(canonical(state)), 'coverage': [{'name': 'Face', 'included': True}]}
        record_path = self.root / (name + '.json'); record_path.write_text(json.dumps(record), encoding='utf-8')
        return self.s.import_scene(str(record_path))['state']

    def members(self, result):
        with np.load(self.s.store.resolve_blob(result['members']['asset']), allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def test_two_disconnected_shells_have_compact_metrics_and_no_member_lists(self):
        state = self.import_arrays('two', tetrahedra(2))
        r = cc.build(self.s, state, 'Face')
        s = r['summary']
        self.assertEqual((s['components'], s['triangles'], s['vertices'], s['unique_edges']), (2, 8, 8, 12))
        self.assertEqual((s['boundary_edges'], s['nonmanifold_edges'], s['max_edge_degree']), (0, 0, 2))
        self.assertEqual([p['id'] for p in r['preview']], ['c0', 'c4'])
        first = r['preview'][0]
        self.assertEqual((first['triangles'], first['vertices'], first['boundary_edges']), (4, 4, 0))
        self.assertAlmostEqual(first['surface_area'], 1.5 + np.sqrt(3) / 2)
        self.assertEqual(first['bounds'], [[0, 0, 0], [1, 1, 1]])
        self.assertEqual(r['preview'][1]['bounds'], [[5, 0, 0], [6, 1, 1]])
        self.assertAlmostEqual(s['surface_area'], 2 * first['surface_area'])
        self.assertEqual(r['connectivity']['kind'], 'shared_edge')
        self.assertLess(json_chars(r), 8000)
        self.assertFalse(integer_lists(r)); self.assertNotIn('"items"', json.dumps(r))
        self.assertIn('member_order', r['members']['arrays'])
        z = self.members(r)
        self.assertEqual(z['component_label'].tolist(), [0] * 4 + [1] * 4)
        self.assertEqual(z['geometry_hash'].item(), r['geometry_hash'])
        self.assertEqual(self.s.store.get(r['catalog'], 'component_catalog')['summary'], s)
        self.assertEqual(r['anatomy'], 'not implied by any component identity or metric')

    def test_vertex_touch_and_shared_edge_are_different_semantics(self):
        arrays = dict(co=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], float),
                      tri=np.array([[0, 1, 2], [0, 3, 4]], np.int32))
        state = self.import_arrays('bowtie', arrays)
        edge = cc.build(self.s, state, 'Face')
        touch = cc.build(self.s, state, 'Face', connectivity='shared_vertex')
        self.assertEqual(edge['summary']['components'], 2)
        self.assertEqual(edge['summary']['vertices_in_multiple_components'], 1)
        self.assertEqual(touch['summary']['components'], 1)
        self.assertEqual(touch['summary']['vertices_in_multiple_components'], 0)
        self.assertEqual(touch['preview'][0]['vertices'], 5)
        self.assertEqual([p['vertices'] for p in edge['preview']], [3, 3])
        self.assertNotEqual(edge['catalog'], touch['catalog'])
        self.assertEqual(touch['connectivity']['kind'], 'shared_vertex')
        with self.assertRaisesRegex(ValueError, 'Connectivity'): cc.build(self.s, state, 'Face', connectivity='nearby')

    def test_coincident_coordinates_do_not_connect_distinct_indices(self):
        p = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
        state = self.import_arrays('dup', dict(co=np.vstack([p, p]), tri=np.array([[0, 1, 2], [3, 4, 5]], np.int32)))
        for connectivity in ('shared_edge', 'shared_vertex'):
            r = cc.build(self.s, state, 'Face', connectivity=connectivity)
            self.assertEqual(r['summary']['components'], 2, connectivity)
            self.assertEqual(r['summary']['referenced_vertices'], 6)
            self.assertEqual(r['preview'][0]['bounds'], r['preview'][1]['bounds'])
        self.assertTrue(any('coincident' in x.lower() for x in r['limits']))

    def test_nonmanifold_edge_joins_incident_faces_and_is_recorded(self):
        arrays = dict(co=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]], float),
                      tri=np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]], np.int32))
        r = cc.build(self.s, self.import_arrays('fan', arrays), 'Face')
        s = r['summary']
        self.assertEqual(s['components'], 1)
        self.assertEqual((s['unique_edges'], s['boundary_edges'], s['nonmanifold_edges'], s['max_edge_degree']), (7, 6, 1, 3))
        self.assertEqual(s['edge_degree_histogram'], {'1': 6, '2': 0, '3': 1, '4+': 0})
        row = r['preview'][0]
        self.assertEqual((row['nonmanifold_edges'], row['boundary_edges'], row['max_edge_degree']), (1, 6, 3))
        self.assertTrue(any('ambiguous' in x for x in r['limits']))

    def test_empty_mesh_catalogs_zero_components(self):
        state = self.import_arrays('empty', tetrahedra(0))
        r = cc.build(self.s, state, 'Face')
        s = r['summary']
        self.assertEqual((s['components'], s['triangles'], s['vertices'], s['unique_edges'], s['surface_area']), (0, 0, 0, 0, 0))
        self.assertEqual(r['preview'], []); self.assertIsNone(r['select']); self.assertIsNone(r['locate'])
        page = cc.read(self.s, r['catalog'])
        self.assertEqual((page['total'], page['items'], page['content_complete']), (0, [], True))
        with self.assertRaisesRegex(ValueError, 'Unknown component'): cc.select(self.s, r['catalog'], 'c0', state)
        with self.assertRaisesRegex(ValueError, 'outside'): cc.locate(self.s, r['catalog'], 0)

    def test_invalid_indices_nonfinite_and_degenerate_triangles_are_explicit(self):
        co = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]], float)
        with self.assertRaisesRegex(ValueError, 'outside'): cc.analyze(dict(co=co, tri=np.array([[0, 1, 9]])))
        with self.assertRaisesRegex(ValueError, 'outside'): cc.analyze(dict(co=co, tri=np.array([[0, -1, 2]])))
        with self.assertRaisesRegex(ValueError, 'finite'): cc.analyze(dict(co=co * np.nan, tri=np.array([[0, 1, 2]])))
        with self.assertRaisesRegex(ValueError, 'integer'): cc.analyze(dict(co=co, tri=np.array([[0., 1, 2]])))
        tri = np.array([[0, 0, 1], [0, 1, 2], [3, 4, 5], [2, 1, 0]], np.int32)
        summary, columns, processing = cc.analyze(dict(co=co, tri=tri))
        self.assertEqual(summary['degenerate_triangles'], dict(repeated_index=1, zero_area=1, total=2))
        self.assertEqual(summary['duplicate_triangles'], 1)
        # The line-like triangle shares undirected edge (0,1) with the real face; it is a member, not hidden.
        self.assertEqual(columns['component_label'].tolist(), [0, 0, 1, 0])
        self.assertEqual(summary['components'], 2)
        self.assertEqual(summary['unique_edges'], 6)
        self.assertEqual(columns['component_degenerate_triangles'].tolist(), [1, 1])
        self.assertEqual(columns['component_duplicate_triangles'].tolist(), [1, 0])
        self.assertEqual(columns['component_max_edge_degree'].tolist(), [3, 1])
        self.assertFalse(processing['truncated'])
        canonical(summary)

    def test_large_catalog_stays_compact_and_selection_is_lazy(self):
        count = 20000
        state = self.import_arrays('many', tetrahedra(count, spacing=2.))
        r = cc.build(self.s, state, 'Face')
        self.assertEqual(r['summary']['components'], count)
        self.assertEqual(r['summary']['triangles'], 4 * count)
        self.assertEqual(r['processing']['triangles_processed'], 4 * count)
        self.assertLess(json_chars(r), 8000); self.assertFalse(integer_lists(r))
        self.assertEqual(len(r['preview']), cc.PREVIEW)
        record_bytes = (self.s.store.root / 'records' / (r['catalog'] + '.json')).stat().st_size
        self.assertLess(record_bytes, 20000)
        self.assertEqual(len(self.members(r)['component_label']), 4 * count)
        page = cc.read(self.s, r['catalog'], limit=20)
        self.assertEqual((page['total'], page['shown'], page['offset']), (count, 20, 0))
        self.assertEqual(page['next']['arguments']['offset'], 20)
        self.assertLessEqual(json_chars(page), 8000); self.assertFalse(integer_lists(page))
        by_label = cc.read(self.s, r['catalog'], offset=12345, limit=3, order='label')
        self.assertEqual([i['value']['id'] for i in by_label['items']], ['c49380', 'c49384', 'c49388'])
        chosen = 'c' + str(4 * 7777)
        sel = cc.select(self.s, r['catalog'], chosen, state)
        self.assertEqual(sel['members']['items'], [31108, 31109, 31110, 31111])
        self.assertEqual(sel['members']['total'], 4)
        self.assertTrue(sel['members']['content_complete'])
        self.assertEqual(sel['component']['id'], chosen)
        self.assertEqual(sel['component']['bounds'][0], [2. * 7777, 0, 0])
        self.assertLessEqual(json_chars(sel), 8000)
        self.assertEqual(cc.locate(self.s, r['catalog'], 31110)['component']['id'], chosen)
        vertices = cc.select(self.s, r['catalog'], chosen, state, members='vertices')
        self.assertEqual(vertices['members']['items'], [31108, 31109, 31110, 31111])

    def test_stale_state_and_mismatched_identity_are_rejected(self):
        original = self.import_arrays('a', tetrahedra(2))
        changed = self.import_arrays('b', tetrahedra(2, spacing=7.))
        same = self.import_arrays('c', tetrahedra(2), controls={'blink': 1})
        r = cc.build(self.s, original, 'Face')
        with self.assertRaisesRegex(ValueError, 'Stale catalog.*different or missing'):
            cc.select(self.s, r['catalog'], 'c0', changed)
        with self.assertRaisesRegex(ValueError, 'Stale catalog.*identical object geometry'):
            cc.select(self.s, r['catalog'], 'c0', same)
        with self.assertRaisesRegex(ValueError, 'Stale catalog.*unknown'):
            cc.select(self.s, r['catalog'], 'c0', 'f' * 64)
        with self.assertRaisesRegex(ValueError, 'expected stored state'): cc.select(self.s, r['catalog'], 'c0', '')
        with self.assertRaisesRegex(ValueError, 'geometry hash differs'):
            cc.select(self.s, r['catalog'], 'c0', original, expected_geometry_hash='0' * 64)
        with self.assertRaisesRegex(ValueError, 'Expected component_catalog'): cc.select(self.s, original, 'c0', original)
        with self.assertRaisesRegex(ValueError, 'Unknown component'): cc.select(self.s, r['catalog'], 'c1', original)
        with self.assertRaisesRegex(ValueError, 'Unknown component'): cc.select(self.s, r['catalog'], 0, original)
        good = cc.select(self.s, r['catalog'], 'c4', original, expected_geometry_hash=r['geometry_hash'])
        self.assertEqual(good['members']['items'], [4, 5, 6, 7])
        rebuilt = cc.build(self.s, same, 'Face')
        self.assertNotEqual(rebuilt['catalog'], r['catalog'])
        self.assertEqual([p['id'] for p in rebuilt['preview']], [p['id'] for p in r['preview']])
        self.assertEqual(rebuilt['members']['asset']['sha256'] == r['members']['asset']['sha256'], False)
        self.assertEqual(cc.select(self.s, rebuilt['catalog'], 'c4', same)['members']['items'], [4, 5, 6, 7])

    def test_stored_geometry_or_member_mutation_fails_integrity(self):
        state = self.import_arrays('m', tetrahedra(2))
        r = cc.build(self.s, state, 'Face')
        asset = self.s.store.get(state, 'state')['objects'][0]['asset']
        (self.s.store.root / asset['path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'integrity'): cc.select(self.s, r['catalog'], 'c0', state)
        with self.assertRaisesRegex(ValueError, 'integrity'): cc.build(self.s, state, 'Face')
        self.assertEqual(cc.read(self.s, r['catalog'])['total'], 2)
        (self.s.store.root / r['members']['asset']['path']).write_bytes(b'changed')
        for call in (lambda: cc.read(self.s, r['catalog']), lambda: cc.locate(self.s, r['catalog'], 0)):
            with self.assertRaisesRegex(ValueError, 'integrity'): call()

    def test_original_triangle_and_polygon_identities_are_preserved(self):
        co = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [3, 0, 0], [4, 0, 0], [4, 1, 0], [3, 1, 0]], float)
        # Two quads, each tessellated into two triangles, interleaved in triangle order.
        tri = np.array([[0, 1, 2], [4, 5, 6], [0, 2, 3], [4, 6, 7]], np.int32)
        state = self.import_arrays('quads', dict(co=co, tri=tri, triangle_polygon=np.array([0, 1, 0, 1])))
        r = cc.build(self.s, state, 'Face')
        self.assertEqual(r['summary']['polygons'], dict(r['summary']['polygons'], status='recorded', count=2, split_across_components=0))
        self.assertEqual([p['id'] for p in r['preview']], ['c0', 'c1'])
        self.assertEqual([p['polygons'] for p in r['preview']], [1, 1])
        self.assertIn('not recomputed', r['connectivity']['basis'])
        second = cc.select(self.s, r['catalog'], 'c1', state)
        self.assertEqual(second['members']['items'], [1, 3])
        self.assertEqual(cc.select(self.s, r['catalog'], 'c1', state, members='polygons')['members']['items'], [1])
        self.assertEqual(cc.select(self.s, r['catalog'], 'c1', state, members='vertices')['members']['items'], [4, 5, 6, 7])
        self.assertEqual(cc.locate(self.s, r['catalog'], 3)['component']['id'], 'c1')
        split = self.import_arrays('split', dict(co=co, tri=tri, triangle_polygon=np.array([0, 0, 0, 0])))
        r2 = cc.build(self.s, split, 'Face')
        self.assertEqual(r2['summary']['components'], 2)
        self.assertEqual(r2['summary']['polygons']['split_across_components'], 1)
        plain = self.import_arrays('plain', dict(co=co, tri=tri))
        r3 = cc.build(self.s, plain, 'Face')
        self.assertEqual(r3['summary']['polygons']['status'], 'absent')
        self.assertNotIn('polygons', r3['preview'][0])
        self.assertIn('native polygon connectivity is not established', r3['connectivity']['basis'])
        with self.assertRaisesRegex(ValueError, 'triangle_polygon'): cc.select(self.s, r3['catalog'], 'c0', plain, members='polygons')
        bad = self.import_arrays('badmap', dict(co=co, tri=tri, triangle_polygon=np.array([0, 1])))
        with self.assertRaisesRegex(ValueError, 'triangle_polygon'): cc.build(self.s, bad, 'Face')

    def test_recorded_component_semantics_do_not_define_connectivity(self):
        p = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
        arrays = dict(co=np.vstack([p, p + [3, 0, 0]]), tri=np.array([[0, 1, 2], [3, 4, 5]], np.int32),
                      triangle_component=np.array([7, 7]))
        r = cc.build(self.s, self.import_arrays('semantic', arrays), 'Face')
        self.assertEqual(r['summary']['components'], 2)
        self.assertEqual(r['summary']['recorded_component_semantics']['values'], 1)
        self.assertEqual(r['summary']['recorded_component_semantics']['spanning_multiple_catalog_components'], 1)
        self.assertEqual([p['recorded_components'] for p in r['preview']], [1, 1])

    def test_budget_is_explicit_and_never_truncates(self):
        state = self.import_arrays('budget', tetrahedra(2))
        with self.assertRaisesRegex(ValueError, 'budget exceeded.*nothing was truncated'):
            cc.build(self.s, state, 'Face', max_triangles=3)
        for bad in (0, -1, 2.5, cc.CEILING_TRIANGLES + 1):
            with self.assertRaisesRegex(ValueError, 'max_triangles'): cc.build(self.s, state, 'Face', max_triangles=bad)
        r = cc.build(self.s, state, 'Face', max_triangles=8)
        self.assertEqual(r['processing']['budget'], dict(max_triangles=8, ceiling=cc.CEILING_TRIANGLES))
        self.assertEqual(self.s.store.get(r['catalog'], 'component_catalog')['processing']['triangles_processed'], 8)

    def test_rebuild_is_deterministic_and_descriptors_execute(self):
        state = self.import_arrays('det', tetrahedra(3))
        first = cc.build(self.s, state, 'Face'); second = cc.build(self.s, state, 'Face')
        self.assertEqual(first['catalog'], second['catalog'])
        self.assertEqual(first['members']['asset'], second['members']['asset'])
        for name, descriptor in first['reads'].items():
            if descriptor['operation'] != 'read_record':
                self.assertEqual((name, descriptor['operation']), ('components', cc.READ_OPERATION)); continue
            out = self.s.execute(descriptor['operation'], descriptor['arguments'])
            self.assertNotEqual(out['status'], 'failed', out)
            self.assertLessEqual(json_chars(out), descriptor['arguments']['max_chars'])
        self.assertEqual(first['select']['operation'], cc.SELECT_OPERATION)
        chosen = cc.select(self.s, **first['select']['arguments'])
        self.assertEqual(chosen['component']['id'], first['preview'][0]['id'])
        page = cc.read(self.s, **first['reads']['components']['arguments'])
        self.assertEqual(page['total'], 3)
        located = cc.locate(self.s, **first['locate']['arguments'])
        self.assertEqual(located['component']['id'], 'c0')
        self.assertEqual(cc.select(self.s, **located['select']['arguments'])['members']['items'], [0, 1, 2, 3])

    def test_member_windows_page_within_max_chars(self):
        n = 3000
        co = np.column_stack([np.arange(n + 2, dtype=float), np.arange(n + 2) % 2, np.zeros(n + 2)])
        tri = np.column_stack([np.arange(n), np.arange(n) + 1, np.arange(n) + 2]).astype(np.int32)
        state = self.import_arrays('strip', dict(co=co, tri=tri))
        r = cc.build(self.s, state, 'Face')
        self.assertEqual(r['summary']['components'], 1)
        self.assertEqual(r['summary']['boundary_edges'], n + 2)
        small = cc.select(self.s, r['catalog'], 'c0', state, limit=1024, max_chars=2048)
        self.assertLess(small['members']['shown'], 1024); self.assertGreater(small['members']['shown'], 0)
        self.assertLessEqual(json_chars(small), 2048)
        self.assertEqual(small['members']['next']['arguments']['offset'], small['members']['shown'])
        seen = []; args = dict(catalog=r['catalog'], component='c0', expected_state=state, limit=1024, max_chars=8000)
        while True:
            out = cc.select(self.s, **args); seen += out['members']['items']
            if 'next' not in out['members']: break
            args = out['members']['next']['arguments']
        self.assertEqual(seen, list(range(n)))
        with self.assertRaisesRegex(ValueError, 'at most 64'): cc.read(self.s, r['catalog'], limit=65)
        with self.assertRaisesRegex(ValueError, 'order'): cc.read(self.s, r['catalog'], order='random')
        with self.assertRaisesRegex(ValueError, 'members must be'): cc.select(self.s, r['catalog'], 'c0', state, members='edges')
        with self.assertRaises(IndexError): cc.select(self.s, r['catalog'], 'c0', state, offset=n + 1)


    def test_public_read_routes_are_executable_and_do_not_write(self):
        state=self.import_arrays('public',tetrahedra(2))
        result=self.s.build_component_catalog(state,'Face')
        before={str(p.relative_to(self.s.store.root)):p.read_bytes() for p in self.s.store.root.rglob('*') if p.is_file()}
        page=self.s.read_component_catalog(result['catalog'],offset=1,limit=1)
        selected=self.s.execute(page['select']['operation'],page['select']['arguments'])
        self.assertEqual(selected['component']['id'],page['items'][0]['value']['id'])
        self.s.locate_component(result['catalog'],0)
        after={str(p.relative_to(self.s.store.root)):p.read_bytes() for p in self.s.store.root.rglob('*') if p.is_file()}
        self.assertEqual(before,after)
        for budget in range(2048,2600,53):
            row=self.s.select_component(result['catalog'],'c0',state,max_chars=budget)
            self.assertLessEqual(json_chars(row),budget)


if __name__ == '__main__': unittest.main()
