from copy import deepcopy
from types import SimpleNamespace as NS
import unittest

from .controller import fingerprint
from .mirror_inventory import (ARRAY_FIELDS, BOOL_FIELDS, FLOAT_FIELDS,
                               DISPLAY_FIELDS, MIRROR_FIELDS, mirror_structure)


def fixture():
    inherited = [NS(identifier=k, is_readonly=False) for k in DISPLAY_FIELDS]
    fields = inherited + [NS(identifier=k, is_readonly=False) for k in MIRROR_FIELDS]
    mod = NS(type='MIRROR', bl_rna=NS(identifier='MirrorModifier', properties=fields,
             base=NS(properties=inherited)), mirror_object=None)
    for k in ARRAY_FIELDS: setattr(mod, k, [True, False, False])
    for k in BOOL_FIELDS + DISPLAY_FIELDS: setattr(mod, k, True)
    for k in FLOAT_FIELDS: setattr(mod, k, .001)
    return mod


class MirrorTests(unittest.TestCase):
    def read(self, mod):
        return mirror_structure(mod, object_reference=lambda value: deepcopy(value))

    def test_every_field_and_object_identity_changes_the_record(self):
        baseline = fixture(); expected = fingerprint(self.read(baseline))
        for key in ARRAY_FIELDS + BOOL_FIELDS + FLOAT_FIELDS + DISPLAY_FIELDS:
            changed = deepcopy(baseline)
            if key in ARRAY_FIELDS: getattr(changed, key)[1] = True
            elif key in BOOL_FIELDS + DISPLAY_FIELDS: setattr(changed, key, False)
            else: setattr(changed, key, .002)
            self.assertNotEqual(fingerprint(self.read(changed)), expected, key)
        baseline.mirror_object = {'id_type':'OBJECT', 'name':'Symmetry plane', 'library':None}
        local = self.read(baseline)
        self.assertNotEqual(fingerprint(local), expected)
        baseline.mirror_object['library'] = '//linked.blend'
        self.assertNotEqual(self.read(baseline), local)

    def test_missing_or_new_fields_fail_closed(self):
        for change in ('missing', 'new'):
            mod = fixture()
            if change == 'missing': mod.bl_rna.properties.pop()
            else: mod.bl_rna.properties.append(NS(identifier='new_effect', is_readonly=False))
            with self.assertRaises(ValueError): self.read(mod)

    def test_invalid_values_or_reference_are_rejected(self):
        for key, value in [('merge_threshold', float('nan')), ('bisect_threshold', -1),
                           ('use_axis', [True, False]), ('use_clip', 1),
                           ('mirror_object', {'name':'ambiguous'}), ('type', 'NODES')]:
            mod = fixture(); setattr(mod, key, value)
            with self.assertRaises(ValueError): self.read(mod)

    def test_record_does_not_mutate_source_and_has_no_pose_coordinates(self):
        mod = fixture(); before = deepcopy(mod.__dict__)
        record = self.read(mod)
        self.assertEqual(mod.__dict__, before)
        self.assertNotIn('matrix_world', record)
        record['use_axis'][0] = False
        self.assertTrue(mod.use_axis[0])


if __name__ == '__main__': unittest.main()
