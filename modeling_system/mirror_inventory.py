"""Strict Mirror modifier source inventory; no Blender operations or mutation.

An adapter supplies object identity resolution and retains the referenced object's
transform, constraints and animation in its full scene dependency inventory.
Describing Mirror does not certify topology preservation or attachment indices.
"""
import math

ARRAY_FIELDS = ('use_axis', 'use_bisect_axis', 'use_bisect_flip_axis')
BOOL_FIELDS = ('use_clip', 'use_mirror_merge', 'use_mirror_u', 'use_mirror_v',
               'use_mirror_udim', 'use_mirror_vertex_groups')
FLOAT_FIELDS = ('merge_threshold', 'bisect_threshold', 'mirror_offset_u',
                'mirror_offset_v', 'offset_u', 'offset_v')
DISPLAY_FIELDS = ('show_viewport', 'show_render', 'show_in_editmode', 'show_on_cage')
MIRROR_FIELDS = frozenset(ARRAY_FIELDS + BOOL_FIELDS + FLOAT_FIELDS + ('mirror_object',))


def mirror_structure(modifier, *, object_reference):
    """Read all supported Mirror RNA fields without inventing absent defaults."""
    if modifier.type != 'MIRROR' or modifier.bl_rna.identifier != 'MirrorModifier':
        raise ValueError('Expected the native Mirror modifier RNA type')
    inherited = {p.identifier for p in modifier.bl_rna.base.properties}
    specific = {p.identifier for p in modifier.bl_rna.properties
                if p.identifier not in inherited and not p.is_readonly}
    if specific != MIRROR_FIELDS:
        raise ValueError('Unsupported Mirror RNA fields: missing=' + str(sorted(MIRROR_FIELDS-specific))
                         + ', extra=' + str(sorted(specific-MIRROR_FIELDS)))
    result = {}
    for key in ARRAY_FIELDS:
        values = list(getattr(modifier, key))
        if len(values) != 3 or any(type(v) is not bool for v in values):
            raise ValueError('Mirror axes require three booleans: ' + key)
        result[key] = values
    for key in BOOL_FIELDS + DISPLAY_FIELDS:
        value = getattr(modifier, key)
        if type(value) is not bool:
            raise ValueError('Mirror setting requires a boolean: ' + key)
        result[key] = value
    for key in FLOAT_FIELDS:
        value = getattr(modifier, key)
        if type(value) not in (float, int) or not math.isfinite(value):
            raise ValueError('Mirror setting requires a finite number: ' + key)
        if key in ('merge_threshold', 'bisect_threshold') and value < 0:
            raise ValueError('Mirror threshold cannot be negative: ' + key)
        result[key] = float(value)
    target = modifier.mirror_object
    reference = None if target is None else object_reference(target)
    if reference is not None and (not isinstance(reference, dict)
            or set(reference) != {'id_type', 'name', 'library'}
            or reference['id_type'] != 'OBJECT' or not isinstance(reference['name'], str)
            or not reference['name'] or reference['library'] is not None
            and not isinstance(reference['library'], str)):
        raise ValueError('Mirror reference requires exact object and library identity')
    if target is not None and reference is None:
        raise ValueError('A configured Mirror object cannot disappear from the inventory')
    result['mirror_object'] = reference
    return result
