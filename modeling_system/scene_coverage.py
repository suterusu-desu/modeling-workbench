"""Normalize declared coverage without inventing omitted object-level evidence."""
from copy import deepcopy


def normalize(value):
    if isinstance(value, list):
        if any(not isinstance(r, dict) or not isinstance(r.get('name',r.get('object')), str)
               or type(r.get('included')) is not bool for r in value):
            raise ValueError('coverage list requires objects with name and boolean included; retain exclusion reasons')
        return dict(schema='object_rows', objects=deepcopy(value), declaration=None)
    if isinstance(value, dict) and 'included' in value and 'objects' in value:
        # Historical importers declared a scope sentence and inventory count,
        # not per-object inclusion. Preserve that distinction and the raw data.
        if isinstance(value['included'], str) and type(value['objects']) is int and value['objects'] >= 0:
            return dict(schema='scope_count', objects=[], declaration=deepcopy(value),
                        object_membership='unknown; scope text and count are not object-level inclusion')
        if isinstance(value['objects'], list):
            rows=normalize(value['objects'])['objects']
            return dict(schema='object_rows', objects=rows, declaration=deepcopy(value))
    raise ValueError('Unsupported coverage: supply [{name, included: boolean, reason?}], or legacy {included: scope text, objects: count}; no all-included fallback')
