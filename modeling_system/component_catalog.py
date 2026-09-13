"""Lazy, revision-bound catalogs of disconnected face components on stored geometry.

Connectivity comes from recorded triangle vertex indices only: shared undirected
triangle edges by default, or explicit vertex touch. Coincident coordinates never
connect distinct indices. Component identities are deterministic within one exact
stored geometry revision and carry no anatomical meaning. The complete member
mapping is retained as a content-addressed NPZ; compact responses expose counts,
metrics, bounded reads and lazy single-component selection. No native access.
"""
import io
import re
import time
import zipfile
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from .bounded_reads import json_chars, validate_window
from .geometry import validate
from .store import atomic_write, canonical, digest

KIND = 'component_catalog'
# Operation names used in returned descriptors; the service wrappers bind them.
BUILD_OPERATION = 'build_component_catalog'
READ_OPERATION = 'read_component_catalog'
SELECT_OPERATION = 'select_component'
LOCATE_OPERATION = 'locate_component'
DEFAULT_MAX_TRIANGLES = 2000000
CEILING_TRIANGLES = 16000000
PREVIEW = 4
ROW_LIMIT = 64
CONNECTIVITY = {
    'shared_edge': 'Two triangles connect when they share one undirected edge of identical vertex indices.',
    'shared_vertex': 'Two triangles connect when they share one vertex index (vertex touch); coarser than shared edges.'}
MEMBER_KINDS = ('triangles', 'vertices', 'polygons')
ORDERS = ('largest', 'label')
_IDENTITY = re.compile(r'^c(0|[1-9][0-9]*)$')
LIMITS = [
    'Connectivity uses recorded triangle vertex indices only; coincident coordinates never connect distinct indices.',
    'Triangulation edges are not native polygon edges; a diagonal cannot be distinguished from a native edge without recorded polygon topology.',
    'Component identities are exact minimum triangle indices within this geometry revision; they carry no anatomical meaning and are never remapped across revisions.',
    'A nonmanifold edge joins every incident face into one component; sheet ownership at that edge remains ambiguous.',
    'Bounds and surface areas are metrics of recorded triangles; closure, volume and appearance are not evaluated.',
    'Historical stored geometry only; no live freshness, native effect or target admission.']


def geometry_hash(arrays):
    """The same fingerprint import_scene records for an object's stored arrays."""
    return digest(canonical({k: digest(np.asarray(v).tobytes()) for k, v in arrays.items()}))


def _budget(max_triangles):
    if max_triangles is None:
        return DEFAULT_MAX_TRIANGLES
    if type(max_triangles) is not int or not 1 <= max_triangles <= CEILING_TRIANGLES:
        raise ValueError('max_triangles must be an integer from 1 to ' + str(CEILING_TRIANGLES))
    return max_triangles


def _mapping(arrays, key, count, nonnegative):
    if key not in arrays:
        return None
    value = np.asarray(arrays[key])
    if value.shape != (count,) or value.dtype.kind not in 'iu' or (nonnegative and value.size and value.min() < 0):
        raise ValueError(key + ' must map every triangle to one ' + ('nonnegative ' if nonnegative else '') + 'integer')
    return value.astype(np.int64)


def _distinct(values, label, total):
    """Distinct mapped values per component and mapped values spanning several components."""
    compact, inverse = np.unique(values, return_inverse=True)
    width = max(len(compact), 1)
    pairs = np.unique(label * width + inverse.ravel())
    per_component = np.bincount(pairs // width, minlength=total)
    spanning = np.bincount(pairs % width, minlength=len(compact))
    return len(compact), per_component.astype(np.int64), int((spanning > 1).sum())


def analyze(arrays, connectivity='shared_edge', max_triangles=None):
    """One sorted grouping pass plus union-find over recorded triangles; no storage."""
    if connectivity not in CONNECTIVITY:
        raise ValueError('Connectivity must be one of: ' + ', '.join(sorted(CONNECTIVITY)))
    budget = _budget(max_triangles)
    validate(arrays)
    co = np.asarray(arrays['co'], dtype=float)
    tri = np.asarray(arrays['tri'])
    count, points = len(tri), len(co)
    if count > budget:
        raise ValueError('Catalog budget exceeded: ' + str(count) + ' triangles > max_triangles ' + str(budget)
                         + '; nothing was truncated. Declare a larger budget (at most ' + str(CEILING_TRIANGLES)
                         + ') or scope the study explicitly.')
    tri64 = tri.astype(np.int64)
    polygon = _mapping(arrays, 'triangle_polygon', count, True)
    recorded = _mapping(arrays, 'triangle_component', count, False)
    repeated = (tri64[:, 0] == tri64[:, 1]) | (tri64[:, 1] == tri64[:, 2]) | (tri64[:, 0] == tri64[:, 2])
    corners = co[tri64]
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    area = 0.5 * np.sqrt(np.einsum('ij,ij->i', cross, cross))
    # Disjoint categories: repeated indices, or distinct indices with exactly zero area.
    zero_area = (area == 0) & ~repeated
    degenerate = repeated | zero_area
    # Undirected edge keys; self edges of repeated-index triangles are dropped.
    pairs = np.stack([tri64[:, [0, 1]], tri64[:, [1, 2]], tri64[:, [2, 0]]], axis=1).reshape(-1, 2)
    low, high = pairs.min(axis=1), pairs.max(axis=1)
    owner = np.repeat(np.arange(count, dtype=np.int64), 3)
    keep = low != high
    keys, owner = (low * max(points, 1) + high)[keep], owner[keep]
    if repeated.any():
        # A triangle with a repeated index lists one edge twice; count it once.
        order = np.lexsort((keys, owner))
        keys, owner = keys[order], owner[order]
        first = np.ones(len(keys), dtype=bool)
        first[1:] = (keys[1:] != keys[:-1]) | (owner[1:] != owner[:-1])
        keys, owner = keys[first], owner[first]
    unique_edges, edge_of = np.unique(keys, return_inverse=True)
    edge_of = edge_of.ravel()
    degree = np.bincount(edge_of, minlength=len(unique_edges)).astype(np.int64)
    edge_order = np.argsort(edge_of, kind='stable')
    if connectivity == 'shared_edge':
        node_sorted, owner_sorted = edge_of[edge_order], owner[edge_order]
    else:
        vertex_owner = np.repeat(np.arange(count, dtype=np.int64), 3)
        vertex_order = np.argsort(tri64.ravel(), kind='stable')
        node_sorted, owner_sorted = tri64.ravel()[vertex_order], vertex_owner[vertex_order]
    same = node_sorted[1:] == node_sorted[:-1]
    rows, cols = owner_sorted[:-1][same], owner_sorted[1:][same]
    if count:
        graph = sparse.coo_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(count, count))
        total, raw = connected_components(graph, directed=False)
    else:
        total, raw = 0, np.zeros(0, dtype=np.int64)
    # Deterministic labels: rank components by their minimum triangle index.
    minimum = np.full(total, count, dtype=np.int64)
    np.minimum.at(minimum, raw, np.arange(count, dtype=np.int64))
    rank = np.empty(total, dtype=np.int64)
    rank[np.argsort(minimum, kind='stable')] = np.arange(total, dtype=np.int64)
    label = rank[raw].astype(np.int64) if count else np.zeros(0, dtype=np.int64)
    minimum = np.sort(minimum)
    triangles = np.bincount(label, minlength=total).astype(np.int64)
    member_order = np.argsort(label, kind='stable').astype(np.int64)
    start = np.zeros(total + 1, dtype=np.int64)
    np.cumsum(triangles, out=start[1:])
    if total:
        bounds_min = np.minimum.reduceat(corners.min(axis=1)[member_order], start[:-1], axis=0)
        bounds_max = np.maximum.reduceat(corners.max(axis=1)[member_order], start[:-1], axis=0)
    else:
        bounds_min = bounds_max = np.zeros((0, 3))
    stride = max(points, 1)
    vertex_pairs = np.unique(np.repeat(label, 3) * stride + tri64.ravel())
    vertices = np.bincount(vertex_pairs // stride, minlength=total).astype(np.int64)
    membership = np.bincount(vertex_pairs % stride, minlength=points)
    referenced = int((membership > 0).sum())
    first_slot = np.searchsorted(edge_of[edge_order], np.arange(len(unique_edges)))
    edge_component = label[owner[edge_order][first_slot]] if len(unique_edges) else np.zeros(0, dtype=np.int64)
    boundary, nonmanifold = degree == 1, degree >= 3
    max_degree = np.zeros(total, dtype=np.int64)
    np.maximum.at(max_degree, edge_component, degree)
    if count:
        _, first_index = np.unique(np.sort(tri64, axis=1), axis=0, return_index=True)
        duplicate = np.ones(count, dtype=bool)
        duplicate[first_index] = False
    else:
        duplicate = np.zeros(0, dtype=bool)
    columns = dict(
        component_label=label, member_order=member_order, member_start=start,
        component_min_triangle=minimum, component_triangles=triangles, component_vertices=vertices,
        component_area=np.bincount(label, weights=area, minlength=total).astype(float),
        component_bounds_min=bounds_min.astype(float), component_bounds_max=bounds_max.astype(float),
        component_boundary_edges=np.bincount(edge_component[boundary], minlength=total).astype(np.int64),
        component_nonmanifold_edges=np.bincount(edge_component[nonmanifold], minlength=total).astype(np.int64),
        component_max_edge_degree=max_degree,
        component_degenerate_triangles=np.bincount(label[degenerate], minlength=total).astype(np.int64),
        component_duplicate_triangles=np.bincount(label[duplicate], minlength=total).astype(np.int64),
        component_order_largest=np.lexsort((minimum, -triangles)).astype(np.int64))
    summary = dict(
        triangles=count, vertices=points, referenced_vertices=referenced, unreferenced_vertices=points - referenced,
        components=total, largest_component_triangles=int(triangles.max()) if total else 0,
        single_triangle_components=int((triangles == 1).sum()),
        unique_edges=int(len(unique_edges)), boundary_edges=int(boundary.sum()), nonmanifold_edges=int(nonmanifold.sum()),
        max_edge_degree=int(degree.max()) if len(degree) else 0,
        edge_degree_histogram={'1': int((degree == 1).sum()), '2': int((degree == 2).sum()),
                               '3': int((degree == 3).sum()), '4+': int((degree >= 4).sum())},
        vertices_in_multiple_components=int((membership > 1).sum()),
        degenerate_triangles=dict(repeated_index=int(repeated.sum()), zero_area=int(zero_area.sum()), total=int(degenerate.sum())),
        duplicate_triangles=int(duplicate.sum()), surface_area=float(area.sum()))
    if polygon is not None:
        polygons, per_component, split = _distinct(polygon, label, total)
        columns['component_polygons'] = per_component
        summary['polygons'] = dict(status='recorded', count=polygons, split_across_components=split,
                                   meaning='Recorded triangle_polygon membership; native polygon-edge adjacency is not recomputed from it')
    else:
        summary['polygons'] = dict(status='absent', meaning='No triangle_polygon mapping is recorded; polygon counts are unavailable')
    if recorded is not None:
        values, per_component, split = _distinct(recorded, label, total)
        columns['component_recorded_components'] = per_component
        summary['recorded_component_semantics'] = dict(status='recorded', values=values, spanning_multiple_catalog_components=split,
                                                       meaning='Recorded triangle_component values are separate semantics; they do not define catalog connectivity')
    else:
        summary['recorded_component_semantics'] = dict(status='absent')
    processing = dict(triangles_processed=count, vertices=points, edge_slots=3 * count, unique_edges=int(len(unique_edges)),
                      adjacency_pairs=int(len(rows)), connectivity=connectivity, truncated=False,
                      budget=dict(max_triangles=budget, ceiling=CEILING_TRIANGLES),
                      algorithm='one sorted grouping pass over edge/vertex keys, chained adjacency and union-find connected components')
    return summary, columns, processing


def _npz_bytes(arrays):
    """Byte-deterministic NPZ so an identical catalog dedupes to one asset."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(name + '.npy', date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            stream = io.BytesIO()
            np.lib.format.write_array(stream, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            archive.writestr(info, stream.getvalue())
    return buffer.getvalue()


def _retain(store, data, suffix='.npz'):
    sha = digest(data)
    dest = store.root / 'assets' / sha[:2] / (sha + suffix)
    if dest.exists():
        if digest(dest.read_bytes()) != sha:
            raise ValueError('Stored evidence was modified: ' + str(dest))
    else:
        atomic_write(dest, data)
    return dict(sha256=sha, path=dest.relative_to(store.root).as_posix(), bytes=len(data),
                source='derived from stored geometry; no external file')


def _row(z, k):
    row = dict(id='c' + str(int(z['component_min_triangle'][k])), label=int(k),
               min_triangle=int(z['component_min_triangle'][k]), triangles=int(z['component_triangles'][k]),
               vertices=int(z['component_vertices'][k]),
               bounds=[z['component_bounds_min'][k].tolist(), z['component_bounds_max'][k].tolist()],
               surface_area=float(z['component_area'][k]),
               boundary_edges=int(z['component_boundary_edges'][k]),
               nonmanifold_edges=int(z['component_nonmanifold_edges'][k]),
               max_edge_degree=int(z['component_max_edge_degree'][k]),
               degenerate_triangles=int(z['component_degenerate_triangles'][k]),
               duplicate_triangles=int(z['component_duplicate_triangles'][k]))
    if 'component_polygons' in z:
        row['polygons'] = int(z['component_polygons'][k])
    if 'component_recorded_components' in z:
        row['recorded_components'] = int(z['component_recorded_components'][k])
    return row


def _order(z, order):
    if order not in ORDERS:
        raise ValueError('order must be largest or label')
    return z['component_order_largest'] if order == 'largest' else np.arange(len(z['component_triangles']))


def _identity(record):
    return dict(catalog_kind=KIND, state=record['state'], source_state_id=record['source_state_id'],
                geometry_revision=record['geometry_revision'], object=record['object'],
                geometry_hash=record['geometry_hash'], connectivity=record['connectivity']['kind'])


def _members(service, record):
    """Load the retained member mapping, verifying content hash and record consistency."""
    path = service.store.resolve_blob(record['members']['asset'])
    with np.load(path, allow_pickle=False) as data:
        z = {k: data[k] for k in data.files}
    if (z['geometry_hash'].item() != record['geometry_hash'] or z['state'].item() != record['state']
            or z['object'].item() != record['object'] or z['connectivity'].item() != record['connectivity']['kind']
            or len(z['component_label']) != record['summary']['triangles']
            or len(z['component_triangles']) != record['summary']['components']
            or len(z['member_start']) != record['summary']['components'] + 1):
        raise ValueError('Catalog member mapping is inconsistent with its catalog record')
    return z


def _component_index(z, component):
    minimum = z['component_min_triangle']
    if isinstance(component, str) and _IDENTITY.match(component):
        value = int(component[1:])
        k = int(np.searchsorted(minimum, value))
        if k < len(minimum) and int(minimum[k]) == value:
            return k
    raise ValueError('Unknown component identity ' + repr(component)
                     + ' for this catalog; identities are c<minimum triangle index> within this exact geometry revision')


def _select_descriptor(catalog, state, component, members='triangles'):
    return dict(operation=SELECT_OPERATION, arguments=dict(catalog=catalog, component=component, expected_state=state,
                                                           members=members, offset=0, limit=256, max_chars=8000))


def _read_descriptor(catalog, order, offset, limit, max_chars):
    return dict(operation=READ_OPERATION, arguments=dict(catalog=catalog, order=order, offset=offset, limit=limit, max_chars=max_chars))


def build(service, state, object_name, connectivity='shared_edge', max_triangles=None):
    """Catalog disconnected face components of one stored object; compact result, retained full mapping."""
    started = time.perf_counter()
    record = service.store.get(state, 'state')
    entry = next((o for o in record['objects'] if o['name'] == object_name), None)
    if entry is None or 'asset' not in entry:
        raise ValueError('Recorded mesh unavailable: ' + object_name)
    arrays = service.wb.arrays(state, object_name)
    actual = geometry_hash(arrays)
    if actual != entry['geometry_hash']:
        raise ValueError('Stored geometry does not match its recorded geometry hash; unverified arrays are not cataloged')
    summary, columns, processing = analyze(arrays, connectivity, max_triangles)
    members = dict(columns, geometry_hash=np.array(actual), state=np.array(state), object=np.array(object_name),
                   connectivity=np.array(connectivity))
    asset = _retain(service.store, _npz_bytes(members))
    preview = [_row(columns, int(k)) for k in columns['component_order_largest'][:PREVIEW]]
    if 'triangle_polygon' in arrays:
        basis = ('Recorded triangulation edges define connectivity; triangle_polygon membership is preserved and reported, '
                 'but native polygon-edge adjacency is not recomputed or claimed.')
    else:
        basis = ('Recorded triangulation edges define connectivity; no triangle-to-polygon mapping is recorded for this object, '
                 'so native polygon connectivity is not established.')
    payload = dict(schema_version=1, state=state, source_state_id=record['source_state_id'],
                   geometry_revision=record['geometry_revision'], object=object_name, geometry_hash=actual,
                   connectivity=dict(kind=connectivity, meaning=CONNECTIVITY[connectivity], basis=basis),
                   summary=summary, preview=preview,
                   members=dict(asset=asset, arrays=sorted(members),
                                meaning='Complete triangle-to-component mapping and per-component metrics; load lazily, never inline'),
                   processing=processing, limits=LIMITS, anatomy='not implied by any component identity or metric',
                   native_ready=False)
    key = service.store.put(KIND, payload)
    largest = preview[0]['id'] if preview else None
    reads = {'components': _read_descriptor(key, 'largest', 0, 20, 8000)}
    for name in ('summary', 'preview', 'members', 'processing', 'limits', 'connectivity'):
        reads[name] = dict(operation='read_record', arguments=dict(record=key, path=[name], limit=64, max_chars=8000))
    return dict(catalog=key, status='completed', state=state, object=object_name, geometry_hash=actual,
                geometry_revision=record['geometry_revision'], source_state_id=record['source_state_id'],
                connectivity=payload['connectivity'], summary=summary, preview=preview, members=payload['members'],
                reads=reads, select=_select_descriptor(key, state, largest) if largest is not None else None,
                locate=dict(operation=LOCATE_OPERATION, arguments=dict(catalog=key, triangle=0)) if preview else None,
                processing=dict(processing, seconds=time.perf_counter() - started), limits=LIMITS,
                anatomy=payload['anatomy'], native_ready=False)


def read(service, catalog, offset=0, limit=20, max_chars=8000, order='largest'):
    """Bounded window of per-component metrics; members stay in the retained mapping."""
    validate_window([], offset, limit, max_chars)
    if limit > ROW_LIMIT:
        raise ValueError('Component windows permit at most ' + str(ROW_LIMIT) + ' rows')
    record = service.store.get(catalog, KIND)
    z = _members(service, record)
    permutation = _order(z, order)
    total = len(permutation)
    if offset > total:
        raise IndexError('Read offset exceeds the component count')
    result = dict(projection='bounded_read', source_revision=catalog, catalog=catalog, identity=_identity(record),
                  order=order, ordering='largest triangle count first, then minimum triangle index' if order == 'largest'
                  else 'ascending component label (minimum triangle index)',
                  total=total, offset=offset, shown=0, deferred=total, items=[], max_chars=max_chars,
                  coverage='Per-component metrics only; member identities require explicit selection',
                  select=None)
    allowance = max_chars - 700
    for index in range(offset, min(total, offset + limit)):
        entry = dict(index=index, value=_row(z, int(permutation[index])))
        if json_chars({**result, 'items': result['items'] + [entry]}) > allowance:
            break
        result['items'].append(entry)
        result['shown'] += 1
    end = offset + result['shown']
    result['deferred'] = total - result['shown']
    result['remaining_after_window'] = total - end
    result['content_complete'] = offset == 0 and end == total
    if end < total:
        result['next'] = _read_descriptor(catalog, order, end, limit, max_chars)
    if total > offset and not result['shown']:
        raise ValueError('Read window cannot make progress within max_chars')
    if result['items']:
        result['select']=_select_descriptor(catalog,record['state'],result['items'][0]['value']['id'])
    if json_chars(result)>max_chars:raise ValueError('Component read exceeds max_chars')
    return result


def locate(service, catalog, triangle):
    """Which catalog component contains one exact recorded triangle index."""
    record = service.store.get(catalog, KIND)
    z = _members(service, record)
    if type(triangle) is not int or not 0 <= triangle < len(z['component_label']):
        raise ValueError('Triangle index outside this catalog geometry')
    row = _row(z, int(z['component_label'][triangle]))
    return dict(catalog=catalog, triangle=triangle, component=row, identity=_identity(record),
                select=_select_descriptor(catalog, record['state'], row['id']))


def select(service, catalog, component, expected_state, expected_geometry_hash=None, members='triangles',
           offset=0, limit=256, max_chars=8000):
    """Materialize one component's exact members after verifying catalog, state and geometry identity."""
    validate_window([], offset, limit, max_chars)
    if members not in MEMBER_KINDS:
        raise ValueError('members must be one of: ' + ', '.join(MEMBER_KINDS))
    record = service.store.get(catalog, KIND)
    if not isinstance(expected_state, str) or not expected_state:
        raise ValueError('Selection requires the expected stored state ID')
    if expected_state != record['state']:
        hint = 'the requested state is unknown to this store'
        try:
            other = service.store.get(expected_state, 'state')
            entry = next((o for o in other['objects'] if o['name'] == record['object']), None)
            hint = ('the requested state records identical object geometry, so a rebuild there yields identical identities'
                    if entry and entry.get('geometry_hash') == record['geometry_hash']
                    else 'the requested state records different or missing geometry for this object')
        except (OSError, ValueError, KeyError):
            pass
        raise ValueError('Stale catalog: it was built on another stored state (' + hint
                         + '). No proximity remap is performed; rebuild the catalog on the requested state.')
    if expected_geometry_hash is not None and expected_geometry_hash != record['geometry_hash']:
        raise ValueError('Requested geometry hash differs from the catalog geometry; rebuild on the requested revision')
    arrays = service.wb.arrays(record['state'], record['object'])
    if geometry_hash(arrays) != record['geometry_hash']:
        raise ValueError('Stored geometry no longer matches the catalog geometry hash; the catalog is stale for this store')
    z = _members(service, record)
    k = _component_index(z, component)
    triangles = z['member_order'][int(z['member_start'][k]):int(z['member_start'][k + 1])]
    if members == 'triangles':
        ids, meaning = triangles, 'Original recorded triangle indices of this component, ascending'
    elif members == 'vertices':
        ids, meaning = np.unique(np.asarray(arrays['tri'])[triangles]), 'Distinct recorded vertex indices referenced by this component'
    else:
        if 'triangle_polygon' not in arrays:
            raise ValueError('No triangle_polygon mapping is recorded for this object; polygon members are unavailable')
        ids, meaning = np.unique(np.asarray(arrays['triangle_polygon'])[triangles]), 'Distinct recorded polygon indices owning this component\'s triangles'
    total = int(len(ids))
    if offset > total:
        raise IndexError('Member offset exceeds the component member count')
    take = min(limit, total - offset)
    # Lean fixed overhead so a 2048-character window still carries members; limits live in the catalog record.
    result = dict(catalog=catalog, component=_row(z, k), identity=_identity(record),
                  verification=dict(expected_state='matches catalog', geometry_hash='recomputed from stored arrays; matches',
                                    members='content hash verified', remap='none'),
                  members=dict(kind=members, meaning=meaning, total=total, offset=offset, shown=0, items=[]))
    while take:
        window = ids[offset:offset + take].tolist()
        candidate = {**result, 'members': {**result['members'], 'shown': take, 'items': window}}
        if json_chars(candidate) <= max_chars - 600:
            result = candidate
            break
        take //= 2
    if total > offset and not take:
        raise ValueError('Member window cannot make progress within max_chars')
    end = offset + result['members']['shown']
    result['members']['remaining_after_window'] = total - end
    result['members']['content_complete'] = offset == 0 and end == total
    if end < total:
        result['members']['next'] = dict(operation=SELECT_OPERATION, arguments=dict(
            catalog=catalog, component=component, expected_state=expected_state, members=members,
            offset=end, limit=limit, max_chars=max_chars))
    if json_chars(result)>max_chars:raise ValueError('Component selection exceeds max_chars')
    return result
