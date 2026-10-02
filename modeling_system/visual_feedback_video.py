"""Pinned motion evidence for the existing correction workflow; no native effects."""
import math
from pathlib import Path
import re
import subprocess
import imageio_ffmpeg
from .ledger import Conflict
from .visual_feedback import dated, rectangle, text


def decode_timestamps(path):
    """Decode a bounded MP4 and retain actual presentation times, including VFR."""
    path = Path(path).resolve()
    if path.suffix.lower() != '.mp4' or not 0 < path.stat().st_size <= 128 * 1024 * 1024:
        raise ValueError('Choose an MP4 of at most 128 MiB')
    with path.open('rb') as stream:
        if stream.read(12)[4:8] != b'ftyp':
            raise ValueError('Expected an ISO base media MP4')
    result = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-hide_banner', '-copyts',
        '-protocol_whitelist', 'file,pipe', '-i', str(path), '-map', '0:v:0',
        '-vf', 'showinfo', '-frames:v', '2401', '-an', '-f', 'null', '-'],
        capture_output=True, text=True, timeout=90)
    # Decoder diagnostics stay private; never return paths or frame contents.
    if result.returncode:
        raise ValueError('MP4 decoding failed; the clip has not been added')
    frames = re.findall(r'\bn:\s*(\d+)\s+pts:\s*[-\d]+\s+pts_time:([-+\d.eE]+).*?\bs:(\d+)x(\d+)', result.stderr)
    if not frames or len(frames) > 2400:
        raise ValueError('Use a fully decodable clip of 1 to 2400 frames')
    size = [int(frames[0][2]), int(frames[0][3])]
    times = [float(row[1]) for row in frames]
    if (size[0] * size[1] > 8294400 or
        any(int(row[0]) != i or [int(row[2]), int(row[3])] != size for i, row in enumerate(frames)) or
        any(not math.isfinite(t) or t < 0 or (i and t <= times[i-1]) for i, t in enumerate(times))):
        raise ValueError('Clip requires stable dimensions and ordered finite presentation timestamps')
    return dict(size=size, frame_times=times, playback_times=[t-times[0] for t in times], frame_count=len(times),
                timing_basis='Decoded source presentation timestamps preserved with copyts; browser playback is offset to first frame; zero-based decoded indices')


def video(feedback, board, expected_revision, video_path, metadata):
    value = feedback._board(board, expected_revision)
    allowed = {'label', 'source', 'captured_at', 'baseline_version', 'result_version',
        'camera', 'display_state', 'lighting', 'framing', 'view_role', 'speed_label',
        'sequence_id', 'frame_ids', 'layout', 'library_identity', 'related_image', 'video_sha256'}
    if not isinstance(metadata, dict) or set(metadata) - allowed:
        raise ValueError('Unsupported video metadata fields')
    for key in ('label', 'source', 'speed_label'):
        text(metadata.get(key), key)
    dated(metadata.get('captured_at'), 'captured_at')
    for key in ('baseline_version', 'result_version', 'sequence_id'):
        if metadata.get(key) is not None:
            text(metadata[key], key)
    for key in ('camera', 'display_state', 'lighting', 'framing'):
        if metadata.get(key) is not None and not isinstance(metadata[key], (str, dict)):
            raise ValueError(key + ' requires a descriptor or null')
    if metadata.get('view_role') not in (None, 'eye_context', 'whole_face', 'detail'):
        raise ValueError('Use eye_context, whole_face, detail or null')
    if metadata.get('layout') not in (None, 'baseline_left_trial_right', 'trial_only', 'baseline_only'):
        raise ValueError('Declare paired baseline LEFT/trial RIGHT, trial_only, baseline_only, or unknown layout')
    identity = metadata.get('library_identity')
    if identity is not None:
        if (not isinstance(identity, dict) or set(identity) != {'library_file_id', 'file_id', 'version', 'file_name'} or
            isinstance(identity['version'], bool) or not isinstance(identity['version'], int) or identity['version'] < 0):
            raise ValueError('Library linkage requires exact item, backing file, version and filename')
        for key in ('library_file_id', 'file_id', 'file_name'):
            text(identity[key], key)
    if metadata.get('related_image'):
        feedback._record(value, 'images', metadata['related_image'])
    path = Path(video_path)
    if not 0 < path.stat().st_size <= 128 * 1024 * 1024:
        raise ValueError('Choose an MP4 of at most 128 MiB')
    asset = feedback.store.blob(path)
    if metadata.get('video_sha256') and asset['sha256'] != metadata['video_sha256']:
        raise ValueError('Video bytes differ from the supplied hash')
    decoded = decode_timestamps(feedback.store.resolve_blob(asset))
    ids = metadata.get('frame_ids')
    if ids is not None:
        if (not metadata.get('sequence_id') or not isinstance(ids, list) or len(ids) != decoded['frame_count'] or
            any(not isinstance(x, str) or not x.strip() for x in ids) or len(set(ids)) != len(ids)):
            raise ValueError('Explicit frame mapping needs one unique moment ID per decoded frame and a sequence_id')
    return feedback._append(value, {'videos': [dict(asset=asset, metadata=metadata, media_type='video/mp4',
        **decoded, authority='Pinned bytes and decoded timing; source/candidate/view/Library metadata are caller assertions, not authentication')]})


def motion_review(feedback, board, expected_revision, comparison, videos):
    value = feedback._board(board, expected_revision)
    review = feedback._record(value, 'comparisons', comparison)
    feedback._scope(value, review['target'], review['plan'])
    if feedback._latest(value, 'comparisons', review['target_id']) != comparison:
        raise Conflict('Reopen the current comparison before linking motion')
    if not isinstance(videos, list) or not 1 <= len(videos) <= 2 or len(set(videos)) != len(videos):
        raise ValueError('Link one clip or two distinct normal/slow clips')
    different = ['image comparison: ' + x for x in review['matching']['mismatched']]
    missing = ['image comparison: ' + x for x in review['matching']['unknown']]
    clips = []
    for key in videos:
        clip = feedback._record(value, 'videos', key)
        clips.append(clip)
        m = clip['metadata']
        pair = next((v for v in review.get('context_views', []) if v['role'] == m.get('view_role')), review)
        baseline = feedback._record(value, 'images', pair['baseline'])
        trial = feedback._record(value, 'images', pair['trial'])
        for field, expected in [('baseline_version', baseline['metadata']['source_version']),
                                ('result_version', trial['metadata']['source_version'])]:
            if not m.get(field): missing.append(key + ': ' + field)
            elif m[field] != expected: different.append(key + ': ' + field)
        for field in ('camera', 'display_state', 'lighting', 'framing'):
            for side, image in [('baseline', baseline), ('trial', trial)]:
                if not m.get(field) or not image['metadata'].get(field): missing.append(key + ': ' + side + ' ' + field)
                elif m[field] != image['metadata'][field]: different.append(key + ': ' + side + ' ' + field)
        if not m.get('view_role'): missing.append(key + ': context extent')
        if not m.get('layout'): missing.append(key + ': baseline/trial layout')
        if not m.get('layout'):
            missing.append(key + ': frame scale/dimensions without a known layout')
        else:
            expected_size = ([baseline['size'][0] * 2, baseline['size'][1]]
                             if m['layout'] == 'baseline_left_trial_right' else
                             baseline['size'] if m['layout'] == 'baseline_only' else trial['size'])
            if clip['size'] != expected_size: different.append(key + ': frame scale/dimensions')
        if not m.get('sequence_id') or not m.get('frame_ids'): missing.append(key + ': source moment mapping')
    mapping = 'not supplied'
    speed_different, speed_unknown = [], []
    if len(clips) == 2:
        left, right = [c['metadata'] for c in clips]
        for field in ('result_version', 'camera', 'display_state', 'lighting', 'framing', 'layout', 'view_role'):
            if not left.get(field) or not right.get(field): speed_unknown.append(field)
            elif left[field] != right[field]: speed_different.append(field)
        if left.get('baseline_version') != right.get('baseline_version'): speed_different.append('baseline_version')
        if clips[0]['size'] != clips[1]['size']: speed_different.append('frame size')
        if not left.get('sequence_id') or not right.get('sequence_id') or not left.get('frame_ids') or not right.get('frame_ids'):
            missing.append('normal/slow moment mapping')
            mapping = 'unknown; timestamps are not interchangeable'
        elif left['sequence_id'] != right['sequence_id']:
            different.append('normal/slow source sequence or ordered moment IDs')
            mapping = 'unmatched; timestamps are not interchangeable'
        else:
            shared = set(left['frame_ids']) & set(right['frame_ids'])
            a = [x for x in left['frame_ids'] if x in shared]
            b = [x for x in right['frame_ids'] if x in shared]
            if not shared or a != b:
                different.append('normal/slow source sequence or ordered moment IDs')
                mapping = 'unmatched; timestamps are not interchangeable'
            else:
                mapping = 'same explicit shared source moment IDs (' + str(len(shared)) + '); each clip keeps its own presentation timestamps'
                absent = len(set(left['frame_ids']) ^ set(right['frame_ids']))
                if absent:
                    missing.append(str(absent) + ' source moments occur in only one speed; those moments are not mapped')
    return feedback._append(value, {'motion_reviews': [dict(target_id=review['target_id'], comparison=comparison,
        target=review['target'], plan=review['plan'], videos=videos, moment_mapping=mapping,
        speed_matching=dict(status='unmatched' if speed_different or mapping.startswith('unmatched') else
            'unknown' if speed_unknown or not mapping.startswith('same explicit') else 'matched declared inputs',
            mapping_allowed=len(clips) == 2 and mapping.startswith('same explicit') and not speed_different and
                bool(clips[0]['metadata'].get('result_version')) and
                clips[0]['metadata'].get('result_version') == clips[1]['metadata'].get('result_version'),
            mismatched=speed_different, unknown=speed_unknown,
            basis='Only explicitly shared clip source moments; independent of static-image region/candidate correspondence'),
        matching=dict(status='unmatched' if different else 'unknown' if missing else 'matched declared inputs',
                      mismatched=different, unknown=missing,
                      basis='Recorded source versions, view descriptors and moment IDs; no native or visual authentication'))]})


def bind_moment(feedback, value, comparison, motion, historical):
    allowed = {'review', 'video', 'frame_index', 'timestamp_seconds', 'region', 'panel'}
    if not isinstance(motion, dict) or set(motion) != allowed:
        raise ValueError('Motion feedback requires review, video, frame_index, timestamp_seconds, region and panel')
    review = feedback._record(value, 'motion_reviews', motion['review'])
    if review['comparison'] != comparison or motion['video'] not in review['videos']:
        raise Conflict('Motion feedback differs from the pinned comparison/clip')
    latest = next((k for k in reversed(value['data'].get('motion_reviews', []))
                   if feedback._record(value, 'motion_reviews', k)['comparison'] == comparison), None)
    if motion['review'] != latest and not historical:
        raise Conflict('Motion review is superseded; reopen or explicitly archive this feedback historically')
    clip = feedback._record(value, 'videos', motion['video'])
    index, timestamp = motion['frame_index'], motion['timestamp_seconds']
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < clip['frame_count']:
        raise ValueError('Frame index is outside the exact decoded clip')
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or timestamp != clip['frame_times'][index]:
        raise Conflict('Timestamp differs from the pinned decoded frame')
    region = rectangle(motion['region'])
    panel, layout = motion['panel'], clip['metadata'].get('layout')
    if panel not in ('baseline', 'trial', 'unresolved'):
        raise ValueError('Choose baseline, trial or unresolved panel')
    if layout == 'baseline_left_trial_right':
        if (panel == 'baseline' and region[0] + region[2] > .5 or panel == 'trial' and region[0] < .5):
            raise ValueError('Marked region crosses or differs from the declared baseline/trial panel')
    elif layout and panel != layout.removesuffix('_only'):
        raise ValueError('Marked panel differs from the declared single-side clip')
    elif not layout and panel != 'unresolved':
        raise ValueError('Unknown clip layout must retain unresolved panel provenance')
    metadata = clip['metadata']
    return dict(**motion, video_sha256=clip['asset']['sha256'], source=metadata['source'],
        baseline_version=metadata.get('baseline_version'), result_version=metadata.get('result_version'),
        sequence_id=metadata.get('sequence_id'),
        moment_id=metadata.get('frame_ids', [])[index] if metadata.get('frame_ids') else None,
        library_identity=metadata.get('library_identity'), related_image=metadata.get('related_image'),
        matching=review['matching'], coordinate_system='Normalized rectangle on the complete pinned video frame; no vertex mask or automatic image correspondence')
