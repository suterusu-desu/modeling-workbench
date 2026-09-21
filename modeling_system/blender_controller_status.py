"""Display-only Blender overlay. Load standalone if Blender lacks core dependencies."""
import json
from pathlib import Path
import time

KEY = 'modeling_workbench_controller_status'


def status_lines(status, *, now=None, stale_after=30):
    now = time.time() if now is None else now
    mode = status.get('status', 'unknown')
    age = now - status.get('updated_at', 0)
    if age > stale_after and mode in ('running', 'selecting', 'waiting_plan', 'waiting_owner'):
        mode = 'disconnected / stale (last: ' + mode + ')'
    lines = ['Workbench: ' + mode,
             'Objective: ' + str(status.get('objective') or 'not supplied'),
             'Stage: ' + str(status.get('stage') or 'not supplied'),
             'Action: ' + str(status.get('action') or 'none')]
    if not status.get('action') and status.get('last_action'):
        lines.append('Last action: ' + str(status['last_action']))
    progress = status.get('progress')
    if progress:
        lines.append('Progress: ' + json.dumps(progress, ensure_ascii=False, sort_keys=True))
    timings = status.get('timings_ms')
    if timings:
        lines.append('Time ms: ' + ', '.join(str(k) + '=' + str(v) for k, v in timings.items()))
    budget = status.get('inference_budget')
    lines.append('Inference budget: ' + ('unknown' if budget is None else json.dumps(budget, sort_keys=True)))
    if status.get('planner_pending'):
        lines.append('Planner: update pending')
    if status.get('reason'):
        lines.append('Reason: ' + str(status['reason']))
    return lines


def uninstall():
    import bpy
    previous = bpy.app.driver_namespace.pop(KEY, None)
    if previous is None:
        return
    bpy.types.SpaceView3D.draw_handler_remove(previous['handle'], 'WINDOW')
    if bpy.app.timers.is_registered(previous['timer']):
        bpy.app.timers.unregister(previous['timer'])
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.header_text_set(None)
                area.tag_redraw()


def install(status_path, *, stale_after=30, interval=.5):
    """Owner installs explicitly; timer reads a file and redraws, never executes."""
    import bpy
    import blf
    if stale_after <= 0 or interval <= 0:
        raise ValueError('Positive display refresh and staleness intervals required')
    uninstall()
    path = Path(status_path).resolve()
    state = {'path': str(path), 'lines': ['Workbench: connecting']}

    def draw():
        region = bpy.context.region
        if region is None or bpy.app.driver_namespace.get(KEY) is not state:
            return
        font = 0
        blf.size(font, 14)
        blf.color(font, .9, .95, 1., 1.)
        blf.enable(font, blf.SHADOW)
        blf.shadow(font, 5, 0., 0., 0., 1.)
        for index, text in enumerate(state['lines']):
            blf.position(font, 24, region.height - 70 - 21 * index, 0)
            # The full text remains in status.json; avoid covering the entire view.
            blf.draw(font, text[:140])
        blf.disable(font, blf.SHADOW)

    def refresh():
        if bpy.app.driver_namespace.get(KEY) is not state:
            bpy.types.SpaceView3D.draw_handler_remove(state['handle'], 'WINDOW')
            return None
        try:
            status = json.loads(path.read_text(encoding='utf-8-sig'))
            state['lines'] = status_lines(status, stale_after=stale_after)
        except Exception as exc:
            state['lines'] = ['Workbench: status unavailable', type(exc).__name__]
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    area.header_text_set(' | '.join(state['lines'][::3])[:220])
                    area.tag_redraw()
        return interval

    state['handle'] = bpy.types.SpaceView3D.draw_handler_add(draw, (), 'WINDOW', 'POST_PIXEL')
    state['timer'] = refresh
    bpy.app.driver_namespace[KEY] = state
    bpy.app.timers.register(refresh, first_interval=0., persistent=True)
    return {'status': 'display_installed', 'path': str(path), 'native_dispatch': False,
            'geometry_changed': False, 'stale_after_seconds': stale_after}
