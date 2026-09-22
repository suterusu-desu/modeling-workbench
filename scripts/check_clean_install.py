"""Build/install the actual wheel and tools archive outside the source checkout."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


root = Path(__file__).resolve().parents[1]
env = {k: v for k, v in os.environ.items()
       if k not in {'PYTHONPATH', 'PYTHONHOME', 'TYPESAFE_API_KEY', 'TYPESAFE_API_KEY_FILE', 'MODELING_WORKSPACE'}}


def run(arguments, cwd, *, capture=False):
    result = subprocess.run([str(a) for a in arguments], cwd=cwd, env=env,
                            capture_output=True, text=True, encoding='utf-8')
    if result.returncode:
        print(result.stdout[-6000:] + result.stderr[-6000:], file=sys.stderr)
        result.check_returncode()
    if capture:
        return result.stdout


with tempfile.TemporaryDirectory(prefix='workbench-portability-') as temporary:
    stage = Path(temporary)
    wheels = stage/'wheels'
    run([sys.executable, '-m', 'pip', 'wheel', root, '--no-deps', '--wheel-dir', wheels], stage)
    wheel, = wheels.glob('modeling_workbench-*.whl')
    expected = json.loads((root/'modeling_system/distribution-files.json').read_text(encoding='utf-8'))['files']
    with zipfile.ZipFile(wheel) as archive:
        for relative in expected:
            assert archive.read('modeling_system/'+relative) == (root/'modeling_system'/relative).read_bytes(), relative
    for mode in ('wheel', 'tools'):
        environment = stage/(mode+'-environment')
        run([sys.executable, '-m', 'venv', environment], stage)
        python = environment/('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        if mode == 'wheel':
            source = wheel
        else:
            source = stage/'tools-source'
            with zipfile.ZipFile(stage/'tools.zip') as archive:
                archive.extractall(source)
            for relative in expected:
                assert (source/'modeling_system'/relative).read_bytes() == (root/'modeling_system'/relative).read_bytes(), relative
        run([python, '-m', 'pip', 'install', source], stage)
        workspace = stage/(mode+' character')
        plugin = stage/(mode+' plugin')/'modeling-workbench'
        run([python, '-I', '-m', 'modeling_system.init_workspace', workspace, '--character', 'Synthetic'], stage)
        run([python, '-I', '-m', 'modeling_system.prepare_plugin', plugin, '--workspace', workspace], stage)
        report = json.loads(run([python, '-I', '-m', 'modeling_system.setup_check',
                                '--workspace', workspace, '--plugin', plugin], stage, capture=True))
        assert report['core_ready'] and report['native']['status'] == 'unconfigured', report
        assert not report['jev']['credentials']['available'], report
        # The public integration must import and execute from the installed
        # distribution, without the private project, inherited auth or sys.path.
        run([python, '-I', '-m', 'unittest', 'modeling_system.test_jev_session',
             'modeling_system.test_cooperative_scopes', 'modeling_system.test_setup_portability',
             'modeling_system.test_isolated_blender', 'modeling_system.test_decisions.PackagedMethodTests'], stage)
        if mode == 'wheel':
            run([python, '-I', '-c',
                 'import sys; from modeling_system.distribution import export_tools; export_tools(sys.argv[1])',
                 stage/'tools.zip'], stage)
        print(mode+': clean installed core, skills, Jev recovery tests and plugin launch passed', flush=True)
