"""Deterministic tools-only distribution, with no access to workspace state."""
from pathlib import Path
import io,json,zipfile
from .store import atomic_write,digest


def export_tools(output_path):
    path=Path(output_path).resolve()
    if path.exists(): raise FileExistsError('Preserve the existing archive; select a new output path')
    root=Path(__file__).parent
    files={}
    selected=json.loads((root/'distribution-files.json').read_text())['files']
    for name in selected:
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve()): raise ValueError('Distribution entry escapes source package')
        files['modeling_system/'+name]=p.read_bytes()
    files['pyproject.toml']=(root/'distribution-pyproject.toml').read_bytes()
    files['README.md']=b'''# Modeling Workbench

Install with Python 3.12+ using `python -m pip install .` in a virtual environment.
Read [getting started](modeling_system/plugin/skills/modeling-workbench/references/getting-started.md)
and [portable setup](modeling_system/plugin/skills/modeling-workbench/references/portable-setup.md).
The complete workbench skill and references are included; no external skill is required.
Character evidence and credentials are not included. Native Blender work requires a
qualified workspace adapter; this tools archive does not qualify an arbitrary rig.
'''
    inventory={name:{'sha256':digest(data),'bytes':len(data)} for name,data in sorted(files.items())}
    manifest={'schema_version':1,'kind':'tools_only','workspace_data_included':False,'files':inventory}
    files['package-manifest.json']=json.dumps(manifest,indent=2).encode()
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in sorted(files.items()):
            entry=zipfile.ZipInfo(name,(2026,1,1,0,0,0)); entry.compress_type=zipfile.ZIP_DEFLATED; z.writestr(entry,data)
    atomic_write(path,buffer.getvalue())
    with zipfile.ZipFile(path) as z:
        for name,row in inventory.items():
            if digest(z.read(name))!=row['sha256']: raise ValueError('Archive verification failed')
    return {'path':str(path),'sha256':digest(path.read_bytes()),'files':len(files),
            'kind':'tools_only','workspace_data_included':False,'native_readiness':'requires private workspace adapter'}
