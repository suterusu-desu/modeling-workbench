"""Initialize private character inputs without putting them in the tools checkout."""
from pathlib import Path
import argparse,json


def initialize(path, character):
    target=Path(path).resolve(); source=Path(__file__).resolve().parent
    if target==source.parent or source.parent in target.parents:
        raise ValueError('Keep the private workspace outside the installed tools checkout')
    if not character.strip(): raise ValueError('A character label is required')
    if target.exists() and any(target.iterdir()): raise FileExistsError('Choose a new empty workspace; existing files are preserved')
    target.mkdir(parents=True,exist_ok=True)
    folders=('references','originals','models','guides','checkpoints','observations','experiments','runtime')
    for folder in folders: (target/folder).mkdir()
    authority=[{'id':name,'path':name,'role':role,'priority':priority} for name,role,priority in (
        ('PROJECT.md','current state and scope',0),('DECISIONS.md','user decisions',0),
        ('METHOD.md','method',1),('LESSONS.md','experience',2),('generation-policy.json','provider policy',0))]
    binding={'schema_version':1,'character':character,'store':'.modeling/state','native_adapter':'unconfigured',
        'authority':authority,'procedures':'procedures.json','evidence':[],
        'material_locations':{name:name for name in folders},'objects':{},'controls':{},'semantic_regions':{},
        'units':'unqualified','frames':[], 'limits':['Native object/control/semantic bindings and scale must be established for this workspace.']}
    files={'modeling-workspace.json':json.dumps(binding,indent=2),
        'PROJECT.md':'# Current project\n\nRecord the selected character reference, current saved checkpoint, scope, outstanding quality question, protected objects, native owner and unresolved evidence. No character is selected by the tools package.\n',
        'DECISIONS.md':'# Decisions\n\nRecord user choices, scope, approvals, rejected directions and which earlier decisions they supersede.\n',
        'LESSONS.md':'# Local experience\n\nRetain exact inputs, interventions, comparisons and outcomes with applicability and counterexamples. Promote reusable methods only after demonstrated reuse and privacy review.\n',
        'METHOD.md':(source/'plugin/skills/modeling-workbench/references/method.md').read_text(),
        'OPERATING.md':(source/'plugin/skills/modeling-workbench/references/operating-session.md').read_text(),
        'procedures.json':(source/'procedures.json').read_text(),
        'generation-policy.json':json.dumps({'version':1,'authorization':'No generation authorized by initialization','providers':{},
            'tripo':{'mode':'Smart Mesh','model':'P2.0','transport':{'selected':'unconfigured'},
                'authorized_comparisons':{},'authorized_local_reconstructions':{}}},indent=2),
        '.gitignore':'.modeling/\n*.blend\n*.blend1\n*.npz\n*.png\n*.mp4\n'}
    for name,value in files.items(): (target/name).write_text(value+'\n',encoding='utf-8')
    return {'workspace':str(target),'character':character,'native_adapter':'unconfigured','files':list(files),'folders':list(folders)}


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('workspace'); p.add_argument('--character',required=True); a=p.parse_args()
    print(json.dumps(initialize(a.workspace,a.character),indent=2))
