"""Initialize private character inputs without putting them in the tools checkout."""
from pathlib import Path
import argparse,json,re


def initialize(path, character):
    target=Path(path).resolve(); source=Path(__file__).resolve().parent
    from .skill_bundle import verify_dependencies, install_skills
    if verify_dependencies()['status'] != 'verified':
        raise ValueError('Repair the installed skill dependency before initialization')
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
    method=(source/'plugin/skills/modeling-workbench/references/method.md').read_text(encoding='utf-8')
    method=re.sub(r'\]\((?![a-z]+:|#)([^)]+)\)', r'](.agents/skills/modeling-workbench/references/\1)', method)
    files={'modeling-workspace.json':json.dumps(binding,indent=2),
        'PROJECT.md':'# Current project\n\nRecord the selected character reference, current saved checkpoint, scope, outstanding quality question, protected objects, native owner and unresolved evidence. No character is selected by the tools package.\n',
        'DECISIONS.md':'# Decisions\n\nRecord user choices, scope, approvals, rejected directions and which earlier decisions they supersede.\n',
        'LESSONS.md':'# Local experience\n\nRetain exact inputs, interventions, comparisons and outcomes with applicability and counterexamples. Promote reusable methods only after demonstrated reuse and privacy review.\n',
        'METHOD.md':method,
        'OPERATING.md':'# Operating this workspace\n\nRead [Direct owner operation](.agents/skills/modeling-workbench/references/operating-session.md) and [portable setup](.agents/skills/modeling-workbench/references/portable-setup.md). Bind this workspace explicitly in every process.\n',
        'AGENTS.md':'# Character workspace\n\nRead PROJECT.md for current identity, checkpoint, scope, owner and unresolved work. Use [the workbench skill](.agents/skills/modeling-workbench/SKILL.md). The skill bundle is included locally; do not depend on another agent session or machine.\n\nThe owner chooses and executes qualified operations directly, with fresh state, consumed preservation constraints, recoverable effects and actual visual review. There is no inference provider or provider credential dependency for operating Blender. Initialization grants no native, provider or unattended-work authority. For optional external generation, keep credentials external and retain job accounting; and keep character material private.\n',
        'procedures.json':(source/'procedures.json').read_text(encoding='utf-8'),
        'generation-policy.json':json.dumps({'version':1,'authorization':'No generation authorized by initialization','providers':{},
            'tripo':{'mode':'Smart Mesh','model':'P2.0','transport':{'selected':'unconfigured'},
                'authorized_comparisons':{},'authorized_local_reconstructions':{}}},indent=2),
        '.gitignore':'.modeling/\nruntime/\n.env\n.env.*\n*.key\n*.blend\n*.blend1\n*.npz\n*.png\n*.mp4\n'}
    for name,value in files.items(): (target/name).write_text(value+'\n',encoding='utf-8')
    skills = install_skills(target/'.agents/skills')
    return {'workspace':str(target),'character':character,'native_adapter':'unconfigured','files':list(files),'folders':list(folders),'skills':skills}


def main():
    p=argparse.ArgumentParser(); p.add_argument('workspace'); p.add_argument('--character',required=True); a=p.parse_args()
    print(json.dumps(initialize(a.workspace,a.character),indent=2))


if __name__=='__main__':
    main()
