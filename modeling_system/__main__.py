"""JSON-file CLI transport; the same service implements MCP and Python calls."""
import argparse,json
from pathlib import Path
from .service import ModelingService

parser=argparse.ArgumentParser(description='Question-driven modeling system')
parser.add_argument('--store',help='Optional explicit evidence store; default comes from the bound workspace')
parser.add_argument('--workspace')
parser.add_argument('operation')
parser.add_argument('--input',help='JSON arguments file; preserves exact paths and parameters')
args=parser.parse_args()
try:
    payload=json.loads(Path(args.input).read_text(encoding='utf-8')) if args.input else {}
    aliases={'inspect':'inspect_situation','import-scene':'import_scene','query':'query_geometry'}
    if args.operation=='query' and 'parameters' not in payload:
        payload['parameters']={k:payload.pop(k) for k in list(payload) if k not in ('state','object_name','query','selection')}
    result=ModelingService(args.workspace,args.store).execute(aliases.get(args.operation,args.operation),payload)
    print(json.dumps(result,indent=2,allow_nan=False))
    if result.get('status') in ('failed','needs attention','conflicting'):raise SystemExit(1)
except (ValueError,KeyError,RuntimeError,OSError) as error:
    print(json.dumps({'status':'failed','reason':str(error),'mutation':'No Blender geometry operation was performed'}))
    raise SystemExit(1)
