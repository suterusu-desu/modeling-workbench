"""Local plugin entry point; never relies on the calling task's working directory."""
import sys
import argparse
import os
from pathlib import Path

parser=argparse.ArgumentParser(add_help=False)
parser.add_argument('--workspace',default=os.environ.get('MODELING_WORKSPACE'))
parser.add_argument('--probe',action='store_true')
options,_=parser.parse_known_args()
if not options.workspace:
    raise SystemExit('The local plugin installation must configure --workspace.')
WORKSPACE=Path(options.workspace).resolve()
from modeling_system.mcp_server import main

if __name__=='__main__':
    if options.probe:
        import json
        import anyio
        from modeling_system.service import ModelingService
        from modeling_system.mcp_server import ModelingMCP
        service=ModelingService(WORKSPACE)
        async def probe():
            schemas=await ModelingMCP(service).list_tools()
            return {'runtime':service.runtime_status(),'mcp_tools':[
                {'name':s.name,'input_schema':s.input_schema} for s in schemas],
                'boundary':'Fresh installed launch process/schema; no native/provider calls or operator-context claim'}
        print(json.dumps(anyio.run(probe)))
    else:
        main()
