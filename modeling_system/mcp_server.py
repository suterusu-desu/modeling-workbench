"""Discoverable MCP transport over the same domain service used by Python and CLI."""
import argparse
import functools
import inspect
import json
from pathlib import Path
from typing import Any
import anyio
from mcp.server import MCPServer
from mcp.types import Tool, ToolAnnotations, CallToolResult, TextContent
from .service import ModelingService, READ_OPERATIONS
from .native_bridge import operation_schemas


NATIVE_DESCRIPTIONS={
    'inspect_live':'Inspect live Blender file, dirty state, owner, controls and expected-state fingerprint.',
    'bootstrap':'Bind this verified Blender file to the modeling adapter and acquire its editing owner.',
    'owner_release':'Release editing ownership only against the expected live state.',
    'open_checkpoint':'Open a hash-verified native checkpoint after preserving the current session for recovery.',
    'inspect_feature':'Inspect or refresh actual feature, binding validity and registered guide/depth dependencies.',
    'prepare_state':'Export complete recorded geometry, references and coverage from the expected native state.',
    'capture_view':'Capture the actual selected viewport perspective, zoom and optional crop with matching state, depth/sections and wider context.',
    'set_controls':'Set declared native pose controls and matching guide, with refreshed workbench evidence.',
    'set_display':'Choose registered guide/depth/section/material display or exact viewport conditions.',
    'apply_proposal':'Checkpoint and apply a constrained shape-key or recorded advanced script proposal; requires fresh owner/state.',
    'rollback_trial':'Restore a trial and external registry only when it cannot overwrite a later live edit.',
    'integrate_guide':'Integrate a reviewed registered guide with target registry, fitting, display and semantic provenance.',
    'save_checkpoint':'Save a recoverable native checkpoint with exact file evidence.',
    'export_native':'Export selected native objects as BLEND, NPZ, GLB, OBJ or FBX.',
    'prepare_reopen':'Prepare or dispatch isolated-profile independent native checkpoint verification.',
    'capture_motion':'Capture a finite action with locked close-up perspective, context frames, controls and per-frame geometry.',
}


class ModelingMCP(MCPServer):
    def __init__(self, service):
        self.domain=service
        version=json.loads((Path(__file__).parent/'plugin'/'.codex-plugin'/'plugin.json').read_text(encoding='utf-8'))['version']
        super().__init__('modeling-workbench',title='Modeling Workbench',version=version,
                         description='Numerical scene reasoning, exact-view reference loops, recoverable Blender trials and motion evidence.')
        for name in service.operations():
            fn=getattr(service,name)
            def bound(operation, method):
                @functools.wraps(method)
                async def invoke(**kwargs):
                    return await anyio.to_thread.run_sync(lambda:service.execute(operation,kwargs))
                invoke.__signature__=inspect.signature(method).replace(return_annotation=dict[str,Any])
                invoke.__annotations__={**method.__annotations__,'return':dict[str,Any]}
                return invoke
            self.add_tool(bound(name,fn),name=name,description=fn.__doc__,structured_output=True,
                          annotations=ToolAnnotations(read_only_hint=name in READ_OPERATIONS,
                                                      destructive_hint=False,open_world_hint=False))
        @self.resource('modeling://capabilities')
        def capabilities_resource() -> str:
            return json.dumps(service.capabilities(),allow_nan=False)
        @self.resource('modeling://records/{record}')
        def record_resource(record: str) -> str:
            return json.dumps(service.store.get(record),allow_nan=False)
        @self.resource('modeling://workflows/{handle}')
        def workflow_resource(handle: str) -> str:
            return json.dumps(service.ledger.read(handle),allow_nan=False)

    async def list_tools(self):
        result=await super().list_tools()
        for name,schema in operation_schemas().items():
            schema['properties']['owner']={'type':'string','minLength':1,'description':'Stable editing owner; required for native changes.'}
            if name!='inspect_live':schema['required']=[*schema.get('required',[]),'owner']
            result.append(Tool(name='native_'+name,description=NATIVE_DESCRIPTIONS[name],input_schema=schema,
                               annotations=ToolAnnotations(read_only_hint=name=='inspect_live',destructive_hint=False,open_world_hint=False)))
        return result

    async def call_tool(self,name,arguments,context=None):
        if name.startswith('native_'):
            value=await anyio.to_thread.run_sync(lambda:self.domain.execute(name,arguments))
            return CallToolResult(content=[TextContent(type='text',text=json.dumps(value,allow_nan=False))],
                                  structured_content=value,is_error=value['status'] in ('failed','needs attention','conflicting'))
        result=await super().call_tool(name,arguments,context)
        content=getattr(result,'structured_content',None)
        if isinstance(content,dict) and content.get('status') in ('failed','needs attention','conflicting'):
            return result.model_copy(update={'is_error':True})
        return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace');p.add_argument('--store');a=p.parse_args()
    ModelingMCP(ModelingService(a.workspace,a.store)).run()


if __name__=='__main__':main()
