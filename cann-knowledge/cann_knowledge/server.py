"""Read-only CANN knowledge MCP server (official MCP Python SDK v1)."""
from __future__ import annotations
import argparse
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from .store import KnowledgeStore


def create_server(root: Path, port: int=8765):
    store=KnowledgeStore(root)
    server=FastMCP('cann-operator-knowledge',host='127.0.0.1',port=port,
                   stateless_http=True,json_response=True)
    readonly=ToolAnnotations(readOnlyHint=True,destructiveHint=False,
                             idempotentHint=True,openWorldHint=False)

    @server.tool(name='cann_get_catalog',annotations=readonly)
    def get_catalog() -> dict:
        """List official source snapshots, topics, architecture caveats and coverage."""
        return store.catalog()

    @server.tool(name='cann_search_knowledge',annotations=readonly)
    def search_knowledge(query: str, topic: str='', architecture: str='', limit: int=8) -> dict:
        """Search CANN operator research in Chinese or English. Architecture filters match mentions, not guarantees."""
        return store.search(query,topic,architecture,limit)

    @server.tool(name='cann_read_knowledge',annotations=readonly)
    def read_knowledge(doc_id: str, start_line: int=1, max_lines: int=100, start_column: int=0) -> dict:
        """Read a known document with original lines, provenance and next_read pagination. Max 200 lines / 24K text chars."""
        return store.read(doc_id,start_line,max_lines,start_column)

    @server.tool(name='cann_list_knowledge',annotations=readonly)
    def list_knowledge(prefix: str='', offset: int=0, limit: int=30) -> dict:
        """Browse known document IDs by literal prefix. Use returned next_offset for pagination."""
        return store.list_documents(prefix,offset,limit)

    @server.resource('cann://catalog')
    def catalog_resource() -> str:
        """Versioned CANN knowledge catalog."""
        import json
        return json.dumps(store.catalog(),ensure_ascii=False)

    return server


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--transport',choices=['stdio','streamable-http'],default='stdio')
    parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args()
    if not 1<=args.port<=65535:
        parser.error('port must be 1..65535')
    # Intentionally loopback-only. Remote use requires an authenticated tunnel/gateway.
    create_server(args.root,args.port).run(transport=args.transport)

if __name__=='__main__':
    main()
