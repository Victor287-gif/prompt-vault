"""Command-line read-only access for existing terminal MCP connectors."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from .store import KnowledgeStore

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    subs=parser.add_subparsers(dest='operation',required=True)
    subs.add_parser('catalog')
    search=subs.add_parser('search'); search.add_argument('query')
    search.add_argument('--topic',default=''); search.add_argument('--architecture',default='')
    search.add_argument('--limit',type=int,default=8)
    read=subs.add_parser('read'); read.add_argument('doc_id')
    read.add_argument('--start-line',type=int,default=1); read.add_argument('--max-lines',type=int,default=100)
    read.add_argument('--start-column',type=int,default=0)
    listing=subs.add_parser('list'); listing.add_argument('--prefix',default='')
    listing.add_argument('--offset',type=int,default=0); listing.add_argument('--limit',type=int,default=30)
    args=parser.parse_args(); store=KnowledgeStore(args.root)
    if args.operation=='catalog': result=store.catalog()
    elif args.operation=='search': result=store.search(args.query,args.topic,args.architecture,args.limit)
    elif args.operation=='read': result=store.read(args.doc_id,args.start_line,args.max_lines,args.start_column)
    else: result=store.list_documents(args.prefix,args.offset,args.limit)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
