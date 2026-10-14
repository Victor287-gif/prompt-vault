"""CPU-only retrieval and MCP transport tests; never run CANN/NPU workloads."""
from __future__ import annotations
import asyncio, hashlib, json, socket, subprocess, sys, tempfile, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cann_knowledge.store import KnowledgeStore


def expect_error(fn, *args):
    try: fn(*args)
    except (ValueError,KeyError): return
    raise AssertionError('Expected bounded-input rejection')


def test_store():
    store=KnowledgeStore(ROOT); checks=[]
    before=hashlib.sha256(store.db_path.read_bytes()).hexdigest()
    assert store.catalog()['documents']>100
    for query,topic in [('matmul pingpong','matmul'),('标量 scalar','sync-scalar'),
                        ('同步 SetFlag WaitFlag','sync-scalar'),('msprof','profiling-debug'),
                        ('L1 常驻','pipeline-memory'),('归约 reduce','reduction-broadcast')]:
        result=store.search(query,topic=topic,limit=3)
        assert result['results'], (query,topic)
        doc=result['results'][0]
        content=store.read(doc['id'],max_lines=12)
        assert content['document']['sha256']==doc['sha256']
        assert len(content['lines'])<=12
        assert content['document']['upstream_commit']
        checks.append({'query':query,'topic':topic,'first_result':doc['id'],'pass':True})
    listing=store.list_documents(limit=2)
    assert len(listing['documents'])==2 and listing['next_offset']==2
    assert not store.search('matmul',architecture='NOT_A_REAL_ARCH')['results']
    for bad in ['../README.md','cannbot-skills/../../README.md','C:/Windows/test','x\\y']:
        expect_error(store.read,bad)
    expect_error(store.search,'')
    expect_error(store.search,'x'*513)
    expect_error(store.search,'matmul','','',21)
    expect_error(store.list_documents,'',-1,2)
    expect_error(store.read,listing['documents'][0]['id'],0)
    after=hashlib.sha256(store.db_path.read_bytes()).hexdigest()
    assert before==after, 'Runtime changed the index'
    checks.extend([{'test':'input_bounds_and_path_confinement','pass':True},
                   {'test':'runtime_index_unchanged','pass':True}])
    return checks

async def session_checks(read,write):
    from mcp import ClientSession
    async with ClientSession(read,write) as session:
        await session.initialize()
        tools=await session.list_tools()
        names={tool.name for tool in tools.tools}
        expected={'cann_get_catalog','cann_search_knowledge','cann_read_knowledge','cann_list_knowledge'}
        assert names==expected, names
        for tool in tools.tools:
            assert tool.annotations.readOnlyHint is True
            assert tool.annotations.destructiveHint is False
        result=await session.call_tool('cann_get_catalog',{})
        assert not result.isError
        result=await session.call_tool('cann_search_knowledge',{'query':'matmul pingpong','topic':'matmul','limit':2})
        assert not result.isError
        return sorted(names)

async def test_stdio():
    from mcp import StdioServerParameters
    from mcp.client.stdio import stdio_client
    params=StdioServerParameters(command=sys.executable,args=['-m','cann_knowledge.server','--root',str(ROOT)],cwd=str(ROOT))
    async with stdio_client(params) as (read,write):
        return await session_checks(read,write)

async def test_http(port):
    from mcp.client.streamable_http import streamable_http_client
    async with streamable_http_client(f'http://127.0.0.1:{port}/mcp') as (read,write,_):
        return await session_checks(read,write)

def main():
    report={'cpu_only':True,'npu_experiments':False,'store':test_store()}
    report['stdio_tools']=asyncio.run(asyncio.wait_for(test_stdio(),timeout=90))
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
    with tempfile.TemporaryFile(mode='w+b') as log:
        proc=subprocess.Popen([sys.executable,'-m','cann_knowledge.server','--root',str(ROOT),'--transport','streamable-http','--port',str(port)],cwd=ROOT,stdout=log,stderr=log)
        try:
            ready=False
            for _ in range(100):
                if proc.poll() is not None: break
                try:
                    with socket.create_connection(('127.0.0.1',port),timeout=0.2): ready=True; break
                except OSError: time.sleep(0.1)
            if not ready:
                log.seek(0); raise RuntimeError(log.read().decode('utf-8',errors='replace')[-3000:])
            report['http_tools']=asyncio.run(asyncio.wait_for(test_http(port),timeout=90))
        finally:
            proc.terminate()
            try: proc.wait(timeout=8)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait(timeout=5)
    report['all_passed']=True
    (ROOT/'test-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
