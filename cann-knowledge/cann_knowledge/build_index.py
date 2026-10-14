"""Explicit build step. Runtime tools never write files or update the index."""
from __future__ import annotations
import argparse, collections, json, re, sqlite3
from pathlib import Path
from urllib.parse import quote
from .common import sha256, tokens

TEXT_EXT = {'.md','.mdx','.txt','.rst','.py','.sh','.bash','.ps1','.cpp','.cc','.c','.h','.hpp','.cuh','.asc','.template','.json','.jsonl','.yaml','.yml','.toml','.ini','.cfg','.cmake','.xml','.csv','.ts','.js','.html','.jinja','.j2','.mako','.in'}
TOPICS = {
 'matmul': r'matmul|mmad|gemm|fullload|pingpong|streamk|swat',
 'pipeline-memory': r'pipeline|buffer|datacopy|resident|memory|mte|bank|preload|cache',
 'sync-scalar': r'sync|scalar|deadlock|barrier|event|flag',
 'profiling-debug': r'profil|msprof|simulat|debug|sanitiz|crash',
 'tiling': r'tiling|tile[-_]?design|tail[-_]?block|core[-_]?shrink',
 'reduction-broadcast': r'reduc|softmax|norm|broadcast|transpose|conversion',
 'architecture': r'npu[-_]?arch|architecture|arch[-_]?guide',
 'precision-tests': r'precision|accuracy|whitebox|st-design|ut-develop|test-develop',
 'ascendc-api': r'ascendc/apis|api-best|api-knowledge',
 'catlass': r'catlass', 'tilelang': r'tilelang', 'triton': r'triton', 'pypto': r'pypto',
 'community-strategies': r'community/ops|knowledge/contrib|evolution|strategy|playbook',
 'engineering': r'build|coding|invoke|repo-|spec-gen|design-paradigm|extension|regbase|blaze',
}
MIRROR = 'https://github.com/Victor287-gif/prompt-vault/blob/cann-knowledge-v1/cann-knowledge/'

def describe(path, root, sources):
    raw = path.read_bytes()
    if b'\x00' in raw: return None
    try: text = raw.decode('utf-8-sig')
    except UnicodeDecodeError: return None
    rel = path.relative_to(root).as_posix()
    source_id, upstream_rel = rel.split('/',2)[1:]
    source = sources[source_id]
    title_match = re.search(r'^#\s+(.+)$',text,re.M)
    title = title_match.group(1).strip()[:200] if title_match else path.stem
    fm = None
    if text.startswith('---\n'):
        end = text.find('\n---',4)
        if end >= 0: fm = text[4:end]
    architectures = sorted(set(re.findall(r'DAV_\d+|Ascend\s*910[A-Za-z0-9_]*|\b910[BC]\b', text, flags=re.I)))
    topics = [key for key,pat in TOPICS.items() if re.search(pat,rel,re.I)]
    if not topics: topics = ['other-operator-knowledge']
    record = {'id':rel[len('third_party/'):], 'path':rel, 'title':title,
      'sha256':sha256(raw), 'bytes':len(raw), 'line_count':len(text.splitlines()),
      'source_id':source_id, 'upstream_path':upstream_rel, 'upstream_commit':source['commit'],
      'source_url':source['url']+'/blob/'+source['commit']+'/'+quote(upstream_rel,safe='/'),
      'mirror_url':MIRROR+quote(rel,safe='/'), 'topics':topics,
      'architectures_mentioned':architectures,
      'architecture_note':'Mentions only; not verified support. Read frontmatter and constraints.',
      'frontmatter_raw':fm, 'status':'community' if ('community/' in rel or '/contrib/' in rel) else 'upstream',
      'license_path':source['license_path']}
    return record, text

def build(root: Path):
    root = root.resolve()
    lock = json.loads((root/'sources.lock.json').read_text(encoding='utf-8'))
    sources = {s['id']:s for s in lock['upstream']}
    index_dir = root/'index'; index_dir.mkdir(exist_ok=True)
    target = index_dir/'knowledge.sqlite3'
    temp = index_dir/'knowledge.sqlite3.building'
    temp.unlink(missing_ok=True)
    db = sqlite3.connect(temp)
    db.execute('PRAGMA journal_mode=OFF')
    db.execute('CREATE TABLE docs (id TEXT PRIMARY KEY, metadata TEXT NOT NULL)')
    db.execute('CREATE VIRTUAL TABLE chunks USING fts5(doc_id UNINDEXED, first_line UNINDEXED, last_line UNINDEXED, terms, tokenize="unicode61")')
    records=[]; skipped=[]; chunks=0
    for path in sorted((root/'third_party').rglob('*')):
        if not path.is_file() or path.is_symlink(): continue
        if path.suffix.lower() not in TEXT_EXT and path.name not in {'LICENSE','NOTICE','COPYRIGHT','CMakeLists.txt'}: continue
        result = describe(path,root,sources)
        if result is None:
            skipped.append(path.relative_to(root).as_posix()); continue
        meta, text = result; records.append(meta)
        db.execute('INSERT INTO docs VALUES (?,?)',(meta['id'],json.dumps(meta,ensure_ascii=False)))
        lines=text.splitlines()
        prefix=tokens(meta['title']+' '+meta['upstream_path'])
        for start in range(0,len(lines),64):
            stop=min(start+80,len(lines))
            indexed=' '.join(prefix*2+tokens('\n'.join(lines[start:stop])))
            if indexed:
                db.execute('INSERT INTO chunks VALUES (?,?,?,?)',(meta['id'],start+1,stop,indexed)); chunks+=1
    db.execute("INSERT INTO chunks(chunks) VALUES('optimize')")
    db.commit(); db.close()
    temp.replace(target)
    with (root/'registry.jsonl').open('w',encoding='utf-8',newline='\n') as f:
        for m in records: f.write(json.dumps(m,ensure_ascii=False)+'\n')
    topics=collections.defaultdict(list)
    for m in records:
        for topic in m['topics']: topics[topic].append(m)
    catalog=root/'catalog'; catalog.mkdir(exist_ok=True)
    intro=['# CANN 算子研究资料索引','',
      '此目录为可通过 GitHub MCP 逐层读取的索引；不要一次读取整个 registry.jsonl。',
      '正文保留上游原文。community/contrib 是社区贡献，不冒充官方硬件保证。',
      '架构标签仅为原文提及，不代表当前芯片支持。固定版本见 ../sources.lock.json。','',
      f'可检索文本：{len(records)} 份；检索片段：{chunks} 个。','', '|主题|文档数|分页入口|','|---|---:|---|']
    pages={}
    for topic,docs in sorted(topics.items()):
        docs.sort(key=lambda m:(0 if Path(m['path']).name in {'SKILL.md','guide.md','index.md'} else 1,m['path']))
        topic_dir=catalog/topic; topic_dir.mkdir(exist_ok=True)
        names=[]
        for start in range(0,len(docs),70):
            name=f'page-{start//70+1:03}.md'; names.append(name)
            body=[f'# {topic}: {start+1}–{min(start+70,len(docs))} / {len(docs)}','',
              '文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。','']
            for m in docs[start:start+70]:
                body.append(f"- **{m['title'].replace(chr(10),' ')}** · {m['line_count']} 行 · {m['status']}\n  `{m['id']}`")
            if start+70<len(docs): body.append(f'\n下一页：page-{start//70+2:03}.md')
            (topic_dir/name).write_text('\n'.join(body)+'\n',encoding='utf-8')
        intro.append(f'|{topic}|{len(docs)}|[{names[0]}]({topic}/{names[0]})（共 {len(names)} 页）|')
        pages[topic]={'documents':len(docs),'pages':len(names),'first':f'catalog/{topic}/{names[0]}'}
    (catalog/'INDEX.md').write_text('\n'.join(intro)+'\n',encoding='utf-8')
    stats={'documents':len(records),'chunks':chunks,'text_bytes':sum(m['bytes'] for m in records),
      'database_bytes':target.stat().st_size,'topics':pages,'skipped_non_utf8':skipped,
      'source_counts':dict(collections.Counter(m['source_id'] for m in records)),
      'script_execution':'None of the upstream scripts were executed.'}
    (root/'build-report.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(stats,ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    build(parser.parse_args().root)
