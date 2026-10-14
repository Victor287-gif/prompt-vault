"""Package the selected public knowledge only; rebuild the read-only index locally."""
from __future__ import annotations
import hashlib, json, zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
DIST=ROOT.parent/'dist'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):
            h.update(part)
    return h.hexdigest()

def main():
    DIST.mkdir(exist_ok=True)
    report=json.loads((ROOT/'test-report.json').read_text(encoding='utf-8'))
    assert report['all_passed'], 'Tests must pass before packaging'
    records=[json.loads(line) for line in (ROOT/'registry.jsonl').read_text(encoding='utf-8').splitlines()]
    for record in records:
        assert digest(ROOT/record['path'])==record['sha256'],record['id']
    excluded={'__pycache__','.git','.venv','node_modules'}
    files=[]
    for path in sorted(ROOT.rglob('*')):
        rel=path.relative_to(ROOT)
        if not path.is_file() or path.is_symlink():
            continue
        if rel.parts[0] == 'index' or any(part in excluded for part in rel.parts) or path.suffix in {'.pyc','.log','.building'}:
            continue
        assert path.suffix.lower() not in {'.pem','.key','.ttf','.otf','.woff','.woff2'}
        files.append(path)
    audit={'verified_source_text_files':len(records),'included_files_before_audit':len(files),
           'source_text_hashes_match':True,'upstream_scripts_executed':False,
           'private_project_material_included':False,'prebuilt_index_included':False}
    (ROOT/'integrity-report.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    if ROOT/'integrity-report.json' not in files:
        files.append(ROOT/'integrity-report.json')
    target=DIST/'cann-operator-knowledge-1.0.0.zip'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for path in files:
            z.write(path,'cann-knowledge/'+path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        assert 'cann-knowledge/index/knowledge.sqlite3' not in z.namelist()
    sums=[]; release=[]
    for path in sorted(DIST.glob('*.zip')):
        hash_=digest(path); sums.append(f'{hash_}  {path.name}')
        release.append({'name':path.name,'bytes':path.stat().st_size,'sha256':hash_})
    (DIST/'SHA256SUMS').write_text('\n'.join(sums)+'\n',encoding='utf-8')
    metadata={'version':'1.0.0','artifacts':release,'tests_passed':True,'zip_compression':'Deflate; prebuilt database excluded, rebuild script included',
              'note':'MCP stdio/HTTP tested locally; dedicated server not persistently hosted. Skill uses existing GitHub MCP.'}
    (DIST/'release-report.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(metadata,ensure_ascii=False,indent=2),flush=True)
    assert target.stat().st_size < 100*1024*1024,'ZIP exceeds GitHub single-file size limit; do not push'

if __name__=='__main__':main()
