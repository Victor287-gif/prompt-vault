"""Extract only tracked operator-research files; never read the user's projects."""
from __future__ import annotations
import argparse, collections, hashlib, json, shutil, subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = Path.cwd() / 'cann-knowledge'
CONFIG = {
    'cannbot-skills': {
        'url': 'https://gitcode.com/cann/cannbot-skills',
        'prefixes': ['ops/', 'community/ops/', 'tools/'],
        'extra_prefixes': ['infra/repo-build-guide/', 'infra/repo-coding-rules/',
                           'infra/repo-knowledge/', 'infra/repo-op-templates/',
                           'infra/repo-test-develop/'],
    },
    'cannbot-knowledge': {
        'url': 'https://gitcode.com/cann/cannbot-knowledge',
        'prefixes': ['knowledge/ops/', 'knowledge/common/', 'knowledge/contrib/',
                     'governance/', 'recipes/'],
        'extra_prefixes': ['.agents/skills/knowledge-query/',
                           '.agents/skills/ascendc-api-knowledge-query/'],
    },
}
ROOT_FILES = {'LICENSE', 'NOTICE', 'COPYRIGHT', 'README.md'}
TEXT_EXT = {'.md','.mdx','.txt','.rst','.py','.sh','.bash','.ps1','.cpp','.cc','.c',
            '.h','.hpp','.cuh','.asc','.template','.json','.jsonl','.yaml','.yml',
            '.toml','.ini','.cfg','.cmake','.xml','.csv','.ts','.js','.html','.svg',
            '.jinja','.j2','.mako','.in'}
DENY_EXT = {'.pem','.key','.pfx','.p12','.ttf','.otf','.woff','.woff2','.exe','.dll',
            '.so','.pyd','.bin','.o','.a','.lib','.whl','.zip','.tar','.gz','.7z','.pt',
            '.pth','.npy','.npz','.pdf','.docx','.xlsx','.pptx'}
DENY_PARTS = {'__pycache__','node_modules','.venv','.env','.git','evals','tests','test'}

def git(repo, *args):
    return subprocess.check_output(['git','-C',str(repo),*args], encoding='utf-8').strip()

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    counts, exclusions, locks = {}, [], []
    for name, cfg in CONFIG.items():
        repo = ROOT / name
        commit = git(repo, 'rev-parse','HEAD')
        entries = subprocess.check_output(['git','-C',str(repo),'ls-files','--stage','-z']).decode('utf-8').split('\0')
        counts[name] = {'files':0, 'bytes':0}
        for entry in entries:
            if not entry:
                continue
            info, rel = entry.split('\t',1)
            mode = info.split()[0]
            selected = rel in ROOT_FILES or any(rel.startswith(p) for p in cfg['prefixes'] + cfg['extra_prefixes'])
            if not selected:
                continue
            src = repo / rel
            reason = None
            if mode == '120000' or src.is_symlink(): reason = 'symlink'
            elif any(p in DENY_PARTS for p in Path(rel).parts): reason = 'tests_or_runtime_directory'
            elif Path(rel).suffix.lower() in DENY_EXT: reason = 'binary_archive_font_or_sensitive_extension'
            elif src.name.startswith('.env') or src.name.lower() in {'credentials','id_rsa','id_ed25519'}: reason = 'sensitive_filename'
            elif src.stat().st_size > 4*1024*1024: reason = 'file_above_4_mib'
            if reason:
                exclusions.append({'source':name,'path':rel,'reason':reason})
                continue
            dest = OUT / 'third_party' / name / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            counts[name]['files'] += 1
            counts[name]['bytes'] += dest.stat().st_size
        locks.append({'id':name,'url':cfg['url'],'commit':commit,
            'commit_time':git(repo,'show','-s','--format=%cI','HEAD'),
            'license':'CANN Open Software License Agreement 2.0',
            'license_path':f'third_party/{name}/LICENSE',
            'selected_prefixes':cfg['prefixes'] + cfg['extra_prefixes'],
            'note':'Official repository snapshot; community/contrib retains its community status.'})
    data = {'format_version':1, 'built_at_utc':datetime.now(timezone.utc).isoformat(),
        'purpose':'Huawei Ascend/CANN operator research only', 'upstream':locks,
        'counts':counts,'excluded_count':len(exclusions),
        'scope':'No private CANN competition source, OJ records, credentials, or local DSH modifications.'}
    (OUT/'sources.lock.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'excluded-files.json').write_text(json.dumps(exclusions,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'extraction_policy.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(data,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    ROOT = args.source_root.resolve()
    OUT = args.output.resolve()
    main()

