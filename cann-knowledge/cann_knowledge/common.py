"""Shared tokenization and bounded, offline path handling."""
from __future__ import annotations
import hashlib, re, unicodedata
from pathlib import Path

ALIASES = {
    '矩阵乘': ['matmul','mmad'], '批矩阵乘': ['batch','matmul'],
    '双缓冲': ['double','buffer','pingpong'], '流水': ['pipeline'],
    '常驻': ['resident','fullload'], '归约': ['reduce','reduction'],
    '标量': ['scalar'], '同步': ['synchronization','sync'],
    '搬运': ['datacopy','mte'], '布局': ['layout','nd2nz'],
    '精度': ['precision'], '性能分析': ['profiling','msprof'],
    '分块': ['tiling','tile'], '核间': ['cross','core'],
    '死锁': ['deadlock'], '缓存': ['cache'], '转置': ['transpose'],
}

def tokens(text: str, expand: bool = False) -> list[str]:
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text)
    if expand:
        text += ' ' + ' '.join(term for key, terms in ALIASES.items() if key in text for term in terms)
    text = text.lower()
    out = re.findall(r'[a-z][a-z0-9]*', text)
    for phrase in re.findall(r'[\u3400-\u9fff]+', text):
        if len(phrase) == 1:
            out.append(phrase)
        else:
            out.extend(phrase[i:i+2] for i in range(len(phrase)-1))
            out.extend(phrase[i:i+3] for i in range(len(phrase)-2))
    return out

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def confined(root: Path, relative: str) -> Path:
    if '\\' in relative or '\x00' in relative or ':' in relative:
        raise ValueError('Invalid relative path')
    rel = Path(relative)
    if rel.is_absolute() or any(p in {'.','..'} for p in rel.parts):
        raise ValueError('Path must remain inside the knowledge root')
    current = root
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('Symlinks are not served')
    result = current.resolve()
    result.relative_to(root.resolve())
    return result
