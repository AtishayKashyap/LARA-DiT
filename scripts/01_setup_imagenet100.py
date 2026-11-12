#!/usr/bin/env python3
from __future__ import annotations
import argparse,subprocess,shutil,json,os
from pathlib import Path

def download(root):
    raw=root/'raw'; raw.mkdir(parents=True,exist_ok=True)
    marker=raw/'.download_complete'
    if not marker.exists():
        subprocess.run(['kaggle','datasets','download','-d','ambityga/imagenet100','-p',str(raw)],check=True)
        zips=list(raw.glob('*.zip'))
        if not zips: raise RuntimeError('Kaggle download produced no zip')
        import zipfile
        for z in zips:
            print('extracting',z)
            with zipfile.ZipFile(z) as f: f.extractall(raw)
        marker.touch()
    return raw

def find_raw(raw):
    candidates=[raw,raw/'imagenet100']
    for base in list(raw.rglob('*')):
        if base.is_dir() and (base/'train.X1').is_dir() and (base/'val.X').is_dir(): candidates.append(base)
    for b in candidates:
        if (b/'train.X1').is_dir() and (b/'train.X4').is_dir() and (b/'val.X').is_dir(): return b
    raise RuntimeError(f'Could not find train.X1..train.X4 and val.X under {raw}')

def link_split(dest,shards):
    dest.mkdir(parents=True,exist_ok=True)
    classes=set()
    for shard in shards:
        for c in sorted(p for p in shard.iterdir() if p.is_dir()): classes.add(c.name)
    if len(classes)!=100: raise RuntimeError(f'Expected 100 classes, found {len(classes)}')
    for c in sorted(classes): (dest/c).mkdir(exist_ok=True)
    for shard in shards:
        tag=shard.name.replace('.','_')
        for c in sorted(p for p in shard.iterdir() if p.is_dir()):
            for img in c.iterdir():
                if img.is_file():
                    out=dest/c.name/f'{tag}__{img.name}';
                    if not out.exists(): out.symlink_to(img.resolve())

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='/workspace/data/imagenet100'); args=ap.parse_args(); root=Path(args.root); root.mkdir(parents=True,exist_ok=True)
    raw=download(root); base=find_raw(raw)
    link_split(root/'train',[base/f'train.X{i}' for i in range(1,5)])
    link_split(root/'val',[base/'val.X'])
    print('ImageNet-100 prepared at',root)
    print('Run scripts/02_verify_dataset.py before training.')
if __name__=='__main__': main()
