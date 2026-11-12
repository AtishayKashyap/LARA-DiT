from pathlib import Path
import argparse
from torchvision.datasets import ImageFolder
ap=argparse.ArgumentParser(); ap.add_argument('--root',default='/workspace/data/imagenet100'); a=ap.parse_args(); r=Path(a.root)
tr=ImageFolder(r/'train'); va=ImageFolder(r/'val')
print('train:',len(tr),'images',len(tr.classes),'classes'); print('val:',len(va),'images',len(va.classes),'classes')
if len(tr)!=130000 or len(va)!=5000 or len(tr.classes)!=100 or len(va.classes)!=100 or tr.class_to_idx!=va.class_to_idx: raise SystemExit('DATASET VERIFICATION FAILED')
print('DATASET VERIFICATION PASSED')
