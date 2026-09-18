"""Export the two user-facing artifacts from a verified internal delivery manifest."""
from pathlib import Path
import argparse,datetime,hashlib,json,re,shutil
p=argparse.ArgumentParser()
p.add_argument('manifest',type=Path)
p.add_argument('--number',required=True,type=int)
p.add_argument('--date',required=True)
p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]/'成品')
a=p.parse_args()
assert a.number>0
assert datetime.date.fromisoformat(a.date).isoformat()==a.date
m=a.manifest.resolve();r=m.parent;d=json.loads(m.read_text(encoding='utf-8'))
assert d['status']=='complete_for_user_viewing'
title=d['book'];assert title and not re.search(r'[<>:"/\\|?*\x00-\x1f]',title)
h=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
dest=a.root.resolve()/f'{a.number}+{title}+{a.date}'
expected={'成品视频.mp4':d['final_video'],'封面图.png':d['cover']}
for name,src in expected.items():
 source=(r/src).resolve();assert source.is_relative_to(r) and source.is_file()
 assert h(source)==d['artifacts'][src]['sha256']
if dest.exists():
 assert set(x.name for x in dest.iterdir())<=set(expected),'unexpected files in delivery directory'
 for name,src in expected.items():
  if (dest/name).exists():assert h(dest/name)==h(r/src),'existing different export; do not overwrite'
dest.mkdir(parents=True,exist_ok=True)
for name,src in expected.items():
 if not (dest/name).exists():shutil.copy2(r/src,dest/name)
assert set(x.name for x in dest.iterdir())==set(expected)
assert all((dest/n).is_file() and h(dest/n)==h(r/s) for n,s in expected.items())
record=dict(folder=str(dest),number=a.number,title=title,generation_date=a.date,files={n:dict(sha256=h(dest/n),bytes=(dest/n).stat().st_size) for n in expected})
(r/'export-record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(record,ensure_ascii=False))
