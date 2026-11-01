"""Build a double-click HTML and a distributable archive without third-party tools."""
from pathlib import Path
import json
import re
import shutil
import zipfile
from urllib.parse import quote

ROOT=Path(__file__).resolve().parent
web=ROOT/'web'
html=(web/'index.html').read_text(encoding='utf-8')
html=html.replace('<link rel="stylesheet" href="style.css">','<style>'+(web/'style.css').read_text(encoding='utf-8')+'</style>')
for name in ('demo-data.js','study-data.js','core.js','app.js'):
    html=html.replace(f'<script src="{name}"></script>','<script>'+(web/name).read_text(encoding='utf-8')+'</script>')
html=html.replace('href="paper.pdf"','href="paper/paper.pdf"')
html=html.replace('href="evaluation.json"','href="data:application/json;charset=utf-8,'+quote((ROOT/'results'/'evaluation.json').read_text())+'"')
(ROOT/'offline_demo.html').write_text(html,encoding='utf-8')
shutil.copy2(ROOT/'paper_latex'/'main.pdf',ROOT/'paper'/'paper.pdf')
shutil.copy2(ROOT/'paper_latex'/'main.pdf',web/'paper.pdf')
(web/'paper').mkdir(exist_ok=True)
shutil.copy2(ROOT/'paper'/'paper.pdf',web/'paper'/'paper.pdf')
(ROOT/'dist').mkdir(exist_ok=True)
dest=ROOT/'dist'/'retractable_imagination_artifact.zip'
with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
    for p in ROOT.rglob('*'):
        excluded=any(part in ('__pycache__','.git','dist','external_eval') for part in p.relative_to(ROOT).parts)
        clutter=p.suffix in ('.pyc','.aux','.bbl','.blg','.log','.out') or p.name.startswith(('preview-','final_'))
        if p.is_file() and not excluded and not clutter:
            z.write(p,Path('retractable')/p.relative_to(ROOT))
print('Offline demo:',ROOT/'offline_demo.html')
print('Archive:',dest)
