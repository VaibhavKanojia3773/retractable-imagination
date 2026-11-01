"""Compile the official NeurIPS LaTeX project and synchronize distribution copies."""
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parent
paper=ROOT/'paper_latex'
commands=[['pdflatex','-interaction=nonstopmode','-halt-on-error','-file-line-error','main.tex'],
          ['bibtex','main'],
          ['pdflatex','-interaction=nonstopmode','-halt-on-error','-file-line-error','main.tex'],
          ['pdflatex','-interaction=nonstopmode','-halt-on-error','-file-line-error','main.tex']]
for command in commands:
    subprocess.run(command,cwd=paper,check=True)
for destination in (ROOT/'paper'/'paper.pdf',ROOT/'web'/'paper.pdf',ROOT/'web'/'paper'/'paper.pdf'):
    destination.parent.mkdir(exist_ok=True,parents=True)
    shutil.copy2(paper/'main.pdf',destination)
print('Built:',paper/'main.pdf')
