@echo off
setlocal
cd /d "%~dp0"
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
if errorlevel 1 exit /b 1
bibtex main
if errorlevel 1 exit /b 1
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
if errorlevel 1 exit /b 1
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
if errorlevel 1 exit /b 1
echo Built %CD%\main.pdf
