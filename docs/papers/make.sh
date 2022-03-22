pdflatex -halt-on-error -output-directory=01_manuscript/out 01_manuscript/manuscript.tex
bibtex 01_manuscript/out/manuscript.aux
pdflatex -halt-on-error -output-directory=01_manuscript/out 01_manuscript/manuscript.tex
pdflatex -halt-on-error -output-directory=01_manuscript/out 01_manuscript/manuscript.tex
