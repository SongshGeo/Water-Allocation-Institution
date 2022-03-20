pdflatex -halt-on-error -output-directory=01_manuscript/out manuscript.tex
biblatex -halt-on-error -output-directory=out 01_manuscript/outmanuscript.aux
pdflatex -halt-on-error -output-directory=01_manuscript/out manuscript.tex
pdflatex -halt-on-error -output-directory=01_manuscript/out manuscript.tex