FILE=paper

build: clean
	pdflatex ${FILE}
	bibtex ${FILE}
	pdflatex ${FILE}
	pdflatex ${FILE}

clean:
	rm -rf *.log *.aux *.bbl *.blg *.dvi

