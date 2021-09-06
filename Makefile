STUDENTID = SID500710654

build/$(STUDENTID)_report.pdf: build/$(STUDENTID)_code.ipynb.pdf build/results.log report/report.tex report/references.bib
	cd report && pdflatex --synctex=1 report.tex && biber report && pdflatex --synctex=1 report.tex && pdflatex --synctex=1 report.tex
	cp report/report.pdf $@

build/results.log: assignment1.py build
	mkdir -p Algorithm/Output
	cd Algorithm && tar xf ../Input.tar.xz
	./assignment1.py | tee $@

build/$(STUDENTID)_code.ipynb.pdf: build/$(STUDENTID)_code.ipynb
	jupyter-nbconvert --output $(notdir $@) --to pdf $<

build/$(STUDENTID)_code.ipynb: assignment1.py build
	yapf -d $<
	pycodestyle $<
	jupytext --output $@ $<

build:
	mkdir $@

clean:
	rm -rf Algorithm build/*
