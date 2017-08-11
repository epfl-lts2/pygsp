.PHONY: help clean lint test doc dist release

help:
	@echo "clean    remove non-source files"
	@echo "lint     check style"
	@echo "test     run tests and check coverage"
	@echo "doc      generate HTML documentation and check links"
	@echo "dist     package (source & wheel)"
	@echo "release  package and upload to PyPI"

clean:
	# Python files.
	find . -name '__pycache__' -exec rm -rf {} +
	# Documentation.
	rm -rf doc/_build
	# Coverage.
	rm -rf .coverage
	rm -rf htmlcov
	# Package build.
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info

lint:
	flake8 --doctests

test:
	xvfb-run coverage run --branch --source pygsp setup.py test
	coverage report
	coverage html

doc:
	sphinx-build -b html -d doc/_build/doctrees doc doc/_build/html
	sphinx-build -b linkcheck -d doc/_build/doctrees doc doc/_build/linkcheck

dist: clean
	python setup.py sdist
	python setup.py bdist_wheel --universal
	ls -l dist

release: dist
	twine upload dist/*
