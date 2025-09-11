NB = $(sort $(wildcard examples/*.ipynb))
.PHONY: help clean install lint test doc dist release

help:
	@echo "clean    remove non-source files and clean source files"
	@echo "install  install package in development mode with all dependencies"
	@echo "lint     check style"
	@echo "test     run tests and check coverage"
	@echo "doc      generate HTML documentation and check links"
	@echo "dist     package (source & wheel)"
	@echo "release  package and upload to PyPI"

clean:
	git clean -Xdf
	jupyter nbconvert --inplace --ClearOutputPreprocessor.enabled=True $(NB)

install:
	uv sync --all-extras


lint:
	uv run flake8 --doctests --exclude=doc,.venv,build --max-line-length=88 --extend-ignore=E203

# Matplotlib doesn't print to screen. Also faster.
export MPLBACKEND = agg
# Virtual framebuffer nonetheless needed for the pyqtgraph backend.
export DISPLAY = :99

test:
	Xvfb $$DISPLAY -screen 0 800x600x24 &
	uv run coverage run --branch --source pygsp -m pytest
	uv run coverage report
	uv run coverage html
	killall Xvfb

doc:
	uv run sphinx-build -b html -d doc/_build/doctrees doc doc/_build/html
	uv run sphinx-build -b linkcheck -d doc/_build/doctrees doc doc/_build/linkcheck

dist: clean
	uv sync --all-extras
	uv build
	ls -lh dist/*
	uv run twine check dist/*
	@echo "The built packages are valid and can be uploaded successfully"

release: dist
	uv publish
