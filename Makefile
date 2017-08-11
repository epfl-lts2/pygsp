.PHONY: clean-pyc clean-build doc clean

help:
	@echo "clean-build - remove build artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "lint - check style"
	@echo "test - run tests quickly with the default Python"
	@echo "coverage - check code coverage quickly with the default Python"
	@echo "doc - generate Sphinx HTML documentation, including API doc"
	@echo "release - package and upload a release"
	@echo "dist - package"

clean: clean-build clean-pyc
	rm -fr htmlcov/

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr *.egg-info

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +

lint:
	flake8 --doctests

test:
	python setup.py test

coverage:
	coverage run --source pyGSP setup.py test
	coverage report -m
	coverage html
	open htmlcov/index.html

doc:
	$(MAKE) -C doc clean
	$(MAKE) -C doc html

release: clean
	python setup.py register
	python setup.py sdist upload
#	python setup.py bdist_wheel upload

dist: clean
	python setup.py sdist
#	python setup.py bdist_wheel
	ls -l dist
