# Makefile for Quantum Gravity Lab.  `make` or `make help` lists the targets.
#
# make install       Ubuntu packages (apt), a virtualenv in .venv, and every Python dependency, EinsteinPy included
# make test          fast test suite;  make test-slow also runs the slow EinsteinPy geodesic checks
# make figures       regenerate every figure in figures/

VENV      ?= .venv
PY        := $(VENV)/bin/python
PIP       := $(VENV)/bin/pip
SUDO      := $(if $(filter 0,$(shell id -u)),,sudo)
APT_PKGS  := python3 python3-venv python3-pip python3-dev build-essential git

# no .pyc files or .pytest_cache in the working tree; headless matplotlib
export PYTHONDONTWRITEBYTECODE := 1
export MPLBACKEND := Agg
PYTEST    := $(PY) -m pytest -p no:cacheprovider

.DEFAULT_GOAL := help
.PHONY: help install install-system install-python check-install test test-slow test-all \
        figures figure-compare figure-horizon figure-tokamak run-compare run-horizon clean distclean

help:
	@echo "Quantum Gravity Lab"
	@echo "  make install         apt packages + virtualenv ($(VENV)) + Python packages incl. EinsteinPy"
	@echo "                       (make install SKIP_APT=1 to skip the apt step)"
	@echo "  make check-install   confirm the key imports, EinsteinPy included, work"
	@echo "  make test            fast tests (about 3 min)"
	@echo "  make test-slow       only the slow EinsteinPy geodesic tests (about 2 min)"
	@echo "  make test-all        every test, slow ones included"
	@echo "  make figures         all figures: compare, horizon (pancake + Mixmaster), tokamak"
	@echo "  make figure-compare | figure-horizon | figure-tokamak   one figure set"
	@echo "  make run-compare     print the comparison table;  make run-horizon  print the horizon summary"
	@echo "  make clean           remove caches;  make distclean also removes $(VENV)"

install: $(if $(SKIP_APT),,install-system) install-python check-install

install-system:
	$(SUDO) apt-get update
	$(SUDO) env DEBIAN_FRONTEND=noninteractive apt-get install -y $(APT_PKGS)

install-python: $(PY)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

$(PY):
	python3 -m venv $(VENV)

check-install: $(PY)
	$(PY) -c "import numpy, scipy, pandas, matplotlib, sympy, mpmath, einsteinpy; print(\"einsteinpy\", einsteinpy.__version__, \"| numpy\", numpy.__version__, \"| scipy\", scipy.__version__)"

test: $(PY)
	$(PYTEST) tests -q

test-slow: $(PY)
	$(PYTEST) tests/test_einsteinpy.py -q --runslow -m slow

test-all: $(PY)
	$(PYTEST) tests -q --runslow

figures: figure-compare figure-horizon figure-tokamak

figure-compare: $(PY)
	$(PY) qgl_compare.py --figure

figure-horizon: $(PY)
	$(PY) qgl_horizon.py --figures

figure-tokamak: $(PY)
	$(PY) qgl_tokamak.py --figure

run-compare: $(PY)
	$(PY) qgl_compare.py

run-horizon: $(PY)
	$(PY) qgl_horizon.py

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache

distclean: clean
	rm -rf $(VENV)
