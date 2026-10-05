.POSIX:
.PHONY: help test quick figures clean distclean lint env reproduce reproduce-dynamics exp01 exp02 exp03 exp04 exp05 exp06 exp07 exp08 exp09

PYTHON ?= python3
EXP    := $(PYTHON) experiments

# Every experiment script, in the order of Section 10 of the paper.
SCRIPTS := exp01_forward exp02_transition exp03_inequalities exp04_free_vs_classical \
           exp05_transfer exp06_design_mp exp07_reverse exp08_spiked exp09_generative

help:
	@echo "Targets:"
	@echo "  make test       run the test suite (about 15 s)"
	@echo "  make quick      run all experiments in reduced configuration (a few minutes)"
	@echo "  make figures    run all experiments at the settings used in the paper (hours)"
	@echo "  make expNN      run a single experiment, e.g. make exp06"
	@echo "  make reproduce  run the self-contained checks in reproduce/ (about a minute)"
	@echo "  make clean      remove generated figures, results and caches"
	@echo "  make env        print the versions the results depend on"
	@echo ""
	@echo "Variables: PYTHON (default python3), QUICK (set to 1 to add --quick),"
	@echo "           SEED (passed through as --seed)."

FLAGS :=
ifdef QUICK
FLAGS += --quick
endif
ifdef SEED
FLAGS += --seed $(SEED)
endif

test:
	$(PYTHON) -m pytest tests/ -q

quick:
	@$(MAKE) figures QUICK=1

figures:
	@for s in $(SCRIPTS); do \
	  echo "==> $$s"; \
	  $(PYTHON) experiments/$$s.py $(FLAGS) || exit 1; \
	done
	@echo ""
	@echo "Figures in figures/, numerical output in results/."

# Individual experiments: make exp01, make exp02, ...
exp01: ; $(PYTHON) experiments/exp01_forward.py $(FLAGS)
exp02: ; $(PYTHON) experiments/exp02_transition.py $(FLAGS)
exp03: ; $(PYTHON) experiments/exp03_inequalities.py $(FLAGS)
exp04: ; $(PYTHON) experiments/exp04_free_vs_classical.py $(FLAGS)
exp05: ; $(PYTHON) experiments/exp05_transfer.py $(FLAGS)
exp06: ; $(PYTHON) experiments/exp06_design_mp.py $(FLAGS)
exp07: ; $(PYTHON) experiments/exp07_reverse.py $(FLAGS)
exp08: ; $(PYTHON) experiments/exp08_spiked.py $(FLAGS)
exp09: ; $(PYTHON) experiments/exp09_generative.py $(FLAGS)

reproduce:
	bash reproduce/run_all.sh

reproduce-dynamics:
	bash reproduce/run_all.sh --with-dynamics

env:
	@$(PYTHON) -c "import sys, numpy, scipy, matplotlib; \
print('python     ', sys.version.split()[0]); \
print('numpy      ', numpy.__version__); \
print('scipy      ', scipy.__version__); \
print('matplotlib ', matplotlib.__version__)"

lint:
	@$(PYTHON) -m compileall -q freeddpm experiments tests && echo "syntax ok"

clean:
	rm -rf figures/*.png results/*.json
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache

distclean: clean
	rm -rf .venv
