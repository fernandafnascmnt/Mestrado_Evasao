PYTHON ?= python3
DATA   ?= data/survey_responses.csv
OUT    ?= results

.PHONY: help install check experiment figures clean

help:
	@echo "install     install the package and its dependencies"
	@echo "check       single reduced repetition, to verify the installation"
	@echo "experiment  full run: 50 repetitions x 10-fold (several hours)"
	@echo "figures     rebuild the figures from an existing results directory"
	@echo "clean       remove generated artefacts"

install:
	$(PYTHON) -m pip install -e .

check:
	$(PYTHON) -m dropout_prediction --data $(DATA) --output $(OUT)_check --reduced

experiment:
	$(PYTHON) -m dropout_prediction --data $(DATA) --output $(OUT)

figures:
	$(PYTHON) -c "from pathlib import Path; from dropout_prediction import figures; figures.build_all(Path('$(OUT)'))"

clean:
	rm -rf $(OUT) $(OUT)_check
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
