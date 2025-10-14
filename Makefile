
PYTHON_VERSION := 3.11
VENV  := .venv

TEST_SCRIPT := src/test.py
TRAIN_SCRIPT := src/train/train.py
TEST_INFERENCE_SCRIPT := src/inference/test_inference.py

.PHONY: env test train test_inference

env:
	@command -v uv >/dev/null 2>&1 || { \
	echo "Installing uv..."; \
	curl -LsSf https://astral.sh/uv/install.sh | sh; \
	}
	@echo "Setting up eval environment..."
	@uv venv $(VENV) --python $(PYTHON_VERSION) --no-project
	@uv pip install -r requirements.txt --python $(VENV)/bin/python
	@echo "Evaluation environment ready."

test:
	@echo "Running tests..."
	@$(VENV)/bin/python  $(TEST_SCRIPT)

train:
	@echo "Running training..."
	PYTHONPATH=. $(VENV)/bin/python -m src.train.train --config src/train/config.yaml

test_inference:
	@echo "Running test_inference..."
	PYTHONPATH=. $(VENV)/bin/python -m src.inference.test_inference --config src/inference/inference_config.yaml
