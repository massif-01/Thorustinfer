.PHONY: check check-locks check-dependency-age fmt lint build info python-env cuda-env check-offline check-gpu

UV ?= uv
PYTHON = $(UV) run --locked python
CUDA_PYTHON = $(UV) run --locked --project tools/environments/cuda python

python-env: check-dependency-age
	$(UV) sync --locked

cuda-env: check-dependency-age
	$(UV) sync --locked --project tools/environments/cuda

check-dependency-age:
	python3 -m tools.dev.dependencies check

check-locks: check-dependency-age
	$(UV) lock --check --offline
	$(UV) lock --check --offline --project tools/environments/cuda

build: check-dependency-age
	cargo build --release --locked --offline

fmt:
	cargo fmt --all
	$(PYTHON) -m ruff format tools kernels python tests setup.py

lint:
	cargo fmt --all -- --check
	$(PYTHON) -m ruff format --check tools kernels python tests setup.py
	$(PYTHON) -m ruff check tools kernels python tests setup.py
	cargo clippy --workspace --all-targets --locked --offline -- -D warnings

check: check-locks lint
	cargo check --workspace --locked --offline
	cargo test --workspace --locked --offline
	$(PYTHON) -m thorustinfer_build.common.testing --environment cpu

info:
	cargo run --offline -p thorustinfer-cli -- info

check-offline:
	$(CUDA_PYTHON) -m thorustinfer_build.common.check_imports --backend cuda
	$(CUDA_PYTHON) -m thorustinfer_build.common.testing --environment cuda

check-gpu:
	$(PYTHON) tools/bench/check_gpu.py --model "$(MODEL)" --output "$(OUTPUT)"
