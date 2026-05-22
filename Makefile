.PHONY: install-dev test-cpu benchmark reproduce docker-smoke

install-dev:
	python -m pip install -r requirements-dev.txt
	python -m pip install -e .

test-cpu:
	pytest -m "not gpu"

benchmark:
	python scripts/generate_scaling_report.py --output benchmarks/results/scaling_report.md
	python scripts/run_benchmark_suite.py --lengths 128 256 512 --csv benchmarks/results/model_family_scaling.csv --no-plot

reproduce: test-cpu benchmark
	python scripts/prepare_dataset.py --examples 64 --length 96 --output-dir data/processed/promoter-smoke
	python scripts/train_tiny_checkpoint.py --output-dir artifacts/tiny-hogfm

docker-smoke:
	docker build -t hogfm-api:smoke .
	docker run --rm -p 8000:8000 hogfm-api:smoke
