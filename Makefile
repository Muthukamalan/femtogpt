.PHONY: clean
clean:
	@find ./ -type d -name ".venv" -exec rm -rf {} +
	@find ./ -type f -name "uv.lock" -exec rm -f {} +
	@find ./ -type d -name "__pycache__" -o -name ".ruff_cache" -exec rm -rf {} +

clean-logs:
	rm -rf checkpoints/*
	find ./ -type f -name "mlflow.db" -exec rm -rf {} +
	find ./ -type f -name "mlruns.db" -exec rm -rf {} +

mlflow:
	mlflow server --port 8080 --backend-store-uri sqlite:///mlruns.db