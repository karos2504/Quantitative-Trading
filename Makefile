.PHONY: install test lint run-kpi run-strats clean

install:
	pip3 install -r requirements.txt

test:
	python3 -m pytest tests/ -v

lint:
	python3 -m flake8 . || echo "Flake8 not installed or found issues"

run-kpi:
	python3 -m examples.kpi_demo

run-strats:
	python3 -m strategies.rebalance_portfolio
	python3 -m strategies.resistance_breakout
	python3 -m strategies.renko_obv

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -f yfinance.cache.sqlite
