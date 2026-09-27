.PHONY: data preprocess fallback test

data:
	python ml/generate_data.py --mode month

preprocess:
	python ml/preprocess.py

fallback:
	python scripts/train_demo_fallback.py

test:
	pytest -q
