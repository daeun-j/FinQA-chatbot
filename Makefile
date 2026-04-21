.PHONY: data index serve eval ui setup

setup:
	pip install -r requirements.txt

data:
	python data/download.py

index:
	python scripts/run_index.py

serve:
	bash scripts/serve_vllm.sh

eval:
	python scripts/run_eval.py

ui:
	python app/main.py

all: data index ui
