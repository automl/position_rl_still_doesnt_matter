install:
	@echo "Installing..."
	uv venv --python 3.11 && source .venv/bin/activate
	uv pip install -r requirements.txt
	@echo "Done!"
	@echo "To uninstall, run 'make uninstall'"

format:
	@echo "Formatting code"
	@uv run ruff format processing

process:
	@echo "Processing data"
	@echo "First, scrape and process conference papers."
	@echo "This will very likely take a while, so grab a coffee (or lunch)."
	@echo ""
	python automatic_data_processing.py
	@echo ""
	@echo "Now let's process the manually collected data as well:"
	@echo ""
	python manual_data_processing.py

