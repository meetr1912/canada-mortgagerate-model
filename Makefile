# Makefile for setting up the project

# Define the Python interpreter and virtual environment directory
PYTHON = python
VENV_DIR = venv

# Default target
all: setup run

# Create a virtual environment and install dependencies
setup:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "Activating virtual environment and installing dependencies..."
	$(VENV_DIR)/Scripts/activate && pip install --upgrade pip && pip install -r requirements.txt

# Run the Python script
run:
	@echo "Running the script..."
	$(VENV_DIR)/Scripts/activate && $(PYTHON) predict_mortgage_rates.py

# Clean up the virtual environment
clean:
	@echo "Cleaning up..."
	rm -rf $(VENV_DIR)

.PHONY: all setup run clean