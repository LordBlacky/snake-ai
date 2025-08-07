VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

venv:
	python3.12 -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

clean:
	rm -rf $(VENV)

run: install
	$(VENV)/bin/python main.py
