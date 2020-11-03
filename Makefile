file_finder = find . -type f $(1) -not -path './venv/*'

PY_FILES = $(call file_finder,-name "*.py")

check: check_format

format:
	$(PY_FILES) | xargs black

check_format:
	$(PY_FILES) | xargs black --diff --check

install:
	pip3 install -r requirements.txt

init_venv:
	python3 -m venv venv
