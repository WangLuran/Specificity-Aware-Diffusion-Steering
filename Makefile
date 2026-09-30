.PHONY: check reference-t2i reference-binder

check:
	python -m pytest -q

reference-t2i:
	python t2i/summarize_results.py --condition-csv t2i/reference/condition_results.csv

reference-binder:
	python binder/summarize_table.py --reference

