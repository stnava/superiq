pip install coverage
python3 -m coverage run --source=superiq -m unittest discover -s tests
python3 -m coverage report
