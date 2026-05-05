# Negative test: importlib bypass — must be caught by check-no-network.py
import importlib
http = importlib.import_module("http.client")
