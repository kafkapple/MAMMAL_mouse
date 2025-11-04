import pickle
import numpy as np
import os
import sys

print(f"Python executable: {sys.executable}")
print(f"NumPy version: {np.__version__}")

file_path = "/home/joon/dev/MAMMAL_mouse/data/preprocessed_shank3/new_cam.pkl"

if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
else:
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print("Successfully loaded pickle file.")
        print(f"Type of loaded data: {type(data)}")
        # Optionally print some keys or structure if it's a dict
        if isinstance(data, dict):
            print(f"Keys: {data.keys()}")
    except Exception as e:
        print(f"Error loading pickle file: {e}")
