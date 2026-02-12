import sys
import os

print("-" * 30)
print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")
print("-" * 30)

try:
    import nltk
    print(f"NLTK Location: {nltk.__file__}")
    print("NLTK Import Successful")
except ImportError as e:
    print(f"NLTK Import Failed: {e}")
print("-" * 30)
