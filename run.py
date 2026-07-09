import os
import sys

print("Current Folder:")
print(os.getcwd())

print("\nPython Path:")
for path in sys.path:
    print(path)