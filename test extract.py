import sys
import os

if len(sys.argv) < 2:
    print("Press enter to process all files with .txt extension.")
    input()
    files = [f for f in os.listdir('.') if os.path.isfile(f) and f.endswith('.txt')]
else:
    files = sys.argv[1:]

print("Files: %s" % ', '.join(files))
print()
