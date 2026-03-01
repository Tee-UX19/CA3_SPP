"""Download SPP files from OR-Library."""

import urllib.request
import os

os.makedirs("data", exist_ok=True)

files = {
    "sppnw41.txt": "http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/sppnw41.txt",
    "sppnw42.txt": "http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/sppnw42.txt",
    "sppnw43.txt": "http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/sppnw43.txt",
}

for filename, url in files.items():
    dest = f"data/{filename}"
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  Saved to {dest}")
    except Exception as e:
        print(f"  FAILED: {e}")

print("Done.")
