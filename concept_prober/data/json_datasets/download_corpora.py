import requests
import bz2
import os
from pathlib import Path

def download_file(url, filename):
    """Download a file from URL and save it to the specified filename"""
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for bad status codes
    
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded {filename}")

def decompress_bz2(filename):
    """Decompress a bz2 file and remove the original compressed file"""
    print(f"Decompressing {filename}...")
    with bz2.open(filename, 'rb') as source:
        with open(filename[:-4], 'wb') as dest:  # Remove .bz2 from filename
            dest.write(source.read())
    os.remove(filename)  # Remove the compressed file
    print(f"Decompressed {filename}")

# Create current directory if it doesn't exist
current_dir = Path('.')
current_dir.mkdir(exist_ok=True)

# Download and process Italian abstract
it_abstract_url = "https://databus.dbpedia.org/dbpedia/text/long-abstracts/2021.06.01/long-abstracts_lang=it.ttl.bz2"
it_abstract_file = "long-abstracts_lang=it.ttl.bz2"
download_file(it_abstract_url, it_abstract_file)
decompress_bz2(it_abstract_file)

# Download Flex_it files
flex_it_nouns_url = "https://raw.githubusercontent.com/franfranz/Flex_it/main/Flex_it_nouns.csv"
flex_it_adj_url = "https://raw.githubusercontent.com/franfranz/Flex_it/main/Flex_it_adj.csv"
download_file(flex_it_nouns_url, "Flex_it_nouns.csv")
download_file(flex_it_adj_url, "Flex_it_adj.csv")

# Download and process French abstract
fr_abstract_url = "https://databus.dbpedia.org/dbpedia/text/long-abstracts/2021.06.01/long-abstracts_lang=fr.ttl.bz2"
fr_abstract_file = "long-abstracts_lang=fr.ttl.bz2"
download_file(fr_abstract_url, fr_abstract_file)
decompress_bz2(fr_abstract_file)