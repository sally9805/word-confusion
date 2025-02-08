import requests
import bz2
import os
from pathlib import Path

# Get the directory of the current script
script_dir = Path(__file__).parent


def download_file(url, filename):
    """Download a file from URL and save it to the specified filename"""
    file_path = script_dir / filename  # Save inside the script's directory
    print(f"Downloading {file_path}...")

    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for bad status codes

    with open(file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Downloaded {file_path}")


def decompress_bz2(filename):
    """Decompress a bz2 file and remove the original compressed file"""
    file_path = script_dir / filename  # Locate file in script's directory
    decompressed_file_path = script_dir / filename[:-4]  # Remove .bz2 from filename

    print(f"Decompressing {file_path}...")

    with bz2.open(file_path, 'rb') as source:
        with open(decompressed_file_path, 'wb') as dest:
            dest.write(source.read())

    os.remove(file_path)  # Remove the compressed file
    print(f"Decompressed {file_path}")


# Ensure script's directory exists (should always exist)
script_dir.mkdir(exist_ok=True)

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
