import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from datasets import load_dataset
from dataclasses import dataclass, field
from pathlib import Path
import string
from concept_prober.src.components.config import Experiment

def preprocess_text(text: str) -> str:
       
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    
    return text
   
class TextDataLoader:
    """
    This class covers loading the various text datasets. English comes from HuggingFace, Italian and French come from the ttl files.
    the ttl files are abstracts from a DBpedia dump.
    """
    def __init__(self, config: Experiment):
        self.config = config
        self.texts: List[str] = []
        print("Loading data...")
        
    def load_ttl_file(self, file_path: str) -> List[str]:
        """Load and process text from TTL file."""
        texts = []
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    try:
                        # Extract text content after second '>'
                        text = line.split('>')[2].strip().replace('\n', '')
                        texts.append(text)
                    except IndexError:
                        continue
            return texts
        except FileNotFoundError:
            raise FileNotFoundError(f"TTL file not found at: {file_path}")

    def load_english_data(self) -> List[str]:
        """Load English text data from HuggingFace dataset."""
        dataset = load_dataset(
            "wikitext", 
            "wikitext-103-v1", 
            cache_dir=self.config.cache_folder
        )
        dataset = dataset.shuffle(seed=self.config.random_seed)
        return dataset["train"]["text"] # type: ignore

    def filter_texts(self, texts: List[str]) -> List[str]:
        """Filter texts based on word count criteria."""
        return [
            text for text in texts 
            if self.config.occurrences_config.min_words < len(text.split()) < self.config.occurrences_config.max_words
        ]

    def load_data(self, folder_path: Optional[str] = None) -> List[str]:
        """Load and process text data based on language."""
        if self.config.language.lower() == "english":
            texts = self.load_english_data()
        
        elif self.config.language.lower() in ["italian", "french"]:
            if not folder_path:
                raise ValueError("folder_path is required for Italian/French data")
            
            file_path = Path(folder_path) / f"long-abstracts_lang={self.config.language.lower()[:2]}.ttl"
            texts = self.load_ttl_file(str(file_path))
        
        else:
            raise ValueError(f"Unsupported language: {self.config.language}")

        # Filter and shuffle texts and also preprocess them
        texts = self.filter_texts(texts)
        texts = [preprocess_text(text) for text in texts]
        random.seed(self.config.random_seed)
        random.shuffle(texts)
        
        # Limit sample size
        self.texts = texts[:self.config.occurrences_config.max_samples_for_texts]
        print(f"Loaded {len(self.texts)} texts")
        return self.texts