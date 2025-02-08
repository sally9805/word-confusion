import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from collections import defaultdict
from typing import List
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

import os
import pickle

from concept_prober.src.components.data import Data
from concept_prober.src.components.data_loading import Experiment

class Embedder:

    def __init__(self, config: Experiment, device:str="cuda"):
        self.model_name = config.embedding_config.model_name
        self.max_length = config.embedding_config.max_length
        self.model = AutoModel.from_pretrained(self.model_name, output_hidden_states=True).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.device = device
        self.config = config

    def find_sub_list(self, sl, l):
        results = list()
        sll = len(sl)
        for ind in (i for i, e in enumerate(l) if e == sl[0]):
            if l[ind: ind + sll] == sl:
                results.append((ind, ind + sll))

        return results

    def subset_of_tokenized(self, list_of_tokens:List[int]):
        idx = list_of_tokens.index(self.tokenizer.pad_token_id) # type: ignore
        return list_of_tokens[0:idx]

    def embed(self, target_texts: List[str], words: List[str], layers_id: List[int], batch_size:int, * , averaging:bool=False):

        words = [f" {word.strip()}" for word in words]

        original = pd.DataFrame({"text": target_texts, "words": words})

        test_dataset = datasets.Dataset.from_pandas(original)

        def tokenizer_function(examples):
            text_inputs = self.tokenizer(examples["text"], max_length=self.max_length, padding="max_length", truncation=True)
            word_inputs = self.tokenizer(examples["words"], max_length=20, padding="max_length", truncation=True,
                                         add_special_tokens=False)

            examples["input_ids"] = text_inputs.input_ids
            examples["attention_mask"] = text_inputs.attention_mask
            examples["words_input_ids"] = word_inputs.input_ids

            return examples

        encoded_test = test_dataset.map(tokenizer_function, remove_columns=["text", "words"])
        encoded_test.set_format("pt")

        dl = DataLoader(encoded_test, batch_size=batch_size) # type: ignore

        embs = defaultdict(list)
        pbar = tqdm(total=len(dl), position=0)

        for batch in dl:
            words_ids = batch["words_input_ids"]
            pbar.update(1)
            del batch["words_input_ids"]

            batch = {k: v.to(self.device) for k, v in batch.items()}

            features = self.model(**batch)["hidden_states"]

            for layer in layers_id:
                layer_features = features[layer]

                try:
                    idx = [
                        self.find_sub_list(self.subset_of_tokenized(tok_word.tolist()), input_ids.tolist())[0]
                        for tok_word, input_ids in zip(words_ids, batch["input_ids"])
                    ]
                except IndexError as e:
                    raise Exception("Index Error: do all the words occur in the respective sentences?")

                if averaging:
                    for embedded_sentence_tokens, (l_idx, r_idx) in zip(layer_features, idx):
                        embs[layer].append(embedded_sentence_tokens[l_idx:r_idx, :].mean(0).detach().cpu().numpy())
                else:
                    for embedded_sentence_tokens, (l_idx, r_idx) in zip(layer_features, idx):
                        embs[layer].append(embedded_sentence_tokens[l_idx:l_idx+1, :].mean(0).detach().cpu().numpy())

        pbar.close()
        return embs

    def process_embedding(self, data: pd.DataFrame, output_location: str, batch_size: int = 8):

        if os.path.exists(output_location):
            print(f"LOADING EMBEDDINGS FROM {output_location}")
            with open(output_location, "rb") as f:
                embeddings = pickle.load(f)
        else:
            embeddings = self.embed(data["sentence"].values.tolist(),  # type: ignore
                                    data["word"].values.tolist(),  # type: ignore
                                     self.config.embedding_config.embedding_layers_to_average,
                                      batch_size=batch_size, averaging=True)
            
            with open(output_location, "wb") as f:
                pickle.dump(embeddings, f)

        if self.config.embedding_config.embedding_averaging:
            embeddings = np.average([embeddings[12], embeddings[11], embeddings[10], embeddings[9]], axis=0)
        else:
            raise ValueError("Non averaging embedding not supported, but if you want to use it you can edit here")
        if self.config.embedding_config.embedding_unit_normalization:
            print("Unit normalization... Check if this is correct")
            embeddings = embeddings/np.linalg.norm(embeddings, axis=1, keepdims=True)
        # else:
        #     raise ValueError("Non unit normalization embedding not supported, but if you want to use it you can edit here")
        
        return embeddings
    
    def process_seeds(self, seed_data: pd.DataFrame, batch_size: int = 8) -> np.ndarray:
        output_location = f"{self.config.experiment_folder}/seeds_embeddings.pkl"
        return self.process_embedding(seed_data, output_location, batch_size)

    def process_instances(self, instance_data: pd.DataFrame, batch_size: int = 8) -> np.ndarray:
        output_location = f"{self.config.experiment_folder}/instances_embeddings.pkl"
        return self.process_embedding(instance_data, output_location, batch_size)



