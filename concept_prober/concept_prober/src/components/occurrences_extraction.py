from tqdm import tqdm
from multiprocessing import Pool
import string
import os
import pandas as pd
from concept_prober.src.components.data import Seed, Instance
from concept_prober.src.components.data_loading import Experiment


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


def generate_data_from_json(json_data, *, label="seeds"):
    elements_to_find = json_data[label]["words"]
    elements2concept_dict = {k: v for k, v in
                             zip(json_data[label]["words"], json_data[label]["concepts"])}
    element_to_concept = (lambda x: elements2concept_dict[x])
    return elements_to_find, element_to_concept


class FindWordTextOccurrence:
    def __init__(self, config: Experiment):
        self.config = config

    def func(self, line: str) -> list[tuple[str, str]]:
        collected = []
        for word in self.stimuli:
            target_word = word.word

            # we do minicleaning of the line
            clean_line = preprocess_text(line)
            set_of_words = set(clean_line.split())
            if target_word in set_of_words:
                collected.append((target_word, clean_line))
        return collected

    def extract(self, stimuli: list[Seed] | list[Instance], sentences: list[str], output_location: str, cpus=4):
        # not very pythonic but it works
        self.stimuli = stimuli

        with Pool(cpus) as pool:
            matches = list(tqdm(pool.imap(self.func, sentences), total=len(sentences), position=0))

        with open(output_location, "w") as filino:
            filino.write("word\tsentence\n")
            for k in matches:
                for key, value in k:
                    filino.write(f"{key}\t{value.strip()}\n")

    def process(self, stimuli: list[Seed] | list[Instance], sentences: list[str], output_location: str, cpus=4):
        if not os.path.exists(output_location):
            self.extract(stimuli, sentences, output_location, cpus)
        else:
            print("Occurrences Exists. Skipping...")

        return pd.read_csv(output_location, sep="\t")

    def process_seeds(self, seed_to_find: list[Seed], texts: list[str], cpus=4):
        print("Searching for seeds...")
        output_location = f"{self.config.experiment_folder}/seeds.tsv"
        return self.process(seed_to_find, texts, output_location, cpus)

    def process_instances(self, instance_to_find: list[Instance], texts: list[str], cpus=4):
        print("Searching for instances...")
        output_location = f"{self.config.experiment_folder}/instances.tsv"
        return self.process(instance_to_find, texts, output_location, cpus)
