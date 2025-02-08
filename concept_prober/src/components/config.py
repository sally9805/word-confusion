from dataclasses import dataclass, field
from typing import Optional
from concept_prober.src.components.data import Data, load_data_from_json


@dataclass(kw_only=True, frozen=True)
class EmbeddingConfig:
    model_name: str
    embedding_averaging: bool = True # right now this is the only option
    embedding_unit_normalization: bool = False
    max_length: int = 400 # max length of the text to embed (in tokens  )
    embedding_layers_to_average: list[int] = field(default_factory=lambda: [9, 10, 11, 12]) # layers to average for the embedding
    need_to_do_instances: bool = True # do you want to do instances or not? this will simply skip generating data for the instances (e.g., similarity datasets do not need instances)


@dataclass(kw_only=True, frozen=True)
class OccurrencesConfig:
    max_samples_for_texts: int = 25
    min_words: int = 20 # min words to keep a sentence of the corpus
    max_words: int = 200 # max words to keep a sentence of the corpus (if you increase this too much you might end up getting an error because the model will not be able to process the text)
    random_seed: int = 42


@dataclass(kw_only=True, frozen=True)
class TrainingDataSamplingConfig:
    balance_classes_on_seeds: bool = False #  do you want to sample on the seeds, thus taking 20 of each occurrence of "evil" or balance on the concepts, thus taking 20 of "positive"
    seed_sample_with_replacement: bool = False # do you want to sample with replacement
    instances_sample_with_replacement: bool = False
    balance_classes_on_instances: bool = False
    balancing_value_seeds: int = 100 # the actual number of samples for each concept or seed
    balancing_value_instances: int = 100 # the actual number of samples for each concept or seed


@dataclass(kw_only=True)
class Experiment:
    embedding_config: EmbeddingConfig
    occurrences_config: OccurrencesConfig
    training_data_sampling_config: TrainingDataSamplingConfig
    task: str # basically the name of the dataset
    language: str # the language of the dataset
    data_folder: str # the folder where the dataset is stored
    cache_folder: str # the folder where the cache is stored
    random_seed: int = 42
    _data_structure: Data # this is the dataset structure with seeds and instances

    def __init__(self, embedding_config: EmbeddingConfig, occurrences_config: OccurrencesConfig, training_data_sampling_config: TrainingDataSamplingConfig, task: str, data_folder: str, cache_folder: str, language: str, random_seed: int = 42):
        self.embedding_config = embedding_config
        self.occurrences_config = occurrences_config
        self.training_data_sampling_config = training_data_sampling_config
        self.task = task
        self.data_folder = data_folder
        self.cache_folder = cache_folder
        self.language = language
        self.random_seed = random_seed
        self.load_task_data()

    # get all params and make a folder with the name of the experiment
    @property
    def experiment_folder(self) -> str:
        import os
        os.makedirs(f"{self.cache_folder}/{self.task}", exist_ok=True)
        return f"{self.cache_folder}/{self.task}"
    

    def load_task_data(self) -> 'Data':
        """Load task-specific JSON data."""
        try:
            print(f"Reading task data from {self.data_folder}/{self.task}")
            self._data_structure = load_data_from_json(f"{self.data_folder}/{self.task}")
            return self._data_structure
        except FileNotFoundError:
            raise FileNotFoundError(f"Task file not found: {self.task}")

    @property
    def data_structure(self) -> Optional['Data']:
        """Get the loaded data structure."""
        return self._data_structure
    