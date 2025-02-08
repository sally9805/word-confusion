from concept_prober.src.components.data import Data
from concept_prober.src.components.data_loading import *
from concept_prober.src.components.occurrences_extraction import FindWordTextOccurrence
from concept_prober.constants import BASE_DIR_CACHE, DATA_FOLDER
import pandas as pd
from sklearn.metrics import classification_report
from concept_prober.src.components.embedder import Embedder
from concept_prober.src.components.prober import Prober
from concept_prober.runners.metrics import *
from concept_prober.src.components.config import *

if __name__ == "__main__":


    embedding_config = EmbeddingConfig(model_name="bert-base-uncased", 
                                       embedding_averaging=True, 
                                       embedding_unit_normalization=True, 
                                       embedding_layers_to_average=[9, 10, 11, 12],
                                       need_to_do_instances=True)
    
    occurrences_config = OccurrencesConfig(max_samples_for_texts=1000,
                                           min_words=20, 
                                           max_words=200, 
                                           random_seed=42)
    
    training_data_sampling_config = TrainingDataSamplingConfig(balance_classes_on_seeds=True, 
                                                               balance_classes_on_instances=True, 
                                                               seed_sample_with_replacement=True,
                                                               instances_sample_with_replacement=True,
                                                               balancing_value_seeds=5, 
                                                               balancing_value_instances=5)

    config = Experiment(embedding_config, occurrences_config, training_data_sampling_config,
                         "similarity/wordsim.json",
                         DATA_FOLDER, 
                         BASE_DIR_CACHE,
                         language="english", 
                         random_seed=42)

    text_loader = TextDataLoader(config)
    texts = text_loader.load_data()

    matcher = FindWordTextOccurrence(
        config=config
    )
    embedder = Embedder(config, device="mps")

    #######################################################################################################
    # Process seeds
    #######################################################################################################
    
    seeds = matcher.process_seeds(config._data_structure.seeds, texts) # this is going to save a seeds.tsv file on disk

    seeds["concept"] = seeds["word"].apply(lambda x : config._data_structure.seed_to_concept(x))
    
    if config.training_data_sampling_config.balance_classes_on_seeds:
        c = seeds.groupby("word")
    else:
        c = seeds.groupby("concept")
    
    c = c.apply(lambda x: x.sample(config.training_data_sampling_config.balancing_value_seeds, random_state=14, 
                                   replace=config.training_data_sampling_config.seed_sample_with_replacement).reset_index(drop=True))
    seed_data = c
    seed_data.reset_index(drop=True, inplace=True)
    seed_data = seed_data.sample(frac=1, random_state=14).reset_index()

    print(f"Seed data length: {len(seed_data)}")

    seed_embeddings = embedder.process_seeds(seed_data)
    seed_data.to_csv(f"{config.experiment_folder}/seeds_after_balancing.tsv", sep="\t", index=False)
    # embedding.pkl and a seeds_after_balancing.tsv file are saved on disk (one to one mapping between the two)

    assert len(seed_embeddings) == len(seed_data), f"Seed embeddings length {len(seed_embeddings)} does not match seeds length {len(seeds)}"

    if config.embedding_config.need_to_do_instances:

    #######################################################################################################
    # Process instances
    #######################################################################################################

        instances = matcher.process_instances(config._data_structure.instances, texts)

        instances["concept"] = instances["word"].apply(lambda x : config._data_structure.instance_to_concept(x))

        if config.training_data_sampling_config.balance_classes_on_instances:
            c = instances.groupby("concept")
        else:
            c = instances.groupby("word")

        c = c.apply(lambda x: x.sample(config.training_data_sampling_config.balancing_value_instances, random_state=14, 
                                    replace=config.training_data_sampling_config.instances_sample_with_replacement).reset_index(drop=True))
        instance_data = c
        instance_data.reset_index(drop=True, inplace=True)
        instance_data = instance_data.sample(frac=1, random_state=14).reset_index()

        print(f"Instance data length: {len(instance_data)}")    

        instance_embeddings = embedder.process_instances(instance_data)

        assert len(instance_embeddings) == len(instance_data), f"Instance embeddings length {len(instance_embeddings)} does not match instances length {len(instances)}"

        instance_data.to_csv(f"{config.experiment_folder}/instances_after_balancing.tsv", sep="\t", index=False)
