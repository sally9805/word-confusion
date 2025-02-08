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

DATASET = "french_gender_probing"
LANGUAGE = "french"
MODEL = "bert-base-cased"
if LANGUAGE == "french":
    MODEL = "dbmdz/bert-base-french-europeana-cased"
elif LANGUAGE == "italian":
    MODEL = "dbmdz/bert-base-italian-cased"

if __name__ == "__main__":


    embedding_config = EmbeddingConfig(model_name=MODEL,
                                       embedding_averaging=True,
                                       embedding_unit_normalization=True,
                                       embedding_layers_to_average=[9, 10, 11, 12])
    
    occurrences_config = OccurrencesConfig(max_samples_for_texts=300000, min_words=20, 
                                           max_words=200, 
                                           random_seed=42)
    
    training_data_sampling_config = TrainingDataSamplingConfig(balance_classes_on_seeds=True,
                                                               balance_classes_on_instances=True,
                                                               balancing_value_seeds=1000, 
                                                               balancing_value_instances=500)

    config = Experiment(embedding_config, occurrences_config, training_data_sampling_config, f'classification/{DATASET}.json', DATA_FOLDER, BASE_DIR_CACHE, language=LANGUAGE, random_seed=42)

    text_loader = TextDataLoader(config)
    texts = text_loader.load_data(folder_path=DATA_FOLDER)

    matcher = FindWordTextOccurrence(
        config=config
    )

    print("Processing seeds and instances...")

    seeds = matcher.process_seeds(config._data_structure.seeds, texts)
    #######################################################################################################
    # Balancing the data. BE CAREFUL WITH THIS. IF YOU EDIT THIS YOU NEED TO REMOVE THE CACHED EMBEDDINGS #
    #######################################################################################################

    seeds["concept"] = seeds["word"].apply(lambda x : config._data_structure.seed_to_concept(x))
    
    if config.training_data_sampling_config.balance_classes_on_seeds:
        c = seeds.groupby("word")
    else:
        c = seeds.groupby("concept")
    c = c.apply(
        lambda x: x.sample(
            n=min(len(x), config.training_data_sampling_config.balancing_value_seeds),
            random_state=14
        ).reset_index(drop=True)
    )
    seed_data = c
    seed_data.reset_index(drop=True, inplace=True)
    seed_data = seed_data.sample(frac=1, random_state=14).reset_index()
    # Update seeds to match the sampled data
    seeds = seed_data

    #######################################################################################################
    # END  Balancing the data                                                                             #
    #######################################################################################################

    instances = matcher.process_instances(config._data_structure.instances, texts)

    instances["target_concept"] = instances["word"].apply(lambda x : config._data_structure.instance_to_concept(x))

    if config.training_data_sampling_config.balance_classes_on_instances:
        c = instances.groupby("target_concept")
    else:
        c = instances.groupby("word")
    c = c.apply(
        lambda x: x.sample(
            n=min(len(x), config.training_data_sampling_config.balancing_value_instances),
            random_state=14
        ).reset_index(drop=True)
    )
    instance_data = c
    instance_data.reset_index(drop=True, inplace=True)
    instance_data = instance_data.sample(frac=1, random_state=14).reset_index()
    # Update instances to match the sampled data
    instances = instance_data

    #######################################################################################################
    # Embedding                                                                                           #
    #######################################################################################################

    embedder = Embedder(config, device="mps")
    seed_embeddings = embedder.process_seeds(seed_data)
    instance_embeddings = embedder.process_instances(instance_data)

    assert len(seed_embeddings) == len(seeds), f"Seed embeddings length {len(seed_embeddings)} does not match seeds length {len(seeds)}"
    assert len(instance_embeddings) == len(instances), f"Instance embeddings length {len(instance_embeddings)} does not match instances length {len(instances)}"

    print(seed_embeddings.shape)

    X = seed_embeddings
    y = seeds["word"]


    clf = Prober()
    clf.train(X, y, config._data_structure.seed_to_concept)

    X_test = instance_embeddings
    y_test = instances["word"]

    predictions = clf.predict_class_for_each_instance(X_test)
    
    instances["predictions"] = predictions

    print("PROBER")

    report = prober_performance(instances, config._data_structure.instance_to_concept)
    print(report)

    print("BASELINE ONE ")

    print(cosine_all_average(seed_embeddings, instance_embeddings, seed_data, instance_data, 
                             config._data_structure.instance_to_concept, 
                             config._data_structure.seed_to_concept))

    print("BASELINE TWO")

    print(cosine_baseline_average_concepts(seed_embeddings, instance_embeddings, 
                                           seed_data, instance_data, 
                                           config._data_structure.instance_to_concept, 
                                            config._data_structure.seed_to_concept))


    print("BASELINE THREE")
    
    print(cosine_baseline_no_averages(seed_embeddings, instance_embeddings, seed_data, instance_data, 
                                      config._data_structure.instance_to_concept, 
                                      config._data_structure.seed_to_concept))