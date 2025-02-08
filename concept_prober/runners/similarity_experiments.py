import os

from concept_prober.src.components.data import Data
from concept_prober.src.components.data_loading import *
from concept_prober.src.components.occurrences_extraction import FindWordTextOccurrence
from concept_prober.constants import BASE_DIR_CACHE, DATA_FOLDER
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine
from concept_prober.src.components.embedder import Embedder
from concept_prober.src.components.prober import Prober
from concept_prober.runners.metrics import *
from concept_prober.src.components.config import *

DATASET = "wordsim"

if __name__ == "__main__":

    #######################################################################################################
    # Configuration                                                                                        #
    #######################################################################################################

    embedding_config = EmbeddingConfig(
        model_name="bert-base-cased",
        embedding_averaging=True,
        embedding_unit_normalization=True,
        embedding_layers_to_average=[9, 10, 11, 12],
        need_to_do_instances=True
        )

    occurrences_config = OccurrencesConfig(
        max_samples_for_texts=1000,
        min_words=20,
        max_words=200,
        random_seed=42
        )

    training_data_sampling_config = TrainingDataSamplingConfig(
        balance_classes_on_seeds=True,
        balance_classes_on_instances=True,
        seed_sample_with_replacement=True,
        instances_sample_with_replacement=True,
        balancing_value_seeds=5,
        balancing_value_instances=5
        )

    config = Experiment(
        embedding_config, occurrences_config, training_data_sampling_config,
        f'similarity/{DATASET}.json',
        DATA_FOLDER,
        BASE_DIR_CACHE,
        language="english",
        random_seed=42
        )

    #######################################################################################################
    # Data Loading and Processing                                                                         #
    #######################################################################################################

    text_loader = TextDataLoader(config)
    texts = text_loader.load_data(folder_path=DATA_FOLDER)

    matcher = FindWordTextOccurrence(
        config=config
    )

    print("Processing seeds and instances...")

    seeds = matcher.process_seeds(config._data_structure.seeds, texts)
    seeds["concept"] = seeds["word"].apply(lambda x: config._data_structure.seed_to_concept(x))

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
    seeds = seed_data

    instances = matcher.process_instances(config._data_structure.instances, texts)
    instances["target_concept"] = instances["word"].apply(lambda x: config._data_structure.instance_to_concept(x))

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
    instances = instance_data

    #######################################################################################################
    # Embedding Generation                                                                                #
    #######################################################################################################

    embedder = Embedder(config, device="mps")
    seed_embeddings = embedder.process_seeds(seed_data)
    instance_embeddings = embedder.process_instances(instance_data)

    assert len(seed_embeddings) == len(
        seeds
        ), f"Seed embeddings length {len(seed_embeddings)} does not match seeds length {len(seeds)}"
    assert len(instance_embeddings) == len(
        instances
        ), f"Instance embeddings length {len(instance_embeddings)} does not match instances length {len(instances)}"

    #######################################################################################################
    # Prober Training and Prediction                                                                      #
    #######################################################################################################

    X = seed_embeddings
    y = seeds["word"]

    clf = Prober()
    clf.train(X, y, config._data_structure.seed_to_concept)

    X_test = instance_embeddings
    y_test = instances["word"]

    # Get probability predictions instead of class predictions
    probabilities = clf.predict_class_for_each_instance_proba(X_test)

    #######################################################################################################
    # Probing-based Similarity Calculation                                                                #
    #######################################################################################################

    # Average probabilities for each word
    word_avg_probabilities = {}
    for word in instances["word"].unique():
        word_indices = instances[instances["word"] == word].index
        word_probs = [probabilities[idx] for idx in word_indices]
        word_avg_probabilities[word] = np.mean(word_probs, axis=0)

    # Calculate similarity scores using confusion probabilities
    target_pairs_df = pd.read_csv(os.path.join(DATA_FOLDER, f'similarity/{DATASET}.csv'))
    similarity_scores = []
    target_scores = []

    print("Calculating probing-based similarities...")
    for _, row in target_pairs_df.iterrows():
        w1, w2 = row["word1"], row["word2"]
        if w1 in word_avg_probabilities and w2 in word_avg_probabilities:
            w1_probs = word_avg_probabilities[w1]
            w2_idx = list(clf.lr.classes_).index(w2)
            similarity = w1_probs[w2_idx]

            similarity_scores.append(similarity)
            target_scores.append(row["score"])

    probing_correlation = spearmanr(similarity_scores, target_scores)
    print("\nPROBING-BASED SIMILARITY RESULTS")
    print(f"Number of valid pairs: {len(similarity_scores)}")
    print(f"Spearman correlation: {probing_correlation.correlation:.3f}")
    print(f"p-value: {probing_correlation.pvalue:.3f}")

    #######################################################################################################
    # Cosine Similarity Benchmark                                                                         #
    #######################################################################################################

    print("Calculating cosine similarity benchmark...")
    cosine_aggregated_instance_embeddings = {}
    for word in instances["word"].unique():
        word_indices = instances[instances["word"] == word].index
        word_embeddings = [X_test[idx] for idx in word_indices]
        cosine_aggregated_instance_embeddings[word] = np.mean(word_embeddings, axis=0)

    cosine_scores = []
    indices_keep = []
    for _, row in target_pairs_df.iterrows():
        try:
            w1 = row["word1"]
            w2 = row["word2"]
            cosine_scores.append(
                1 - cosine(
                    cosine_aggregated_instance_embeddings[w1],
                    cosine_aggregated_instance_embeddings[w2]
                    )
                )
            indices_keep.append(True)
        except Exception as e:
            # print(f"Error processing pair {row['word1']}-{row['word2']}: {e}. Possibly missing word embeddings.")
            indices_keep.append(False)
            cosine_scores.append(0)

    cosine_scores = [val for is_good, val in zip(indices_keep, cosine_scores) if is_good]
    cosine_target_scores = [val for is_good, val in zip(indices_keep, target_pairs_df["score"]) if is_good]

    cosine_correlation = spearmanr(cosine_scores, cosine_target_scores)
    print("\nCOSINE SIMILARITY BENCHMARK RESULTS")
    print(f"Number of valid pairs: {len(cosine_scores)}")
    print(f"Spearman correlation: {cosine_correlation.correlation:.3f}")
    print(f"p-value: {cosine_correlation.pvalue:.3f}")
