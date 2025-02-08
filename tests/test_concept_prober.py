from concept_prober.src.components.occurrences_extraction import FindWordTextOccurrence
import shutil
from concept_prober.src.components.embedder import Embedder
from scipy.spatial.distance import cosine



def test_extractor():
    matcher = FindWordTextOccurrence()

    matcher.extract(["cat"], ["the cat is on the table", "a cat in my lap"], "temp_name/exp.tsv")



def test_embedder():
    target_texts = ["this was a very nice play and movie to see at the theather", "my dog barks at night",
                    "they play the movie in theather"]
    words = [" play", " dog", " play"]

    embi = Embedder("bert-base-uncased", 100, "cuda")
    embeddings = embi.embed(target_texts, words, [12], 2)

    assert (1 - cosine(embeddings[0], embeddings[2])) > (1 - cosine(embeddings[0], embeddings[1]))