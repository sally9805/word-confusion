# word-confusion
Code implementation for paper **"Rethinking Word Similarity: Semantic Similarity through Classification Confusion"**
## Benchmarking Word Confusion w/ Cosine Similarity
The code for the Benchmarking Word Confusion, as discussed in Section 2.1 of the paper, can be found in the `concept_prober/` folder.
### Installation
`pip install -e .`

`pip install -r requirements.txt`
### Download Data
- Run `python concept_prober/data/json_datasets/download_corpora.py` to download the text corpora used for training. It includes datasets for language English, French and Italian, which covers all experiments in the paper.
### Run Experiments
You can run experiments directly or generate contextual embeddings first:
- [Optional] If you wish to extract contextual words and generate embeddings separately, run `python concept_prober/runners/run_and_embed.py`.
- You could directly run similarity_experiments.py and classification_experiments.py. The embedding generation step is embedded in the experiment scripts and will be skipped if cached embeddings are available.
    - Run `python concept_prober/runners/similarity_experiments.py` for the benchmark experiment in Table 1.
    - Run `python concept_prober/runners/classification_experiments.py` for the benchmark experiment in Table 2.
## French Revolution Analysis
The code for the French Revolution analysis, as discussed in Section 4.2 of the paper, can be found in the revolution/ folder.
### Quickstart
To quickly run the code using pre-generated embeddings:
- Download the revolution folder containing the embeddings from revolution/embeddings
- Open the notebook revolution/revolution.ipynb
- Run Word Confusion >> PCA + Classifier section
### Run from scratch
To run the analysis with _Archives Parlementaires_ corpora:
- Download the data from [https://sul-philologic.stanford.edu/philologic/archparl/](https://sul-philologic.stanford.edu/philologic/archparl/).
>  Due to [Stanford Terms of Use](https://www.stanford.edu/site/terms/), we are unable to reproduce or copy the data here. However, it is publicly available via the [link](https://sul-philologic.stanford.edu/philologic/archparl/).
- Run the notebook revolution/revolution.ipynb from the beginning.

This repo contains the experiments for the paper Rethinking Word Similarity:
Semantic Similarity through Classification Confusion. If you found this repo useful, please cite

@inproceedings{zhou2025rethinksimilarity,
title={Rethinking Word Similarity:
Semantic Similarity through Classification Confusion},
author={Kaitlyn Zhou and Haishan Gao and Sarah Chen and Dan Edelstein and Dan Jurafsky and Chen Shani},
year={2025}
}