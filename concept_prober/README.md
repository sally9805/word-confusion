# **Data Structure**

The main data strucutre we are playing with is this JSON file in which we have seeds and instances.

* Seeds are the elements we are going to use to train the prober. Instances will be used for testing.

* For the sentiment classification task, a seed or instance can be the word "happy" and it's class will be "positive".


* Here is the file for the animal classification task. Seed are going to be "land" and "sea". While instances are going to be animals.

* We are going to find all the occurrences of "land" and "sea", get embeddings and learn to predict them with the probler and then see where things like "elephant" are projected to.



```json
    {
    "seeds": {
        "words": [
            "land",
            "sea"
        ],
        "concepts": [
            "land",
            "sea"
        ]
    },
    "instances": {
        "words": [
            "elephant",
            "spider",
            "rhino",
            [...]
            "shellfish",
            "shells",
            "lobsters"
        ],
        "concepts": [
            "land",
            "land",
            "land",
            [...]
            "sea",
            "sea",
            "sea"
        ]
    }
    }
```