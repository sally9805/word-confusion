from dataclasses import dataclass
import json
from typing import Callable

# Not the best pythonic way to do this, but it's a quick and dirty way to do it
@dataclass
class SeedOrInstance:
    word: str
    concept: str

@dataclass
class Seed(SeedOrInstance):
    pass

@dataclass
class Instance(SeedOrInstance):
    pass

@dataclass
class Data:
    seeds: list[Seed]
    instances: list[Instance]

    # these two properties might be useful for some experiments

    @property
    def seed_to_concept(self) -> Callable[[str], str]:
        seed_dict: dict[str, str] = {seed.word: seed.concept for seed in self.seeds}
        return lambda x: seed_dict[x]
    
    @property
    def instance_to_concept(self) -> Callable[[str], str]:
        instance_dict: dict[str, str] = {instance.word: instance.concept for instance in self.instances}
        return lambda x: instance_dict[x]


def load_data_from_json(file_path: str) -> Data:
    with open(file_path, "r") as file:
        raw_data = json.load(file)
    
    # Transform the nested structure into list of Seed objects
    seeds = [
        Seed(word=word, concept=concept)
        for word, concept in zip(raw_data["seeds"]["words"], raw_data["seeds"]["concepts"])
    ]
    
    # Transform the nested structure into list of Instance objects
    instances = [
        Instance(word=word, concept=concept)
        for word, concept in zip(raw_data["instances"]["words"], raw_data["instances"]["concepts"])
    ]
    
    return Data(seeds=seeds, instances=instances)
