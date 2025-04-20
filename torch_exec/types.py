from typing import TypedDict
from pathlib import Path
from abc import ABC, abstractmethod

class Seed(TypedDict):
    seed_path: Path
    
class Mutation(ABC):
    @abstractmethod
    def mutate(self, seed: Seed) -> Seed:
        pass

    