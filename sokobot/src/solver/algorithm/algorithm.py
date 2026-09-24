from abc import ABC, abstractmethod

class Algorithm(ABC):
    def __init__(self, map_data):
        self.map_data = map_data
        self.frontier = []
        self.explored = []

    @abstractmethod
    def processing(self, initial_items_data):
        pass



