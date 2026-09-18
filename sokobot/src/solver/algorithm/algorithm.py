from abc import ABC, abstractmethod

class Algorithm(ABC):
    def __init__(self, data):
        self.data = data
        self.frontier = []
        self.explored = []

    @abstractmethod
    def processing(self):
        pass



