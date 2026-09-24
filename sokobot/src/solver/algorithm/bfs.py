
from algorithm.algorithm import Algorithm
from state.trigger import Trigger
from examine.examine import Examine

class BFS(Algorithm):
    def __init__(self, map_data):
        super().__init__(map_data)

    def processing(self, initial_items_data):
        super().frontier.append(super().map_data)
        while super().frontier:
            toExamine = super().frontier.pop()
            if Examine.is_goal(super().map_data, toExamine):
                break
            else:
                super().explored.append(toExamine)
                successors = Examine.examine(super().map_data, toExamine)
                super().frontier.append(successors)