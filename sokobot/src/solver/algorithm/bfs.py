
from algorithm.algorithm import Algorithm
from state.trigger import Trigger

class BFS(Algorithm):
    def __init__(self, data):
        super().__init__(data)

    def processing(self):
        super().frontier.append(super().data)
        while super().frontier:
            toExamine = super().frontier.pop()
            if Examine.isGoal(toExamine):
                break
            else:
                super().explored.append(toExamine)
                successors = Trigger.examine(toExamine)
                super().frontier.append(successors)