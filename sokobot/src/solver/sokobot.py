import time
from queue import LifoQueue
from collections import deque

from src.solver.algorithm import algorithm

class SokoBot:
    def solveSokobanPuzzle(self, width, height, mapData, itemsData):
        # YOU NEED TO REWRITE THE IMPLEMENTATION OF THIS METHOD TO MAKE THE BOT SMARTER
        # Default stupid behavior: Think (sleep) for 3 seconds, and then return a
        # sequence
        # that just moves left and right repeatedly.
        stack = []

        frontier = deque()
        explored = deque()
        toExamine = itemsData
        result = ""

        #goal = to be decided
        #examine

        try:
            time.sleep(3)
            result = algorithm.bfs(frontier, explored)
        except Exception as ex:
            print(ex)
        return "lrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlr"