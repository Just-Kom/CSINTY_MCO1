import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque

def getHeuristic(boxCoordinates, goalState):
    boxRows = np.array(boxCoordinates)
    goalRows = np.array(goalState)

    diff = np.abs(boxRows[:, None, :] - goalRows[None, :, :])
    costMatrix = diff.sum(axis=2)

    row, col = linear_sum_assignment(costMatrix)

    return costMatrix[row, col].sum()


def getDirection(playerFrom, playerTo, directions):
    diffRow = playerTo[0] - playerFrom[0]
    diffCol = playerTo[1] - playerFrom[1]
    for direction, (dRow, dCol) in directions.items():
        if (diffRow, diffCol) == (dRow, dCol):
            return direction
    return None

def isGoalState(boxCoordinates, goalState):
    return set(boxCoordinates) == set(goalState)

def isValidMove(dest, boxCoordinates, mapData):
    row, col = dest
    if row < 0 or row >= len(mapData) or col < 0 or col >= len(mapData[0]):
        return False
    if mapData[row][col] == '#':
        return False
    if dest in boxCoordinates:
        return False
    return True

# move player will create a new state based on the current state and the direction of movement. It will return None if the move is invalid, otherwise it will return a new state with the updated player and box coordinates. The new state is a tuple of (playerCoordinates, boxCoordinates)
def movePlayer(playerCoordinates, direction, boxCoordinates, mapData):
    row, col = playerCoordinates
    diffRow, diffCol = direction
    # if the player is trying to move into a box, check if the box can be pushed, so row + diffRow, col + diffCol is the current position of the box, and row + 2 * diffRow, col + 2 * diffCol is the position where the box will be pushed to
    if (row + diffRow, col + diffCol) in boxCoordinates:
        newBoxRow, newBoxCol = row + 2 * diffRow, col + 2 * diffCol
        if not isValidMove((newBoxRow, newBoxCol), boxCoordinates, mapData):
            return None
        boxCoordinates = frozenset(newBox if newBox != (row + diffRow, col + diffCol) else (newBoxRow, newBoxCol) for newBox in boxCoordinates)
        return ((row + diffRow, col + diffCol), boxCoordinates)
    else:
        if not isValidMove((row + diffRow, col + diffCol), boxCoordinates, mapData):
            return None
        return ((row + diffRow, col + diffCol), boxCoordinates)

class SokoBot:
    def solveSokobanPuzzle(self, width, height, mapData, itemsData):
        # YOU NEED TO REWRITE THE IMPLEMENTATION OF THIS METHOD TO MAKE THE BOT SMARTER
        # Default stupid behavior: Think (sleep) for 3 seconds, and then return a
        # sequence
        # that just moves left and right repeatedly.

        #build a state first
        playerCoordinates = ()
        boxCoordinates = []
        goalState = []
        for rowIndex, row in enumerate(itemsData):
            for col, char in enumerate(row):
                match char:
                    case '@':
                        playerCoordinates = (rowIndex, col)
                    case '$':
                        boxCoordinates.append((rowIndex, col))

        for rowIndex, row in enumerate(mapData):
            for col, char in enumerate(row):
                if char == '.':
                    goalState.append((rowIndex, col))

        goalState = tuple(goalState)
        startState = (playerCoordinates, frozenset(boxCoordinates))
        #A single state is a tuple of 2 elements of playerCoordinates and boxCoordinates.
        directions = {
        'u': (-1, 0),   # Up 
        'd': (1, 0),  # Down 
        'l': (0, -1),  # Left
        'r': (0, 1)    # Right
        }

        frontier = deque([startState])
        explored = {startState}
        roadToSucess = {startState: None}

        while frontier:
            currentState = frontier.popleft()
            if isGoalState(currentState[1], goalState):
                path = []
                while roadToSucess[currentState] is not None:
                    path.append(getDirection(roadToSucess[currentState][0], currentState[0], directions) if roadToSucess[currentState] is not None else None)
                    currentState = roadToSucess[currentState]
                path.reverse()
                return path

            for destRow, destCol in directions.values():
                newState = movePlayer(currentState[0], (destRow, destCol), currentState[1], mapData)
                if newState is not None and newState not in explored:
                    explored.add(newState)
                    frontier.append(newState)
                    roadToSucess[newState] = currentState
        return None



            