import time
import numpy as np
import heapq
from scipy.optimize import linear_sum_assignment
from collections import deque
from collections import ChainMap

"""
dest - Target coordinate on the grid
Type: tuple (row, col)

dest[0] - Row index (vertical, top is 0)
Type: int
Note: UP AND DOWN MOVEMENT

dest[1] - Column index (horizontal, left is 0)
Type: int
Note: LEFT AND RIGHT MOVEMENT

playerCoordinate - Current location of the player
Type: tuple (row, col)

boxCoordinates - Current locations of all boxes
Type: frozenset of tuples

goalState - Winning target locations for boxes
Type: frozenset of tuples

goalState[i] - Cant index, but each element is a tuple (row, col) representing a goal location
Type: N/A
Note: DO NOT INDEX

deltaX - Row difference between dest and position
Type: int

deltaY - Column difference between dest and position
Type: int

DIRECTION - The 4 possible adjacent moves (row_change, col_change)
Type: list of tuples
Note: Used for iteration and expansion

DIRECTIONS - Mapping of coordinate changes to output characters ('u', 'd', 'l', 'r')
Type: dict
Note: Used for outputting the solution path, for O(1) lookup
"""

DIRECTIONS = {
    (0, -1): 'l',
    (0, 1): 'r',
    (1, 0): 'd',
    (-1, 0):'u' 
}

DIRECTION = [
    (0, -1),
    (0, 1),
    (1, 0),
    (-1, 0)
]

AXES = [
    [(0, -1), (0, 1)],  # X-axis (left and right)
    [(1, 0), (-1, 0)]   # Y-axis (down and up)
]

def hasFreezeSubset(boxCoordinates, mapData, width, height, freezeSubset):
    # freezeSubset is a set of boxes that are frozen, we want to check if any of the boxes in boxCoordinates are in freezeSubset
    for box in freezeSubset:
        if box.issubset(boxCoordinates):
            return True
    return False

def detectBlockedAxes(boxCoordinate, mapData, width, height, deadLockTable, explored, boxCoordinates, exploring):
    # Turn the boxcoords into a dict, each boxcoord value is (False,False) where (Y-axis, X-Axis)
    # iboxcoordinate is just 1 coordinate, the a pushed box, were just checking its surroundings to check if its a frozen box.
    # explored is the set of all boxes that have been checked for frozen axes, so we dont check them again.
    # axes = [DIRECTION[2:], DIRECTION[:2]
    # axisIndex 0 is the Y axis, axisIndex 1 is the X axis

    currentBox = boxCoordinate
    exploring.add(currentBox)
    explored[currentBox] = [False, False]
    # the axes is just a list of 2 lists, each list contains the 2 directions of a single axis.
    # a single axis looks like this  (0, -1), (0, 1) which represents the left and right of the box at the x axis.

    # so this is saying, foe each axis
    for axisIndex, axis in enumerate(AXES):
        numOfSimpleDeadlocks = 0
        #check the left and right
        for __,leftRight in enumerate(axis):
            #left right is a tuple (row, col)
            dest = (currentBox[0]+leftRight[0], currentBox[1]+leftRight[1])
            # if dest is out of bounds then just continue to the next direction
            if dest[1] < 0 or dest[1] >= width or dest[0] < 0 or dest[0] >= height:
                continue
            if mapData[dest[0]][dest[1]] == '#':
                # Is there a wall on the left or on the right side?
                explored[currentBox][axisIndex] = True
                #. Is there a simple deadlock square on the right and the left side? 
            if deadLockTable[dest[0]][dest[1]]:
                numOfSimpleDeadlocks += 1
                #Is there a box on the left or on the right side, which is already blocked?
            if dest in boxCoordinates:
                if dest not in explored:
                    detectBlockedAxes(dest, mapData, width, height, deadLockTable, explored, boxCoordinates, exploring)
                    # if the box on the left or right side is blocked, then the current box is also blocked on that axis
                    # if both axes are blocked, then the current box is also blocked
                if explored[dest][1- axisIndex] or dest in exploring:
                    explored[currentBox][axisIndex] = True
            # return all of the boxes that are blocked on both axes, which means they are frozen and cannot be moved anymore
        if numOfSimpleDeadlocks == 2:
            explored[currentBox][axisIndex] = True
    exploring.remove(currentBox)
    return {box for box, blocked in explored.items() if blocked[0] and blocked[1]}
    


def isValid(dest, mapData, width, height):
    if dest[1] < 0 or dest[1] >= width or dest[0] < 0 or dest[0] >= height:
        return False
    if mapData[dest[0]][dest[1]] == '#':
        return False
    return True

def isValidPull(dest, mapData, width, height):
    # dest is the pos the player is moving into
    # new dest is the pos the box is moving into
    # a pull can only happen if the box and the player is still on the same column
    # or if they are still on the same row
    if dest[1] < 0 or dest[1] >= width or dest[0] < 0 or dest[0] >= height:
        return False
    if mapData[dest[0]][dest[1]] == '#':
        return False
    return True

def movePlayerPull(playerCoordinate, dest, boxCoordinates, mapData, width, height):
    deltaRow = dest[0] - playerCoordinate[0]
    deltaCol = dest[1] - playerCoordinate[1]
    # (deltaRow, deltaCol) is the direction the player is moving in
    
    # the box must move in the SAME direction as the player
    deltaRowBox = playerCoordinate[0] - boxCoordinates[0]
    deltaColBox = playerCoordinate[1] - boxCoordinates[1]
    
    # Only pull if the box and the player are moving in the same direction
    if (deltaRow == deltaRowBox) and (deltaCol == deltaColBox):
        if isValidPull(dest, mapData, width, height):
            newPlayerCoordinate = dest
            newBoxCoordinates = playerCoordinate
            return (newPlayerCoordinate, newBoxCoordinates)
            
    return None


def generateSimpleDeadlock(mapData, width, height, goalState, playerPos):
    deadLockLookUp = np.full((height, width), True)
    #Check if a box can be pulled from a goalstate to a certain tile
    #BFS ALGO
    #explored is now the deadLockLookUP
    frontier = deque()
    explored = set()
    for Rowindex, goal in enumerate(goalState):
        for Colindex, direction in enumerate(DIRECTION):
            dest = (goal[0] + direction[0], goal[1] + direction[1])
            if isValid(dest, mapData, width, height):
                playerPosition = dest
                boxPosition = goal
                startState = (playerPosition, boxPosition)
                frontier.append(startState)
                explored.add(startState)

    #we are trying to get the player to pull the 
        while frontier:
            currentState = frontier.popleft()
            currentPlayerPos = currentState[0]
            currentBoxCoords = currentState[1]
            deadLockLookUp[currentBoxCoords[0]][currentBoxCoords[1]] = False
            
            # for each direction from the box, put the player in that direction and check if its valid
            deltaRow =  currentPlayerPos[0] - currentBoxCoords[0]
            deltaCol = currentPlayerPos[1] - currentBoxCoords[1]
            # delta row and col is the direction of the player relative to the box
            # we want to move the player 1 backwards
            # the code below is the new state after moving the box
            row = deltaRow
            col = deltaCol
            dest = (currentPlayerPos[0] + row, currentPlayerPos[1] + col)
            newState = movePlayerPull(currentPlayerPos, dest, currentBoxCoords, mapData, width, height)
            if newState is not None and newState not in explored:
                explored.add(newState)
                frontier.append(newState)
            # the code below states that, in the current NEW coordinate of the box, we move the player to the 4 adjacent tiles around the box and check if its valid
            for Colindex, direction in enumerate(DIRECTION):
                destination = (currentBoxCoords[0] + direction[0], currentBoxCoords[1] + direction[1])
                if isValid(destination, mapData, width, height):
                    playerPosition = destination
                    startState = (playerPosition, currentBoxCoords)
                    if startState not in explored:
                        frontier.append(startState)
                        explored.add(startState)
                    
    return deadLockLookUp

def isValidMove(dest, boxCoordinates, mapData, width, height):
    if dest[1] < 0 or dest[1] >= width or dest[0] < 0 or dest[0] >= height:
        return False
    if mapData[dest[0]][dest[1]] == '#':
        return False
    if dest in boxCoordinates:
        return False
    return True

def movePlayer(playerCoordinate, dest, boxCoordinates, mapData, width, height, deadLockTable, freezeSubset, goalState):
    deltaX = dest[0] - playerCoordinate[0]
    deltaY = dest[1] - playerCoordinate[1]
    newDest = (dest[0] + deltaX, dest[1] + deltaY)

    # if player is pushing a box
    if dest in boxCoordinates:
        # if pushing into a wall or something, prune
        if not isValidMove(newDest, boxCoordinates, mapData, width, height):
            return None
        # if moving into a simple deadlock, prune
        if deadLockTable[newDest[0]][newDest[1]]:
            return None

        # hypothetical situation where we already pushed the box
        newBoxCoordinates = frozenset(boxCoordinates - {dest} | {newDest})

        
        # if we already know that its a frozen box, then prune
        if hasFreezeSubset(newBoxCoordinates, mapData, width, height, freezeSubset):
            return None

        # else, check if the move we just did causes a frozen box, if it does, then add it to the freezeSubset and prune
        freezeExplored = dict()
        exploring = set()
        frozenBoxes = detectBlockedAxes(newDest, mapData, width, height, deadLockTable, freezeExplored, newBoxCoordinates, exploring)
        # if the box is frozen, add it to the list of patterns then prune
        if newDest in frozenBoxes:
            if any(box not in goalState for box in frozenBoxes):
                freezeSubset.append(frozenset(frozenBoxes))
                return None 
        # else, the box is not frozen, so continue
        
        return (dest, newBoxCoordinates)
    # else the player is just moving to an empty space
    elif isValidMove(dest, boxCoordinates, mapData, width, height):
        return (dest, boxCoordinates)
    # if all else fails, return None
    return None

def generateDistanceMatrices(goalState, mapData, width, height, deadLockTable):
    #create goalState many matrices in a dict
    matrices = {}

    for goalIndex, goalVal in enumerate(goalState):
        matrix = np.full((height, width), -1)

        startState = (goalVal, 0)
        matrix[goalVal[0]][goalVal[1]] = 0
        frontier = deque()
        frontier.append(startState)
        explored = {goalVal}

        while frontier:
            currentState = frontier.popleft()
            currentPos = currentState[0]
            currentCtr = currentState[1]
            for index,value in enumerate(DIRECTION):
                row = value[0]
                col = value[1]
                dest = (currentPos[0] + row, currentPos[1] + col)
                #def detectBlockedAxes(boxCoordinate, mapData, width, height, deadLockTable, explored, axes, boxCoordinates):
                if isValid(dest, mapData, width, height) and deadLockTable[dest[0]][dest[1]] == False:
                    if dest not in explored:
                        explored.add(dest)
                        newCtr = currentCtr + 1
                        frontier.append((dest, newCtr))
                        matrix[dest[0]][dest[1]] = newCtr
        matrices[goalVal] = matrix
    return matrices
        #expand from the goalVal to the map     

def buildDistanceStack(distanceMatrices, goalState):

    distanceStack = np.stack([distanceMatrices[goalVal] for goalVal in goalState])
    return np.where(distanceStack == -1, 999999, distanceStack)

def calculateHeuristic(boxCoordinates, distanceMatrices, goalState, heuristicCache):
    if boxCoordinates in heuristicCache:
        return heuristicCache[boxCoordinates]

    boxRows = [boxVal[0] for boxVal in boxCoordinates]
    boxCols = [boxVal[1] for boxVal in boxCoordinates]
    costMatrix = distanceMatrices[:, boxRows, boxCols].T
    row, col = linear_sum_assignment(costMatrix)
    heuristic = int(costMatrix[row, col].sum())
    heuristicCache[boxCoordinates] = heuristic
    return heuristic

def countRegions(mapData, width, height, boxes):
    explored = set()
    rooms = 0

    #For every single node in the map
    for row in range(height):
        for col in range(width):
            currentPos = (row, col)
            
            #If its a wall, a box, or already explored, then skip it
            if not isValid(currentPos, mapData, width, height) or currentPos in boxes or currentPos in explored:
                continue

            #Else, new "room" so basically fill the reigon then increment our room counter
            rooms += 1
            
            #everything below is just standard flood fill bfs
            frontier = [currentPos]
            explored.add(currentPos)

            while frontier:
                currentRow, currentCol = frontier.pop(0)

                for rowY, colX in DIRECTION:
                    dest = (currentRow + rowY, currentCol + colX)
                    
                    if isValid(dest, mapData, width, height) and dest not in boxes and dest not in explored:
                        explored.add(dest)
                        frontier.append(dest)
                        
    # Return the number of rooms found
    return rooms

def reachablePlayerTiles(playerPosition, boxCoordinates, mapData, width, height):
    #BFS of all tiles the player can walk to without pushing a box.
    frontier = deque([playerPosition])
    reachable = {playerPosition}
    while frontier:
        current = frontier.popleft()
        for dr, dc in DIRECTION:
            dest = (current[0] + dr, current[1] + dc)
            if dest not in reachable and isValidMove(dest, boxCoordinates, mapData, width, height):
                reachable.add(dest)
                frontier.append(dest)
    return reachable


def findWalkingPath(start, destination, boxCoordinates, mapData, width, height):
    #Find the shortest sequence of walking moves with the boxes fixed.
    if start == destination:
        return []
    frontier = deque([start])
    parents = {start: None}
    while frontier:
        current = frontier.popleft()
        for dr, dc in DIRECTION:
            dest = (current[0] + dr, current[1] + dc)
            if dest in parents or not isValidMove(dest, boxCoordinates, mapData, width, height):
                continue
            parents[dest] = (current, DIRECTIONS[(dr, dc)])
            if dest == destination:
                path = []
                while dest != start:
                    previous, move = parents[dest]
                    path.append(move)
                    dest = previous
                return path[::-1]
            frontier.append(dest)
    return None


class SokoBot:
    def solveSokobanPuzzle(self, width, height, mapData, itemsData):
        goalState = []
        for row,rowVal in enumerate(mapData):
            for col,colVal in enumerate(rowVal):
                if colVal == '.':
                    goalState.append((row, col))
        goalState = frozenset(goalState)

        boxCoordinates = []
        for row,rowVal in enumerate(itemsData):
            for col,colVal in enumerate(rowVal):
                if colVal == '$':
                    boxCoordinates.append((row, col))
                elif colVal == '@':
                    playerPosition = (row, col)

        deadLockTable = generateSimpleDeadlock(mapData, width, height, goalState, playerPosition)
        
        distances = generateDistanceMatrices(goalState, mapData, width, height, deadLockTable)
        distanceStack = buildDistanceStack(distances, goalState)
        heuristicCache = {}
        freezeSubset = []

        boxCoordinates = frozenset(boxCoordinates)
        startState = (playerPosition, boxCoordinates)
        startReachable = reachablePlayerTiles(playerPosition, boxCoordinates, mapData, width, height)

        def stateKey(state, reachable):
            return (min(reachable), state[1])

        startKey = stateKey(startState, startReachable)
        stateByKey = {startKey: startState}
        
        parents = {startKey: None}
        explored = {startKey}
        frontier = []
        stateCounter = 0
        heuristic = calculateHeuristic(boxCoordinates, distanceStack, goalState, heuristicCache)
        heapq.heappush(frontier, (heuristic, stateCounter, startKey))
        stateCounter += 1

        while frontier:
            _, _, currentKey = heapq.heappop(frontier)
            currentPlayerPos, currentBoxCoords = stateByKey[currentKey]

            if currentBoxCoords == goalState:
                transitions = []
                key = currentKey
                while parents[key] is not None:
                    previousKey, standingTile, pushDirection = parents[key]
                    transitions.append((previousKey, standingTile, pushDirection))
                    key = previousKey
                transitions.reverse()

                winPath = []
                for previousKey, standingTile, pushDirection in transitions:
                    previousPlayer, previousBoxes = stateByKey[previousKey]
                    walk = findWalkingPath(
                        previousPlayer, standingTile, previousBoxes, mapData, width, height
                    )
                    if walk is None:
                        raise RuntimeError('Invalid reconstructed push path')
                    winPath.extend(walk)
                    winPath.append(DIRECTIONS[pushDirection])

                print(f"Total states explored: {len(explored)}")
                distinct = {key[1] for key in explored}
                print(f"Total unique box configurations explored: {len(distinct)}")
                return ''.join(winPath)

            reachable = reachablePlayerTiles(
                currentPlayerPos, currentBoxCoords, mapData, width, height
            )

            for box in currentBoxCoords:
                for dr, dc in DIRECTION:
                    standingTile = (box[0] - dr, box[1] - dc)
                    boxDestination = (box[0] + dr, box[1] + dc)
                    if standingTile not in reachable:
                        continue
                    if not isValidMove(boxDestination, currentBoxCoords, mapData, width, height):
                        continue
                    newState = movePlayer( standingTile, box, currentBoxCoords, mapData, width, height, deadLockTable, freezeSubset, goalState)
                    if newState is None:
                        continue
                    newReachable = reachablePlayerTiles(newState[0], newState[1], mapData, width, height)
                    newKey = stateKey(newState, newReachable)
                    if newKey in explored:
                        continue
                    explored.add(newKey)
                    parents[newKey] = (currentKey, standingTile, (dr, dc))
                    stateByKey[newKey] = newState
                    heuristic = calculateHeuristic(newState[1], distanceStack, goalState, heuristicCache)
                    heapq.heappush(frontier, (heuristic, stateCounter, newKey))
                    stateCounter += 1

        return []
