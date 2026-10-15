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
    def isFrozen(currentBox):
        if currentBox in explored:
            return explored[currentBox]
        if currentBox in exploring:
            return True

        savedExplored = explored.copy()
        exploring.add(currentBox)
        blockedAxes = [False, False]

        for axisIndex, axis in enumerate(AXES):
            numOfSimpleDeadlocks = 0
            adjBoxes = []

            for deltaRow, deltaCol in axis:
                dest = (currentBox[0] + deltaRow, currentBox[1] + deltaCol)
                if dest[1] < 0 or dest[1] >= width or dest[0] < 0 or dest[0] >= height:
                    continue
                if mapData[dest[0]][dest[1]] == '#':
                    blockedAxes[axisIndex] = True
                    break
                if deadLockTable[dest[0]][dest[1]]:
                    numOfSimpleDeadlocks += 1
                if dest in boxCoordinates:
                    if dest in exploring:
                        blockedAxes[axisIndex] = True
                        break
                    adjBoxes.append(dest)

            if not blockedAxes[axisIndex] and numOfSimpleDeadlocks == 2:
                blockedAxes[axisIndex] = True

            if not blockedAxes[axisIndex]:
                for adjBox in adjBoxes:
                    if isFrozen(adjBox):
                        blockedAxes[axisIndex] = True
                        break

            if not blockedAxes[axisIndex]:
                exploring.remove(currentBox)
                explored.clear()
                explored.update(savedExplored)
                explored[currentBox] = False
                return False

        exploring.remove(currentBox)
        explored[currentBox] = True
        return True

    isFrozen(boxCoordinate)
    return {box for box, frozen in explored.items() if frozen}
    


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

def movePlayer(playerCoordinate, dest, boxCoordinates, mapData, width, height, deadLockTable):
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

def reachablePlayerTiles(playerPosition, boxCoordinates, walls):
    frontier = deque([playerPosition])
    reachable = {playerPosition}
    while frontier:
        current = frontier.popleft()
        for destRow, destCol in DIRECTION:
            dest = (current[0] + destRow, current[1] + destCol)
            if dest not in reachable and dest not in walls and dest not in boxCoordinates:
                reachable.add(dest)
                frontier.append(dest)
    return reachable

def getCorralFromTile(startPos, reachable, boxCoordinates, walls, width, height):
    frontier = deque([startPos])
    corral = {startPos}
    corralBoxes = set()
    
    while frontier:
        current = frontier.popleft()
        for destRow, destCol in DIRECTION:
            dest = (current[0] + destRow, current[1] + destCol)
            
            if dest[0] < 0 or dest[0] >= height or dest[1] < 0 or dest[1] >= width:
                continue
            if dest in corral or dest in walls:
                continue
            if dest in boxCoordinates:
                corralBoxes.add(dest)
            else:
                corral.add(dest)
                frontier.append(dest)
    return corral, corralBoxes

def isPICorral(corral, corralBoxes, reachable, boxCoordinates, walls, deadLockTable):
    for box in corralBoxes:
        for destRow, destCol in DIRECTION:
            playerPos = (box[0] - destRow, box[1] - destCol)
            boxDest = (box[0] + destRow, box[1] + destCol)

            if playerPos in walls or boxDest in walls:
                continue

            if playerPos in corral:
                continue

            if playerPos in corralBoxes and boxDest in corralBoxes:
                continue

            if boxDest not in corral:
                if not deadLockTable[boxDest[0]][boxDest[1]]:
                    return False, corral, corralBoxes
                continue

            if playerPos not in reachable and playerPos not in corralBoxes:
                return False, corral, corralBoxes

    return True, corral, corralBoxes

def findWalkingPath(start, destination, boxCoordinates, walls):
    if start == destination:
        return []
    frontier = deque([start])
    parents = {start: None}
    while frontier:
        current = frontier.popleft()
        for destRow, destCol in DIRECTION:
            dest = (current[0] + destRow, current[1] + destCol)
            if dest in parents or dest in walls or dest in boxCoordinates:
                continue
            parents[dest] = (current, DIRECTIONS[(destRow, destCol)])
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

        walls = frozenset(
            (row, col)
            for row in range(height)
            for col in range(width)
            if mapData[row][col] == '#'
        )

        deadLockTable = generateSimpleDeadlock(mapData, width, height, goalState, playerPosition)
        
        distances = generateDistanceMatrices(goalState, mapData, width, height, deadLockTable)
        distanceStack = buildDistanceStack(distances, goalState)
        heuristicCache = {}
        freezeSubset = set()

        boxCoordinates = frozenset(boxCoordinates)
        interiorTiles = reachablePlayerTiles(playerPosition, frozenset(), walls)
        startReachable = reachablePlayerTiles(playerPosition, boxCoordinates, walls)
        startState = (playerPosition, boxCoordinates, startReachable)

        def stateKey(state, reachable):
            return (min(reachable), state[1])

        startKey = stateKey(startState, startReachable)
        stateByKey = {startKey: startState}
        
        parents = {startKey: None}
        explored = {startKey}
        exploredReigons = {boxCoordinates: [startReachable]}
        frontier = []
        stateCounter = 0
        heuristic = calculateHeuristic(boxCoordinates, distanceStack, goalState, heuristicCache)
        heapq.heappush(frontier, (heuristic, stateCounter, startKey))
        stateCounter += 1

        while frontier:
            _, _, currentKey = heapq.heappop(frontier)
            currentPlayerPos, currentBoxCoords, reachable = stateByKey[currentKey]

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
                    previousPlayer, previousBoxes, idc = stateByKey[previousKey]
                    walk = findWalkingPath(
                        previousPlayer, standingTile, previousBoxes, walls
                    )
                    if walk is None:
                        raise RuntimeError('Invalid reconstructed push path')
                    winPath.extend(walk)
                    winPath.append(DIRECTIONS[pushDirection])

                print(f"Total states explored: {len(explored)}")
                distinct = {key[1] for key in explored}
                print(f"Total unique box configurations explored: {len(distinct)}")
                return ''.join(winPath)

            boxesToExplore = currentBoxCoords
            exploredCorral = set()

            for currentPos in interiorTiles:
                if currentPos not in reachable and currentPos not in currentBoxCoords and currentPos not in exploredCorral:
                    corral, corralBoxes = getCorralFromTile(currentPos, reachable, currentBoxCoords, walls, width, height)
                    exploredCorral.update(corral)

                    if not corralBoxes:
                        continue

                    hasUnfinishedGoalOrBox = (
                        any(box not in goalState for box in corralBoxes) or
                        any(tile in goalState for tile in corral)
                    )
                    if not hasUnfinishedGoalOrBox:
                        continue

                    isPIC, totalCorral, totalBoxes = isPICorral(
                        corral, corralBoxes, reachable, currentBoxCoords, walls, deadLockTable
                    )
                    exploredCorral.update(totalCorral)

                    if isPIC:
                        boxesToExplore = totalBoxes
                        break

            statesMade = 0
            for box in boxesToExplore:
                for destRow, destCol in DIRECTION:
                    standingTile = (box[0] - destRow, box[1] - destCol)
                    boxDestination = (box[0] + destRow, box[1] + destCol)

                    #If we cant reach the position of the player after pushing the box, cant push so prune
                    if standingTile not in reachable:
                        continue

                    if boxDestination in walls or boxDestination in currentBoxCoords:
                        continue

                    newState = movePlayer( standingTile, box, currentBoxCoords, mapData, width,
                                           height, deadLockTable)

                    #If newstate is for any reason none (simpledeadlock, frozen box, etc) then prune
                    if newState is None:
                        continue

                    newPlayerPos, newBoxCoords = newState


                    alreadyExplored = False
                    #if the boxes positions have been seen before
                    if newBoxCoords in exploredReigons:
                        #if the player is in the same "room/reigon" then its the exact same state, so prune 
                        for exploredRegion in exploredReigons[newBoxCoords]:
                            if newPlayerPos in exploredRegion:
                                alreadyExplored = True
                                break
                    #same as above
                    if alreadyExplored:
                        continue

                    heuristic = calculateHeuristic(newState[1], distanceStack, goalState, heuristicCache)

                    if heuristic >= 999999:
                        continue

                    if hasFreezeSubset(newBoxCoords, mapData, width, height, freezeSubset):
                        continue

                    freezeExplored = dict()
                    exploring = set()
                    frozenBoxes = detectBlockedAxes(boxDestination, mapData, width, height, deadLockTable, freezeExplored, newBoxCoords, exploring)
                    if boxDestination in frozenBoxes and any(b not in goalState for b in frozenBoxes):
                        if len(frozenBoxes) <= 4:
                            freezeSubset.add(frozenset(frozenBoxes))
                        continue

                    newReachable = reachablePlayerTiles(newState[0], newState[1], walls)

                    #if the boxes positions have not been seen before
                    if newBoxCoords not in exploredReigons:
                        exploredReigons[newBoxCoords] = []
                    #add the new reachable region to the list of explored regions for this box configuration
                    exploredReigons[newBoxCoords].append(newReachable)

                    newState = (newState[0], newState[1], newReachable)
                    newKey = stateKey(newState, newReachable)
  
                    explored.add(newKey)
                    parents[newKey] = (currentKey, standingTile, (destRow, destCol))
                    stateByKey[newKey] = newState

                    heapq.heappush(frontier, (heuristic, stateCounter, newKey))
                    stateCounter += 1
                    statesMade += 1

        return ""
