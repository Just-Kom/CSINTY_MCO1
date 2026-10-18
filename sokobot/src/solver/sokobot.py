import time
import numpy as np
import heapq
from scipy.optimize import linear_sum_assignment
from collections import deque
from collections import ChainMap
from collections import defaultdict

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

def hasFreezeSubset(boxDestination, boxCoordinates, freezeSubset):
    # freezeSubset maps a tile to every frozen set of boxes that has a box on that tile
    # the parent state wasnt frozen, so any new freeze has to include the box we just pushed
    for subset in freezeSubset.get(boxDestination, ()):
        if subset.issubset(boxCoordinates):
            return True
    return False

def detectBlockedAxes(boxCoordinate, mapData, width, height, deadLockTable, explored, boxCoordinates, exploring):
    def isFrozen(currentBox):
        if currentBox in explored:
            return explored[currentBox]
        if currentBox in exploring:
            return True

        blockedAxes = [False, False]
        adjBoxesPerAxis = [[], []]

        #check the walls, simple deadlocks and boxes on both axes first before recursing into anything
        for axisIndex, axis in enumerate(AXES):
            numOfSimpleDeadlocks = 0

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
                    adjBoxesPerAxis[axisIndex].append(dest)

            if not blockedAxes[axisIndex] and numOfSimpleDeadlocks == 2:
                blockedAxes[axisIndex] = True

            #if this axis is open and theres no box beside it, the box can still move so its not frozen
            if not blockedAxes[axisIndex] and not adjBoxesPerAxis[axisIndex]:
                explored[currentBox] = False
                return False

        savedExplored = explored.copy()
        exploring.add(currentBox)

        for axisIndex, adjBoxes in enumerate(adjBoxesPerAxis):
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
    if abs(deltaRowBox) + abs(deltaColBox) == 1 and (deltaRow == deltaRowBox) and (deltaCol == deltaColBox):
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
    for rowIndex, goal in enumerate(goalState):
        for colIndex, direction in enumerate(DIRECTION):
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
            #move the player to the 4 adjacent tiles from where they are standing and check if its valid
            for Colindex, direction in enumerate(DIRECTION):
                destination = (currentPlayerPos[0] + direction[0], currentPlayerPos[1] + direction[1])
                if destination != currentBoxCoords and isValid(destination, mapData, width, height):
                    playerPosition = destination
                    startState = (playerPosition, currentBoxCoords)
                    if startState not in explored:
                        frontier.append(startState)
                        explored.add(startState)
                    
    return deadLockLookUp

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
                #the player has to stand behind dest to push the box back to currentPos, so that cant be a wall
                playerStand = (dest[0] + row, dest[1] + col)
                #def detectBlockedAxes(boxCoordinate, mapData, width, height, deadLockTable, explored, axes, boxCoordinates):
                if isValid(dest, mapData, width, height) and deadLockTable[dest[0]][dest[1]] == False and isValid(playerStand, mapData, width, height):
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

def calculateHeuristic(boxCoordinates, distanceMatrices, goalState, heuristicCache, hotspotDictionary):
    if boxCoordinates in heuristicCache:
        return heuristicCache[boxCoordinates]

    boxRows = [boxVal[0] for boxVal in boxCoordinates]
    boxCols = [boxVal[1] for boxVal in boxCoordinates]
    costMatrix = distanceMatrices[:, boxRows, boxCols].T
    row, col = linear_sum_assignment(costMatrix)
    hotspots = countHotspots(boxCoordinates, hotspotDictionary)
    heuristic = int(costMatrix[row, col].sum()) + hotspots * 10
    heuristicCache[boxCoordinates] = heuristic
    return heuristic

def reachablePlayerTiles(playerPosition, boxCoordinates, neighbors):
    frontier = [playerPosition]
    reachable = {playerPosition}
    minimumCell = playerPosition
    #looping over a list that keeps growing is faster than popleft on a deque
    for current in frontier:
        for dest in neighbors[current]:
            if dest not in reachable and dest not in boxCoordinates:
                reachable.add(dest)
                frontier.append(dest)
                if dest < minimumCell:
                    minimumCell = dest
    return reachable, minimumCell

def getCorralFromTile(startPos, boxCoordinates, neighbors):
    frontier = [startPos]
    corral = {startPos}
    corralBoxes = set()
    
    for current in frontier:
        for dest in neighbors[current]:
            if dest in corral:
                continue
            if dest in boxCoordinates:
                corralBoxes.add(dest)
            else:
                corral.add(dest)
                frontier.append(dest)
    return corral, corralBoxes

def isPICorral(corral, corralBoxes, reachable, boxCoordinates, walls, validBoxTiles):
    hasValidPush = False
    for box in corralBoxes:
        for destRow, destCol in DIRECTION:
            playerPos = (box[0] - destRow, box[1] - destCol)
            boxDest = (box[0] + destRow, box[1] + destCol)

            if playerPos in walls or boxDest not in validBoxTiles:
                continue

            if playerPos in corral:
                continue

            if playerPos in corralBoxes and boxDest in corralBoxes:
                continue

            if boxDest not in corral:
                return False, corral, corralBoxes

            if playerPos not in reachable and playerPos not in corralBoxes:
                return False, corral, corralBoxes

            if playerPos in reachable and boxDest not in boxCoordinates:
                hasValidPush = True

    return hasValidPush, corral, corralBoxes

def findWalkingPath(start, destination, boxCoordinates, neighbors):
    if start == destination:
        return []
    frontier = deque([start])
    parents = {start: None}
    while frontier:
        current = frontier.popleft()
        for dest in neighbors[current]:
            if dest in parents or dest in boxCoordinates:
                continue
            destRow = dest[0] - current[0]
            destCol = dest[1] - current[1]
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

def detectTunnelSquares(interiorTiles, goalState, walls):
    tunnelSquares = set()
    for tile in interiorTiles:
        for deltaRow, deltaCol in DIRECTION:
            prevTile = (tile[0] - deltaRow, tile[1] - deltaCol)
            sideRow, sideCol = -deltaCol, deltaRow
            tileSides = (
                (tile[0] + sideRow, tile[1] + sideCol),
                (tile[0] - sideRow, tile[1] - sideCol)
            )
            prevSides = (
                (prevTile[0] + sideRow, prevTile[1] + sideCol),
                (prevTile[0] - sideRow, prevTile[1] - sideCol)
            )
            if prevTile in interiorTiles and tile not in goalState and all(side in walls for side in tileSides) and all(side in walls for side in prevSides):
                tunnelSquares.add((tile, (deltaRow, deltaCol)))
    return tunnelSquares

def generateHotspotTable(goalState, interiorTiles, validBoxTiles, neighbors):
    def getReachableGoalsCount(hotspotTile):
        counts = {tile: 0 for tile in interiorTiles}
        goalReached = {}
        
        for rowIndex, goal in enumerate(goalState):
            if goal == hotspotTile:
                continue
                
            #Check if a box can be pulled from a goalstate to a certain tile
            #BFS ALGO
            #explored is now tracking the player and box states for this specific target
            frontier = []
            explored = set()
            reached = set()
            
            for dest in neighbors[goal]:
                if dest != hotspotTile:
                    playerPosition = dest
                    boxPosition = goal
                    startState = (playerPosition, boxPosition)
                    frontier.append(startState)
                    explored.add(startState)

            #we are trying to get the player to pull the 
            for currentPlayerPos, currentBoxCoords in frontier:
                if currentBoxCoords not in reached:
                    reached.add(currentBoxCoords)
                    counts[currentBoxCoords] += 1
                
                # for each direction from the box, put the player in that direction and check if its valid
                deltaRow =  currentPlayerPos[0] - currentBoxCoords[0]
                deltaCol = currentPlayerPos[1] - currentBoxCoords[1]
                # delta row and col is the direction of the player relative to the box
                # we want to move the player 1 backwards
                # the code below is the new state after moving the box
                if abs(deltaRow) + abs(deltaCol) == 1:
                    dest = (currentPlayerPos[0] + deltaRow, currentPlayerPos[1] + deltaCol)
                    if dest in neighbors[currentPlayerPos] and dest != hotspotTile:
                        newState = (dest, currentPlayerPos)
                        if newState not in explored:
                            explored.add(newState)
                            frontier.append(newState)
                #move the player to the 4 adjacent tiles from where they are standing and check if its valid
                for destination in neighbors[currentPlayerPos]:
                    if destination != currentBoxCoords and destination != hotspotTile:
                        startState = (destination, currentBoxCoords)
                        if startState not in explored:
                            frontier.append(startState)
                            explored.add(startState)

            goalReached[goal] = reached
                            
        return counts, goalReached

    # get the number of reachable goals if there are no hotspots
    reachableGoalCount, goalReached = getReachableGoalsCount(None)
    
    #map a tuple coordinate acting as a hotspot to a set of trapped tuple coordinates
    hotspotDictionary = {tile: set() for tile in interiorTiles}
    
    #test every valid floor tile as an immovable hotspot
    for hotspotTile in validBoxTiles:
        hotspotGoalCount, _ = getReachableGoalsCount(hotspotTile)
        
        for tile in validBoxTiles:
            if tile == hotspotTile:
                continue
            expectedCount = reachableGoalCount[tile]
            if hotspotTile in goalReached and tile in goalReached[hotspotTile]:
                expectedCount -= 1
            #if the number of reachable goals dropped, the hotspot tile is blocking boxes on this tile from getting into our goal room
            if hotspotGoalCount[tile] < expectedCount:
                hotspotDictionary[hotspotTile].add(tile)
                
    return hotspotDictionary

def countHotspots(boxCoordinates, hotspotDictionary):
    hotspots = 0
    for box in boxCoordinates:
        if not hotspotDictionary[box].isdisjoint(boxCoordinates):
            hotspots += 1
    return hotspots

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
        freezeSubset = defaultdict(list)

        #walls never move, so get the neighbors of every floor tile once instead of every bfs step
        neighbors = {}
        for row in range(height):
            for col in range(width):
                if (row, col) in walls:
                    continue
                neighbors[(row, col)] = tuple(
                    (row + destRow, col + destCol)
                    for destRow, destCol in DIRECTION
                    if 0 <= row + destRow < height and 0 <= col + destCol < width
                    and (row + destRow, col + destCol) not in walls
                )

        boxCoordinates = frozenset(boxCoordinates)
        interiorTiles, _ = reachablePlayerTiles(playerPosition, frozenset(), neighbors)
        #every tile a box can be pushed onto without walking into a wall or a simple deadlock
        validBoxTiles = frozenset(
            tile for tile in interiorTiles if not deadLockTable[tile[0]][tile[1]]
        )
        hotspotDictionary = generateHotspotTable(goalState, interiorTiles, validBoxTiles, neighbors)
        rooms, roomGraph, tileToRoom = generateRoomsAndGraph(interiorTiles)
        heuristicCache = {}
        cellCache = {}
        startReachable, startMinimum = reachablePlayerTiles(playerPosition, boxCoordinates, neighbors)
        startState = (playerPosition, boxCoordinates, startReachable)
        tunnelTiles = detectTunnelSquares(interiorTiles, goalState, walls)

        def stateKey(state, minimumCell):
            return (minimumCell, state[1])

        def getRawCell(boxes):
            if boxes in cellCache:
                return cellCache[boxes]
            oop = getOOPBoxes()
            packed = getPackedBoxes(boxes, goalState)
            connectivity = countRegions(mapData, width, height, boxes)
            roomConnectivity = countRoomConnectivity(boxes, rooms, roomGraph, tileToRoom, neighbors)
            rawCell = (oop, -packed, connectivity, roomConnectivity)
            cellCache[boxes] = rawCell
            return rawCell

        startKey = stateKey(startState, startMinimum)
        stateByKey = {startKey: startState}
        
        parents = {startKey: None}
        explored = {startKey}
        exploredReigons = {boxCoordinates: [startReachable]}
        cellQueues = defaultdict(list)
        stateCounter = 0
        startCell = getRawCell(boxCoordinates)
        bestAncestor = {startKey: startCell}
        heuristic = calculateHeuristic(boxCoordinates, distanceStack, goalState, heuristicCache, hotspotDictionary)
        mobility = countMobility(boxCoordinates, startReachable, neighbors)
        heapq.heappush(cellQueues[startCell], (heuristic, -mobility, -stateCounter, startKey))
        stateCounter += 1

        while cellQueues:
            for cellKey in list(cellQueues.keys()):
                cellHeap = cellQueues.get(cellKey)
                if not cellHeap:
                    cellQueues.pop(cellKey, None)
                    continue
                _, _, _, currentKey = heapq.heappop(cellHeap)
                if not cellHeap:
                    del cellQueues[cellKey]

                currentPlayerPos, currentBoxCoords, reachable = stateByKey[currentKey]
                currentBestAncestor = bestAncestor[currentKey]

                if currentBoxCoords == goalState:
                    transitions = []
                    key = currentKey
                    while parents[key] is not None:
                        previousKey, standingTile, pushDirection, pushCount = parents[key]
                        transitions.append((previousKey, standingTile, pushDirection, pushCount))
                        key = previousKey
                    transitions.reverse()

                    winPath = []
                    for previousKey, standingTile, pushDirection, pushCount in transitions:
                        previousPlayer, previousBoxes, idc = stateByKey[previousKey]
                        walk = findWalkingPath(
                            previousPlayer, standingTile, previousBoxes, neighbors
                        )
                        if walk is None:
                            raise RuntimeError('Invalid reconstructed push path')
                        winPath.extend(walk)
                        winPath.append(DIRECTIONS[pushDirection]*pushCount)

                    print(f"Total states explored: {len(explored)}")
                    distinct = {key[1] for key in explored}
                    print(f"Total unique box configurations explored: {len(distinct)}")
                    return ''.join(winPath)

                boxesToExplore = currentBoxCoords
                exploredCorral = set()

                #only start a corral from the tile right behind a box the player can push right now
                #a pi corral always has one of these, so flooding any other pocket is just wasted work
                corralSeeds = set()
                for box in currentBoxCoords:
                    for destRow, destCol in DIRECTION:
                        standingTile = (box[0] - destRow, box[1] - destCol)
                        boxDestination = (box[0] + destRow, box[1] + destCol)
                        if standingTile in reachable and boxDestination in validBoxTiles:
                            if boxDestination not in reachable and boxDestination not in currentBoxCoords:
                                corralSeeds.add(boxDestination)

                for currentPos in corralSeeds:
                    if currentPos in exploredCorral:
                        continue
                    corral, corralBoxes = getCorralFromTile(currentPos, currentBoxCoords, neighbors)
                    exploredCorral.update(corral)

                    #no boxes or every box in the corral means restricting to it prunes nothing
                    if not corralBoxes or len(corralBoxes) == len(currentBoxCoords):
                        continue

                    #every box is already on a goal and theres no empty goal inside, so nothing to do here
                    if corralBoxes.issubset(goalState) and corral.isdisjoint(goalState):
                        continue

                    isPIC, _, totalBoxes = isPICorral(
                        corral, corralBoxes, reachable, currentBoxCoords, walls, validBoxTiles
                    )

                    if isPIC:
                        boxesToExplore = totalBoxes
                        break

                statesMade = 0
                for box in boxesToExplore:
                    macroFrontier = [(currentKey, box, currentBoxCoords, reachable)]
                    for macroParentKey, currentBox, macroBoxCoords, macroReachable in macroFrontier:
                        for destRow, destCol in DIRECTION:
                            standingTile = (currentBox[0] - destRow, currentBox[1] - destCol)

                            #If we cant reach the position of the player after pushing the box, cant push so prune
                            if standingTile not in macroReachable:
                                continue

                            boxDestination = (currentBox[0] + destRow, currentBox[1] + destCol)

                            #if pushing into a wall, a simple deadlock or another box, prune
                            if boxDestination not in validBoxTiles or boxDestination in macroBoxCoords:
                                continue

                            #after the push the player ends up where the box was
                            newPlayerPos = currentBox
                            newBoxCoords = (macroBoxCoords - {currentBox}) | {boxDestination}
                            pushDirection = (destRow, destCol)
                            pushCount = 1
                            movingBox = boxDestination
                            #keep pushing the box through the tunnel until it comes out the other side or hits something
                            while (movingBox, pushDirection) in tunnelTiles:
                                nextBox = (movingBox[0] + destRow, movingBox[1] + destCol)
                                if nextBox not in validBoxTiles or nextBox in newBoxCoords:
                                    break
                                newBoxCoords = (newBoxCoords - {movingBox}) | {nextBox}
                                newPlayerPos = movingBox
                                movingBox = nextBox
                                pushCount += 1
                            boxDestination = movingBox

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

                            #check the freeze deadlocks first so we dont waste time on the heuristic of a dead state
                            if hasFreezeSubset(boxDestination, newBoxCoords, freezeSubset):
                                continue

                            freezeExplored = dict()
                            exploring = set()
                            frozenBoxes = detectBlockedAxes(boxDestination, mapData, width, height, deadLockTable, freezeExplored, newBoxCoords, exploring)
                            if boxDestination in frozenBoxes and any(b not in goalState for b in frozenBoxes):
                                if len(frozenBoxes) <= 4:
                                    frozenPattern = frozenset(frozenBoxes)
                                    #save the pattern under every box in it so hasFreezeSubset only has to check the pushed box
                                    for frozenBox in frozenPattern:
                                        freezeSubset[frozenBox].append(frozenPattern)
                                continue

                            heuristic = calculateHeuristic(newBoxCoords, distanceStack, goalState, heuristicCache, hotspotDictionary)

                            if heuristic >= 999999:
                                continue

                            newReachable, newMinimum = reachablePlayerTiles(newPlayerPos, newBoxCoords, neighbors)

                            #if the boxes positions have not been seen before
                            if newBoxCoords not in exploredReigons:
                                exploredReigons[newBoxCoords] = []
                            #add the new reachable region to the list of explored regions for this box configuration
                            exploredReigons[newBoxCoords].append(newReachable)

                            newState = (newPlayerPos, newBoxCoords, newReachable)
                            newKey = stateKey(newState, newMinimum)
          
                            explored.add(newKey)
                            parents[newKey] = (macroParentKey, standingTile, (destRow, destCol), pushCount)
                            stateByKey[newKey] = newState

                            rawCell = getRawCell(newBoxCoords)
                            projectedCell, newBestAncestor = getProjectedCell(rawCell, currentBestAncestor)
                            bestAncestor[newKey] = newBestAncestor
                            mobility = countMobility(newBoxCoords, newReachable, neighbors)

                            #negative counter so ties pop the newest state first, keeps following 1 push line instead of jumping around
                            heapq.heappush(cellQueues[projectedCell], (heuristic, -mobility, -stateCounter, newKey))
                            stateCounter += 1
                            statesMade += 1
                            macroFrontier.append((newKey, boxDestination, newBoxCoords, newReachable))

        return ""


def getOOPBoxes():
    return 0

def getPackedBoxes(boxCoordinates, goalState):
    packed = 0
    for box in boxCoordinates:
        if box in goalState:
            packed += 1
    return packed

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

            for currentRow, currentCol in frontier:
                for rowY, colX in DIRECTION:
                    dest = (currentRow + rowY, currentCol + colX)
                    
                    if isValid(dest, mapData, width, height) and dest not in boxes and dest not in explored:
                        explored.add(dest)
                        frontier.append(dest)
                        
    # Return the number of rooms found
    return rooms

def generateRoomsAndGraph(interiorTiles):
    #find 2x3 and 3x2 rectangles
    roomTiles = set()
    for row, col in interiorTiles:
        twoByThree = {(row, col), (row, col+1), (row, col+2), (row+1, col), (row+1, col+1), (row+1, col+2)}
        threeByTwo = {(row, col), (row+1, col), (row+2, col), (row, col+1), (row+1, col+1), (row+2, col+1)}

        if twoByThree.issubset(interiorTiles):
            roomTiles.update(twoByThree)
        if threeByTwo.issubset(interiorTiles):
            roomTiles.update(threeByTwo)

    # roomTiles is just one big set of tiles, to aggregate them into separate rooms
    # do a flood fill bfs, every tile connected to eachother is in the same room
    # it checks the 4 directions manually and checks if that tile exists in the same set,
    # if yes append to same room
    # if no continue
    # Imagine ms paint bucket, theyre separated by corridor tiles.
    # code below just aggregates the tiles into rooms
    rooms = []
    tileToRoom = {}
    
    #destructively popping from roomTiles means we dont need an explored set
    while roomTiles:
        startPos = roomTiles.pop()
        roomId = len(rooms)
        currentRoom = {startPos}
        frontier = [startPos]
        tileToRoom[startPos] = roomId
        # roomID and tile to room is baically just answering
        # which room does this tile belong to in O(1) time

        #looping over a list that keeps growing is faster than popleft on a deque
        #each frontier loop is 1 room
        for currentPos in frontier:
            for destRow, destCol in DIRECTION:
                dest = (currentPos[0] + destRow, currentPos[1] + destCol)
                
                if dest in roomTiles:
                    roomTiles.remove(dest)
                    currentRoom.add(dest)
                    tileToRoom[dest] = roomId
                    frontier.append(dest)
        
        rooms.append(currentRoom)

    # Were just declaring the adj list, no assignment
    # just saying for every room create a set of other rooms adjacent/connected to it
    roomGraph = {id: set() for id in range(len(rooms))}

    # RoomID is synonymous with the index
    # Were now assigning rooms to the adj list
    for roomId, roomTiles in enumerate(rooms):
        #the entire room acts as the starting footprint
        frontier = list(roomTiles)
        #explored is a room check and explored corridor tile check in 1
        explored = set(roomTiles)

        # for every tile in the room, check the 4 directions
        for currentPos in frontier:
            for destRow, destCol in DIRECTION:
                dest = (currentPos[0] + destRow, currentPos[1] + destCol)
                # Its not a tile in the room
                if dest in interiorTiles and dest not in explored:
                    explored.add(dest)

                    # if we found a different room, find which room that tile is in, and add it to the adj list value of the current room
                    if dest in tileToRoom:
                        #we hit a different room so record the edge
                        neighborRoom = tileToRoom[dest]
                        roomGraph[roomId].add(neighborRoom)
                        roomGraph[neighborRoom].add(roomId)
                        #do not add dest to frontier so we dont walk through the neighbor room
                    else:
                        #its a corridor tile so keep flooding outward
                        frontier.append(dest)

    return rooms, roomGraph, tileToRoom

def countRoomConnectivity(boxes, rooms, roomGraph, tileToRoom, neighbors):
    brokenEdges = 0

    for roomId, roomTiles in enumerate(rooms):
        if not roomGraph[roomId]:
            continue

        exploredRoom = set()

        for startPos in roomTiles:
            if startPos in boxes or startPos in exploredRoom:
                continue

            frontier = [startPos]
            explored = {startPos}
            exploredRoom.add(startPos)
            connectedRooms = set()

            for currentPos in frontier:
                for dest in neighbors[currentPos]:
                    if dest not in boxes and dest not in explored:
                        explored.add(dest)

                        if dest in tileToRoom:
                            neighborRoom = tileToRoom[dest]
                            if neighborRoom == roomId:
                                exploredRoom.add(dest)
                                frontier.append(dest)
                            else:
                                connectedRooms.add(neighborRoom)
                        else:
                            frontier.append(dest)

            brokenEdges += len(roomGraph[roomId]) - len(connectedRooms)

        if not exploredRoom:
            brokenEdges += len(roomGraph[roomId])

    return brokenEdges

def countMobility(boxCoordinates, reachable, neighbors):
    mobility = 0
    for box in boxCoordinates:
        for dest in neighbors[box]:
            if dest in reachable:
                mobility += 1
    return mobility

def getProjectedCell(rawCell, bestAncestorCell):
    if rawCell < bestAncestorCell:
        return rawCell, rawCell
    return bestAncestorCell, bestAncestorCell