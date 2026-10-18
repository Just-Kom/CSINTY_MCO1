import numpy as np
import heapq
from scipy.optimize import linear_sum_assignment
from collections import deque
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

def detectBlockedAxes(boxCoordinate, validBoxTiles, walls, explored, boxCoordinates, exploring):
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
                if dest in walls:
                    blockedAxes[axisIndex] = True
                    break
                if dest not in validBoxTiles:
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

    if not isFrozen(boxCoordinate):
        return None
    return {box for box, frozen in explored.items() if frozen}


def generateDistanceMatrices(goalState, height, width, validBoxTiles, neighbors):
    #create goalState many matrices in a dict
    matrices = {}

    for goalVal in goalState:
        matrix = np.full((height, width), -1)

        startState = (goalVal, 0)
        matrix[goalVal[0]][goalVal[1]] = 0
        frontier = deque([startState])
        explored = {goalVal}

        while frontier:
            currentPos, currentCtr = frontier.popleft()
            for row, col in DIRECTION:
                dest = (currentPos[0] + row, currentPos[1] + col)
                #the player has to stand behind dest to push the box back to currentPos, so that cant be a wall
                playerStand = (dest[0] + row, dest[1] + col)
                if dest not in explored and dest in validBoxTiles and playerStand in neighbors[dest]:
                    explored.add(dest)
                    newCtr = currentCtr + 1
                    frontier.append((dest, newCtr))
                    matrix[dest[0]][dest[1]] = newCtr
        matrices[goalVal] = matrix
    return matrices
        #expand from the goalVal to the map

def buildDistanceStack(distanceMatrices, goalState, validBoxTiles):
    distanceStack = np.stack([distanceMatrices[goalVal] for goalVal in goalState])
    distanceStack = np.where(distanceStack == -1, 999999, distanceStack)
    return {tile: distanceStack[:, tile[0], tile[1]].copy() for tile in validBoxTiles}

def calculateHeuristic(boxCoordinates, tileDistances, heuristicCache, hotspotDictionary):
    costMatrix = np.array([tileDistances[box] for box in boxCoordinates])
    row, col = linear_sum_assignment(costMatrix)
    hotspots = countHotspots(boxCoordinates, hotspotDictionary)
    heuristic = int(costMatrix[row, col].sum()) + hotspots * 10
    heuristicCache[boxCoordinates] = heuristic
    return heuristic

def reachablePlayerTiles(playerPosition, boxCoordinates, neighbors):
    frontier = [playerPosition]
    reachable = {playerPosition}
    #looping over a list that keeps growing is faster than popleft on a deque
    for current in frontier:
        for dest in neighbors[current]:
            if dest not in reachable and dest not in boxCoordinates:
                reachable.add(dest)
                frontier.append(dest)
    return reachable

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
                return False

            if playerPos not in reachable and playerPos not in corralBoxes:
                return False

            if playerPos in reachable and boxDest not in boxCoordinates:
                hasValidPush = True

    return hasValidPush

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

def generateHotspotTable(goalState, interiorTiles, neighbors):
    def getReachableGoalsCount(goal, hotspotTile):
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

        return reached, explored

    # get the number of reachable goals if there are no hotspots
    reachableGoalCount = {tile: 0 for tile in interiorTiles}
    goalReached = {}
    visitedByGoal = {}

    for goal in goalState:
        reached, explored = getReachableGoalsCount(goal, None)
        goalReached[goal] = reached
        visitedByGoal[goal] = {playerPos for playerPos, _ in explored} | reached
        for tile in reached:
            reachableGoalCount[tile] += 1

    #every tile a box can be pushed onto without walking into a wall or a simple deadlock
    validBoxTiles = frozenset(
        tile for tile, count in reachableGoalCount.items() if count > 0
    )
    
    #map a tuple coordinate acting as a hotspot to a set of trapped tuple coordinates
    hotspotDictionary = {tile: set() for tile in interiorTiles}
    
    #test every valid floor tile as an immovable hotspot
    for hotspotTile in validBoxTiles:
        hotspotGoalCount = {tile: 0 for tile in validBoxTiles}
        for goal in goalState:
            if goal == hotspotTile:
                continue
            if hotspotTile in visitedByGoal[goal]:
                reached, _ = getReachableGoalsCount(goal, hotspotTile)
            else:
                reached = goalReached[goal]
            for tile in reached:
                if tile in hotspotGoalCount:
                    hotspotGoalCount[tile] += 1
        
        for tile in validBoxTiles:
            if tile == hotspotTile:
                continue
            expectedCount = reachableGoalCount[tile]
            if hotspotTile in goalReached and tile in goalReached[hotspotTile]:
                expectedCount -= 1
            #if the number of reachable goals dropped, the hotspot tile is blocking boxes on this tile from getting into our goal room
            if hotspotGoalCount[tile] < expectedCount:
                hotspotDictionary[hotspotTile].add(tile)
                
    return validBoxTiles, hotspotDictionary

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
        interiorTiles = reachablePlayerTiles(playerPosition, frozenset(), neighbors)
        validBoxTiles, hotspotDictionary = generateHotspotTable(goalState, interiorTiles, neighbors)
        distances = generateDistanceMatrices(goalState, height, width, validBoxTiles, neighbors)
        tileDistances = buildDistanceStack(distances, goalState, validBoxTiles)
        freezeSubset = defaultdict(set)
        rooms, roomGraph, tileToRoom = generateRoomsAndGraph(interiorTiles, neighbors)
        heuristicCache = {}
        featureSpaceCache = {}
        startReachable = reachablePlayerTiles(playerPosition, boxCoordinates, neighbors)
        tunnelTiles = detectTunnelSquares(interiorTiles, goalState, walls)

        def getFeatureSpace(boxes):
            if boxes in featureSpaceCache:
                return featureSpaceCache[boxes]
            packed = getPackedBoxes(boxes, goalState)
            connectivity = countRegions(interiorTiles, boxes, neighbors)
            roomConnectivity = countRoomConnectivity(boxes, rooms, roomGraph, tileToRoom, neighbors)
            rawFeatureSpace = (-packed, connectivity, roomConnectivity)
            featureSpaceCache[boxes] = rawFeatureSpace
            return rawFeatureSpace

        startKey = (playerPosition, boxCoordinates)
        stateByKey = {startKey: startReachable}
        
        parents = {startKey: None}
        exploredReigons = defaultdict(list, {boxCoordinates: [startReachable]})
        featureSpaceQueues = defaultdict(list)
        stateCounter = 0
        startFeatureSpace = getFeatureSpace(boxCoordinates)
        bestAncestorFeatureSpace = {startKey: startFeatureSpace}
        heuristic = calculateHeuristic(boxCoordinates, tileDistances, heuristicCache, hotspotDictionary)
        mobility = countMobility(boxCoordinates, startReachable, neighbors)
        heapq.heappush(featureSpaceQueues[startFeatureSpace], (heuristic, -mobility, -stateCounter, startKey))
        stateCounter += 1

        while featureSpaceQueues:
            for featureSpaceKey in list(featureSpaceQueues.keys()):
                featureSpaceHeap = featureSpaceQueues.get(featureSpaceKey)
                if not featureSpaceHeap:
                    featureSpaceQueues.pop(featureSpaceKey, None)
                    continue
                _, _, _, currentKey = heapq.heappop(featureSpaceHeap)
                if not featureSpaceHeap:
                    del featureSpaceQueues[featureSpaceKey]

                currentPlayerPos, currentBoxCoords = currentKey
                reachable = stateByKey.pop(currentKey)
                currentBestAncestor = bestAncestorFeatureSpace[currentKey]

                if currentBoxCoords == goalState:
                    transitions = []
                    key = currentKey
                    while parents[key] is not None:
                        previousKey, standingTile, pushDirection, pushCount = parents[key]
                        transitions.append((previousKey, standingTile, pushDirection, pushCount))
                        key = previousKey
                    transitions.reverse()

                    winPath = []
                    for (previousPlayer, previousBoxes), standingTile, pushDirection, pushCount in transitions:
                        walk = findWalkingPath(
                            previousPlayer, standingTile, previousBoxes, neighbors
                        )
                        if walk is None:
                            raise RuntimeError('Invalid reconstructed push path')
                        winPath.extend(walk)
                        winPath.append(DIRECTIONS[pushDirection]*pushCount)

                    print(f"Total states explored: {len(parents)}")
                    print(f"Total unique box configurations explored: {len(exploredReigons)}")
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

                    if isPICorral(
                        corral, corralBoxes, reachable, currentBoxCoords, walls, validBoxTiles
                    ):
                        boxesToExplore = corralBoxes
                        break

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
                            pushDirection = (destRow, destCol)
                            pushCount = 1
                            movingBox = boxDestination
                            #keep pushing the box through the tunnel until it comes out the other side or hits something
                            while (movingBox, pushDirection) in tunnelTiles:
                                nextBox = (movingBox[0] + destRow, movingBox[1] + destCol)
                                if nextBox not in validBoxTiles or nextBox in macroBoxCoords:
                                    break
                                newPlayerPos = movingBox
                                movingBox = nextBox
                                pushCount += 1
                            boxDestination = movingBox
                            newBoxCoords = (macroBoxCoords - {currentBox}) | {boxDestination}

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

                            if newBoxCoords in heuristicCache:
                                heuristic = heuristicCache[newBoxCoords]
                            else:
                                #check the freeze deadlocks first so we dont waste time on the heuristic of a dead state
                                if hasFreezeSubset(boxDestination, newBoxCoords, freezeSubset):
                                    continue

                                freezeExplored = dict()
                                exploring = set()
                                frozenBoxes = detectBlockedAxes(boxDestination, validBoxTiles, walls, freezeExplored, newBoxCoords, exploring)
                                if frozenBoxes is not None and any(b not in goalState for b in frozenBoxes):
                                    if len(frozenBoxes) <= 4:
                                        frozenPattern = frozenset(frozenBoxes)
                                        #save the pattern under every box in it so hasFreezeSubset only has to check the pushed box
                                        for frozenBox in frozenPattern:
                                            freezeSubset[frozenBox].add(frozenPattern)
                                    continue

                                heuristic = calculateHeuristic(newBoxCoords, tileDistances, heuristicCache, hotspotDictionary)

                            if heuristic >= 999999:
                                continue

                            newReachable = reachablePlayerTiles(newPlayerPos, newBoxCoords, neighbors)

                            #add the new reachable region to the list of explored regions for this box configuration
                            exploredReigons[newBoxCoords].append(newReachable)

                            newKey = (newPlayerPos, newBoxCoords)
                            parents[newKey] = (macroParentKey, standingTile, (destRow, destCol), pushCount)
                            stateByKey[newKey] = newReachable

                            rawFeatureSpace = getFeatureSpace(newBoxCoords)
                            projectedFeatureSpace, newBestAncestor = getProjectedFeatureSpace(rawFeatureSpace, currentBestAncestor)
                            bestAncestorFeatureSpace[newKey] = newBestAncestor
                            mobility = countMobility(newBoxCoords, newReachable, neighbors)

                            #negative counter so ties pop the newest state first, keeps following 1 push line instead of jumping around
                            heapq.heappush(featureSpaceQueues[projectedFeatureSpace], (heuristic, -mobility, -stateCounter, newKey))
                            stateCounter += 1
                            macroFrontier.append((newKey, boxDestination, newBoxCoords, newReachable))

        return ""


def getPackedBoxes(boxCoordinates, goalState):
    return len(boxCoordinates & goalState)

def countRegions(interiorTiles, boxes, neighbors):
    explored = set()
    rooms = 0

    #For every single node in the map
    for currentPos in interiorTiles:
        #If its a wall, a box, or already explored, then skip it
        if currentPos in boxes or currentPos in explored:
            continue

        #Else, new "room" so basically fill the reigon then increment our room counter
        rooms += 1
        
        #everything below is just standard flood fill bfs
        frontier = [currentPos]
        explored.add(currentPos)

        for current in frontier:
            for dest in neighbors[current]:
                if dest not in boxes and dest not in explored:
                    explored.add(dest)
                    frontier.append(dest)
                        
    # Return the number of rooms found
    return rooms

def generateRoomsAndGraph(interiorTiles, neighbors):
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
            for dest in neighbors[currentPos]:
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
            for dest in neighbors[currentPos]:
                # Its not a tile in the room
                if dest not in explored:
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

        explored = set()
        connectedRooms = set()

        for startPos in roomTiles:
            if startPos in boxes or startPos in explored:
                continue

            frontier = [startPos]
            explored.add(startPos)

            for currentPos in frontier:
                for dest in neighbors[currentPos]:
                    if dest not in boxes and dest not in explored:
                        explored.add(dest)

                        if dest in tileToRoom:
                            neighborRoom = tileToRoom[dest]
                            if neighborRoom == roomId:
                                frontier.append(dest)
                            else:
                                connectedRooms.add(neighborRoom)
                        else:
                            frontier.append(dest)

            if len(connectedRooms) == len(roomGraph[roomId]):
                break

        for neighborRoom in roomGraph[roomId]:
            if roomId < neighborRoom and neighborRoom not in connectedRooms:
                brokenEdges += 1

    return brokenEdges

def countMobility(boxCoordinates, reachable, neighbors):
    mobility = 0
    for box in boxCoordinates:
        for dest in neighbors[box]:
            if dest in reachable:
                mobility += 1
    return mobility

def getProjectedFeatureSpace(rawFeatureSpace, bestAncestorFeatureSpace):
    if rawFeatureSpace < bestAncestorFeatureSpace:
        return rawFeatureSpace, rawFeatureSpace
    return bestAncestorFeatureSpace, bestAncestorFeatureSpace