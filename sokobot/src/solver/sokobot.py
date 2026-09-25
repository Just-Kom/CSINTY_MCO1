import time
import numpy as np
import heapq
from scipy.optimize import linear_sum_assignment
from collections import deque

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

"""
From JM: 
Understood all functions except for the generating of deadlocks, lol.
Pa-check nalang Aaron kapag tama HAHAHA
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

"""
Where is this used tho?
"""
def hasFreezeSubset(boxCoordinates, mapData, width, height, freezeSubset):
    # freezeSubset is a set of boxes that are frozen, we want to check if any of the boxes in boxCoordinates are in freezeSubset
    for box in freezeSubset:
        if box.issubset(boxCoordinates):
            return True
    return False

def detectBlockedAxes(boxCoordinate, mapData, width, height, deadLockTable, explored, boxCoordinates):
    # Turn the boxcoords into a dict, each boxcoord value is (False,False) where (Y-axis, X-Axis)
    # iboxcoordinate is just 1 coordinate, the a pushed box, were just checking its surroundings to check if its a frozen box.
    # explored is the set of all boxes that have been checked for frozen axes, so we dont check them again.
    # axes = [DIRECTION[2:], DIRECTION[:2]]
    # axisIndex 0 is the Y axis, axisIndex 1 is the X axis
    # Additional (JM): Do note that this function is a recursive one
    # JM # 2: the 'dest' here refers to an ADJACENT block of any given box, NOT the target goal
    """
    :param boxCoordinate:
    :param mapData:
    :param width:
    :param height:
    :param deadLockTable:
    :param explored:
    :param boxCoordinates:
    :return:
    """

    currentBox = boxCoordinate
    explored[currentBox] = [False, False]
    # the axes is just a list of 2 lists, each list contains the 2 directions of a single axis.
    # a single axis looks like this  (0, -1), (0, 1) which represents the left and right of the box at the x axis.

    # so this is saying, foe each axis
    for axisIndex, axis in enumerate(AXES):
        numOfSimpleDeadlocks = 0
        #check the left and right
        for __,leftRight in enumerate(axis):
            #left right is a tuple (row, col)
            # if the left or right of the box has a wall
            dest = (currentBox[0]+leftRight[0], currentBox[1]+leftRight[1])
            # if dest is out of bounds then just continue to the next direction
            """
            Equivalent to: checking for ArrayOutOfBoundsError
            Something I noticed lang, we used this exact block for four times throughout the code. 
            Do you think we can turn this into one method instead?
            
            if isOutOfBounds(dest[], height, width):
                continue
                
            def isOutOfBounds(dest[], height, width):
                if dest[1] < 0 or dest[1] >= width or dest[0] < 0 or dest[0] >= height:
                    return True
                else:
                    return False
            """
            if dest[1] < 0 or dest[1] >= width or dest[0] < 0 or dest[0] >= height:
                continue
            """
            On First Iteration:
            Equivalent to: checking a wall on the left or right
            #$ - returns True
            $# - return True
            
            #
            $ - returns False
            
            On Second Iteration:
            #
            $ - return True
            
            $
            # - returns True
            
            """
            if mapData[dest[0]][dest[1]] == '#':
                # Is there a wall on the left or on the right side?
                explored[currentBox][axisIndex] = True


            """
            
            """
            # Is there a simple deadlock square on the right and the left side?
            if deadLockTable[dest[0]][dest[1]]:
                numOfSimpleDeadlocks += 1


            """
            Di ko rin nagets toh hahaha. I'll review it later. 
            """
            # Is there a box on the left or on the right side, which is already blocked?
            if dest in boxCoordinates:
                if dest not in explored:
                    detectBlockedAxes(dest, mapData, width, height, deadLockTable, explored, boxCoordinates)
                    # if the box on the left or right side is blocked, then the current box is also blocked on that axis
                    # if both axes are blocked, then the current box is also blocked
                if explored[dest][0] and explored[dest][1]:
                    explored[currentBox][axisIndex] = True

        # return all of the boxes that are blocked on both axes, which means they are frozen and cannot be moved anymore
        if numOfSimpleDeadlocks == 2:
            explored[currentBox][axisIndex] = True

    return {box for box, blocked in explored.items() if blocked[0] and blocked[1]}
    

#this seems to be the same function as the after one. Should we combine the two?
def isValid(dest, mapData, width, height):
    if dest[1] < 0 or dest[1] >= width or dest[0] < 0 or dest[0] >= height:
        return False
    if mapData[dest[0]][dest[1]] == '#':
        return False
    return True

#this seems to be the same function as the previous one. Should we combine the two?
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
    #Check if a box can be pulled from a goal state to a certain tile
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

"""
Note: use this for a movement. This doesn't have boxes checking. 

Suggestion, can be made simpler by utilizing isValid. 
"""
def isValidMove(dest, boxCoordinates, mapData, width, height):
    #Out of bounds
    if dest[1] < 0 or dest[1] >= width or dest[0] < 0 or dest[0] >= height:
        return False
    #If the place where the character/box is going is a wall
    if mapData[dest[0]][dest[1]] == '#':
        return False
    #if the adjacent space is a box
    if dest in boxCoordinates:
        return False
    return True

def movePlayer(playerCoordinate, dest, boxCoordinates, mapData, width, height, deadLockTable, freezeSubset, goalState):
    deltaX = dest[0] - playerCoordinate[0]
    deltaY = dest[1] - playerCoordinate[1]
    newDest = (dest[0] + deltaX, dest[1] + deltaY)

    # if player is pushing a box
    if dest in boxCoordinates:
        # if pushing into a wall or something, dont prune
        if not isValidMove(newDest, boxCoordinates, mapData, width, height):
            return None
        # if moving into a simple deadlock prune
        if deadLockTable[newDest[0]][newDest[1]]:
            return None

        # hypothetical situation where we already pushed the box
        newBoxCoordinates = frozenset(boxCoordinates - {dest} | {newDest})

        """"
        # if we already know that its a frozen box, then prune
        if hasFreezeSubset(newBoxCoordinates, mapData, width, height, freezeSubset):
            return None

        # else, check if the move we just did causes a frozen box, if it does, then add it to the freezeSubset and prune
        freezeExplored = dict()
        frozenBoxes = detectBlockedAxes(newDest, mapData, width, height, deadLockTable, freezeExplored, newBoxCoordinates)
        # if the box is frozen, add it to the list of patterns then prune
        if newDest in frozenBoxes:
            if any(box in goalState for box in frozenBoxes):
                freezeSubset.append(frozenset(frozenBoxes))
                return None 
        # else, the box is not frozen, so continue
        """
        return (dest, newBoxCoordinates)
    # else the player is just moving to an empty space
    elif isValidMove(dest, boxCoordinates, mapData, width, height):
        return (dest, boxCoordinates)
    # if all else fails, return None
    return None

class SokoBot:
    def solveSokobanPuzzle(self, width, height, mapData, itemsData):

        """
        This block identifies the target destinations and places their coordinates onto the mapData.
        It also turns the goalState into a frozzenset - a unique concept
        """
        goalState = []
        for row,rowVal in enumerate(mapData):
            for col,colVal in enumerate(rowVal):
                if colVal == '.':
                    goalState.append((row, col))
        goalState = frozenset(goalState)

        """
        This block identifies the following:
         1. box coordinates
         2. player coordinates at this specific instance
        """
        boxCoordinates = []
        for row,rowVal in enumerate(itemsData):
            for col,colVal in enumerate(rowVal):
                if colVal == '$':
                    boxCoordinates.append((row, col))
                elif colVal == '@':
                    playerPosition = (row, col)
        boxCoordinates = frozenset(boxCoordinates)
        startState = (playerPosition, boxCoordinates)

        """
        The actual bfs algorithm. 
        Frontier is a stack, but using a python "deque" collection as it is faster. 
        Paths is a set in Python which can be appended and not. 
        Explored is also a set in Python that is unordered. 
        
        
        """
        frontier = deque()
        paths = {startState: None}
        frontier.append(startState)
        explored = {startState}
        freezeSubset = []

        """
        Generating the deadlock table
        """
        deadLockTable = generateSimpleDeadlock(mapData, width, height, goalState, playerPosition)

        # shorthand for "while may laman ang frontier":
        while frontier:
            #known as pop the frontier in our discussions
            currentState = frontier.popleft()

            #time to explore the state!
            currentPlayerPos = currentState[0]
            currentBoxCoords = currentState[1]

            #if the current box coodinates align fully with goal state, consider it a win and invoke finding the win path.
            if currentBoxCoords == goalState:
                winPath = []
                while currentState is not None:
                    #gets the currentState of each previous path, literally embedded per state
                    #aka. per state has its previous state stored within it
                    prevState = paths[currentState]
                    #it starts from destnation baliktad
                    if prevState is not None:
                        winPath.append(DIRECTIONS[(currentState[0][0] - prevState[0][0], currentState[0][1] - prevState[0][1])])
                    currentState = prevState
                #it then reserveses the current state
                winPath.reverse()
                print(f"Total states explored: {len(explored)}")
                return winPath
                #loop to find path then return path
            
            for index,value in enumerate(DIRECTION):
                row = value[0]
                col = value[1]
                dest = (currentPlayerPos[0] + row, currentPlayerPos[1] + col)
                #def detectBlockedAxes(boxCoordinate, mapData, width, height, deadLockTable, explored, axes, boxCoordinates):
                # movePlayer is where all of the shenanigans happens!
                newState = movePlayer(currentPlayerPos, dest, currentBoxCoords, mapData, width, height, deadLockTable, freezeSubset, goalState)
                if newState is not None and newState not in explored:
                    explored.add(newState)
                    paths[newState] = currentState
                    frontier.append(newState)
        return "l"


            