#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/*
dest - Target coordinate on the grid
Type: pair (row, col) / flat tile index

dest[0] - Row index (vertical, top is 0)
Type: int
Note: UP AND DOWN MOVEMENT

dest[1] - Column index (horizontal, left is 0)
Type: int
Note: LEFT AND RIGHT MOVEMENT

playerCoordinate - Current location of the player
Type: int (row * width + col)

boxCoordinates - Current locations of all boxes
Type: sorted vector of tile indices

goalState - Winning target locations for boxes
Type: sorted vector of tile indices

deltaX - Row difference between dest and position
Type: int

deltaY - Column difference between dest and position
Type: int

DIRECTION - The 4 possible adjacent moves (row_change, col_change)
Type: array of pairs
Note: Used for iteration and expansion

DIRECTIONS - Mapping of coordinate changes to output characters ('u', 'd', 'l', 'r')
Type: helper function
Note: Used for outputting the solution path, for O(1) lookup
*/

static const int DIRECTION[4][2] = {
    {0, -1},
    {0, 1},
    {1, 0},
    {-1, 0}
};

static const int AXES[2][2][2] = {
    {{0, -1}, {0, 1}},  // X-axis (left and right)
    {{1, 0}, {-1, 0}}   // Y-axis (down and up)
};

static inline char getDirectionChar(int deltaRow, int deltaCol) {
    if (deltaRow == 0 && deltaCol == -1) return 'l';
    if (deltaRow == 0 && deltaCol == 1) return 'r';
    if (deltaRow == 1 && deltaCol == 0) return 'd';
    return 'u';
}

struct VectorHash {
    std::size_t operator()(const std::vector<int>& v) const noexcept {
        std::size_t seed = v.size();
        for (int x : v) {
            seed ^= static_cast<std::size_t>(x) + 0x9e3779b9UL + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

using FeatureSpace = std::tuple<int, int, int>;

static inline bool containsBox(const std::vector<int>& boxCoordinates, int tile) {
    return std::binary_search(boxCoordinates.begin(), boxCoordinates.end(), tile);
}

static inline bool isSubsetOf(const std::vector<int>& subset, const std::vector<int>& superset) {
    return std::includes(superset.begin(), superset.end(), subset.begin(), subset.end());
}

bool hasFreezeSubset(
    int boxDestination,
    const std::vector<int>& boxCoordinates,
    const std::vector<std::vector<std::vector<int>>>& freezeSubset
) {
    // freezeSubset maps a tile to every frozen set of boxes that has a box on that tile
    // the parent state wasnt frozen, so any new freeze has to include the box we just pushed
    for (const auto& subset : freezeSubset[boxDestination]) {
        if (isSubsetOf(subset, boxCoordinates)) {
            return true;
        }
    }
    return false;
}

bool detectBlockedAxes(
    int boxCoordinate,
    int width,
    int height,
    const std::vector<uint8_t>& validBoxTiles,
    const std::vector<uint8_t>& walls,
    std::vector<int8_t>& explored,
    const std::vector<uint8_t>& boxGrid,
    std::vector<uint8_t>& exploring,
    std::vector<int>& frozenBoxesOut
) {
    std::function<bool(int)> isFrozen = [&](int currentBox) -> bool {
        if (explored[currentBox] != -1) {
            return explored[currentBox] == 1;
        }
        if (exploring[currentBox]) {
            return true;
        }

        bool blockedAxes[2] = {false, false};
        std::vector<int> adjBoxesPerAxis[2];
        int curRow = currentBox / width;
        int curCol = currentBox % width;

        //check the walls, simple deadlocks and boxes on both axes first before recursing into anything
        for (int axisIndex = 0; axisIndex < 2; ++axisIndex) {
            int numOfSimpleDeadlocks = 0;

            for (int d = 0; d < 2; ++d) {
                int deltaRow = AXES[axisIndex][d][0];
                int deltaCol = AXES[axisIndex][d][1];
                int nr = curRow + deltaRow;
                int nc = curCol + deltaCol;
                if (nr < 0 || nr >= height || nc < 0 || nc >= width) {
                    blockedAxes[axisIndex] = true;
                    break;
                }
                int dest = nr * width + nc;
                if (walls[dest]) {
                    blockedAxes[axisIndex] = true;
                    break;
                }
                if (!validBoxTiles[dest]) {
                    numOfSimpleDeadlocks += 1;
                }
                if (boxGrid[dest]) {
                    if (exploring[dest]) {
                        blockedAxes[axisIndex] = true;
                        break;
                    }
                    adjBoxesPerAxis[axisIndex].push_back(dest);
                }
            }

            if (!blockedAxes[axisIndex] && numOfSimpleDeadlocks == 2) {
                blockedAxes[axisIndex] = true;
            }

            //if this axis is open and theres no box beside it, the box can still move so its not frozen
            if (!blockedAxes[axisIndex] && adjBoxesPerAxis[axisIndex].empty()) {
                explored[currentBox] = 0;
                return false;
            }
        }

        std::vector<int8_t> savedExplored = explored;
        exploring[currentBox] = 1;

        for (int axisIndex = 0; axisIndex < 2; ++axisIndex) {
            if (!blockedAxes[axisIndex]) {
                for (int adjBox : adjBoxesPerAxis[axisIndex]) {
                    if (isFrozen(adjBox)) {
                        blockedAxes[axisIndex] = true;
                        break;
                    }
                }
            }

            if (!blockedAxes[axisIndex]) {
                exploring[currentBox] = 0;
                explored = savedExplored;
                explored[currentBox] = 0;
                return false;
            }
        }

        exploring[currentBox] = 0;
        explored[currentBox] = 1;
        return true;
    };

    if (!isFrozen(boxCoordinate)) {
        return false;
    }
    frozenBoxesOut.clear();
    int numTiles = width * height;
    for (int tile = 0; tile < numTiles; ++tile) {
        if (explored[tile] == 1) {
            frozenBoxesOut.push_back(tile);
        }
    }
    return true;
}

std::vector<std::vector<int>> generateDistanceMatrices(
    const std::vector<int>& goalState,
    int height,
    int width,
    const std::vector<uint8_t>& validBoxTiles,
    const std::vector<std::vector<int>>& neighbors
) {
    //create goalState many matrices in a vector
    int numTiles = height * width;
    std::vector<std::vector<int>> matrices(goalState.size(), std::vector<int>(numTiles, -1));

    for (std::size_t gIdx = 0; gIdx < goalState.size(); ++gIdx) {
        int goalVal = goalState[gIdx];
        auto& matrix = matrices[gIdx];

        matrix[goalVal] = 0;
        std::deque<std::pair<int, int>> frontier;
        frontier.push_back({goalVal, 0});
        std::vector<uint8_t> explored(numTiles, 0);
        explored[goalVal] = 1;

        while (!frontier.empty()) {
            int currentPos = frontier.front().first;
            int currentCtr = frontier.front().second;
            frontier.pop_front();
            int curRow = currentPos / width;
            int curCol = currentPos % width;

            for (int d = 0; d < 4; ++d) {
                int row = DIRECTION[d][0];
                int col = DIRECTION[d][1];
                int destRow = curRow + row;
                int destCol = curCol + col;
                if (destRow < 0 || destRow >= height || destCol < 0 || destCol >= width) {
                    continue;
                }
                int dest = destRow * width + destCol;
                //the player has to stand behind dest to push the box back to currentPos, so that cant be a wall
                int standRow = destRow + row;
                int standCol = destCol + col;
                if (standRow < 0 || standRow >= height || standCol < 0 || standCol >= width) {
                    continue;
                }
                int playerStand = standRow * width + standCol;
                bool canStand = false;
                for (int nb : neighbors[dest]) {
                    if (nb == playerStand) {
                        canStand = true;
                        break;
                    }
                }
                if (!explored[dest] && validBoxTiles[dest] && canStand) {
                    explored[dest] = 1;
                    int newCtr = currentCtr + 1;
                    frontier.push_back({dest, newCtr});
                    matrix[dest] = newCtr;
                }
            }
        }
    }
    return matrices;
    //expand from the goalVal to the map
}

std::vector<std::vector<int>> buildDistanceStack(
    const std::vector<std::vector<int>>& distanceMatrices,
    const std::vector<int>& goalState,
    int numTiles
) {
    std::vector<std::vector<int>> tileDistances(numTiles, std::vector<int>(goalState.size(), 999999));
    for (int tile = 0; tile < numTiles; ++tile) {
        for (std::size_t gIdx = 0; gIdx < goalState.size(); ++gIdx) {
            int d = distanceMatrices[gIdx][tile];
            tileDistances[tile][gIdx] = (d == -1) ? 999999 : d;
        }
    }
    return tileDistances;
}

int countHotspots(
    const std::vector<int>& boxCoordinates,
    const std::vector<std::vector<int>>& hotspotList
) {
    int hotspots = 0;
    for (int box : boxCoordinates) {
        const auto& trappedTiles = hotspotList[box];
        for (int trapped : trappedTiles) {
            if (containsBox(boxCoordinates, trapped)) {
                hotspots += 1;
                break;
            }
        }
    }
    return hotspots;
}

int calculateHeuristic(
    const std::vector<int>& boxCoordinates,
    const std::vector<std::vector<int>>& tileDistances,
    std::unordered_map<std::vector<int>, int, VectorHash>& heuristicCache,
    const std::vector<std::vector<int>>& hotspotList
) {
    int n = static_cast<int>(boxCoordinates.size());
    std::vector<int> u(n + 1, 0), v(n + 1, 0), p(n + 1, 0), way(n + 1, 0);

    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<int> minv(n + 1, 2000000000);
        std::vector<char> used(n + 1, 0);
        do {
            used[j0] = 1;
            int i0 = p[j0], delta = 2000000000, j1 = 0;
            const auto& rowCosts = tileDistances[boxCoordinates[i0 - 1]];
            for (int j = 1; j <= n; ++j) {
                if (!used[j]) {
                    int cur = rowCosts[j - 1] - u[i0] - v[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < delta) {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for (int j = 0; j <= n; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }

    int sumCost = 0;
    for (int j = 1; j <= n; ++j) {
        if (p[j] > 0) {
            sumCost += tileDistances[boxCoordinates[p[j] - 1]][j - 1];
        }
    }

    int hotspots = countHotspots(boxCoordinates, hotspotList);
    int heuristic = sumCost + hotspots * 10;
    heuristicCache[boxCoordinates] = heuristic;
    return heuristic;
}

std::vector<uint8_t> reachablePlayerTiles(
    int playerPosition,
    const std::vector<uint8_t>& boxGrid,
    const std::vector<std::vector<int>>& neighbors,
    int numTiles
) {
    std::vector<int> frontier;
    frontier.reserve(numTiles);
    frontier.push_back(playerPosition);
    std::vector<uint8_t> reachable(numTiles, 0);
    reachable[playerPosition] = 1;

    //looping over a list that keeps growing is faster than popleft on a deque
    for (std::size_t idx = 0; idx < frontier.size(); ++idx) {
        int current = frontier[idx];
        for (int dest : neighbors[current]) {
            if (!reachable[dest] && !boxGrid[dest]) {
                reachable[dest] = 1;
                frontier.push_back(dest);
            }
        }
    }
    return reachable;
}

void getCorralFromTile(
    int startPos,
    const std::vector<uint8_t>& boxGrid,
    const std::vector<std::vector<int>>& neighbors,
    int numTiles,
    std::vector<uint8_t>& corralMask,
    std::vector<int>& corralTiles,
    std::vector<int>& corralBoxes,
    std::vector<uint8_t>& corralBoxMask
) {
    corralTiles.clear();
    corralBoxes.clear();
    corralTiles.push_back(startPos);
    corralMask[startPos] = 1;

    for (std::size_t idx = 0; idx < corralTiles.size(); ++idx) {
        int current = corralTiles[idx];
        for (int dest : neighbors[current]) {
            if (corralMask[dest]) {
                continue;
            }
            if (boxGrid[dest]) {
                if (!corralBoxMask[dest]) {
                    corralBoxMask[dest] = 1;
                    corralBoxes.push_back(dest);
                }
            } else {
                corralMask[dest] = 1;
                corralTiles.push_back(dest);
            }
        }
    }
    std::sort(corralBoxes.begin(), corralBoxes.end());
}

bool isPICorral(
    const std::vector<uint8_t>& corralMask,
    const std::vector<int>& corralBoxes,
    const std::vector<uint8_t>& corralBoxMask,
    const std::vector<uint8_t>& reachable,
    const std::vector<uint8_t>& boxGrid,
    const std::vector<uint8_t>& walls,
    const std::vector<uint8_t>& validBoxTiles,
    int width,
    int height
) {
    bool hasValidPush = false;
    for (int box : corralBoxes) {
        int boxRow = box / width;
        int boxCol = box % width;
        for (int d = 0; d < 4; ++d) {
            int destRow = DIRECTION[d][0];
            int destCol = DIRECTION[d][1];
            int pRow = boxRow - destRow;
            int pCol = boxCol - destCol;
            int bRow = boxRow + destRow;
            int bCol = boxCol + destCol;

            if (pRow < 0 || pRow >= height || pCol < 0 || pCol >= width) continue;
            if (bRow < 0 || bRow >= height || bCol < 0 || bCol >= width) continue;

            int playerPos = pRow * width + pCol;
            int boxDest = bRow * width + bCol;

            if (walls[playerPos] || !validBoxTiles[boxDest]) {
                continue;
            }

            if (corralMask[playerPos]) {
                continue;
            }

            if (corralBoxMask[playerPos] && corralBoxMask[boxDest]) {
                continue;
            }

            if (!corralMask[boxDest]) {
                return false;
            }

            if (!reachable[playerPos] && !corralBoxMask[playerPos]) {
                return false;
            }

            if (reachable[playerPos] && !boxGrid[boxDest]) {
                hasValidPush = true;
            }
        }
    }
    return hasValidPush;
}

std::string findWalkingPath(
    int start,
    int destination,
    const std::vector<int>& boxCoordinates,
    const std::vector<std::vector<int>>& neighbors,
    int width,
    int numTiles
) {
    if (start == destination) {
        return "";
    }
    std::vector<uint8_t> boxGrid(numTiles, 0);
    for (int b : boxCoordinates) boxGrid[b] = 1;

    std::deque<int> frontier;
    frontier.push_back(start);
    std::vector<int> parentTile(numTiles, -1);
    std::vector<char> parentMove(numTiles, 0);
    parentTile[start] = start;

    while (!frontier.empty()) {
        int current = frontier.front();
        frontier.pop_front();
        int curRow = current / width;
        int curCol = current % width;

        for (int dest : neighbors[current]) {
            if (parentTile[dest] != -1 || boxGrid[dest]) {
                continue;
            }
            int destRow = (dest / width) - curRow;
            int destCol = (dest % width) - curCol;
            parentTile[dest] = current;
            parentMove[dest] = getDirectionChar(destRow, destCol);
            if (dest == destination) {
                std::string path;
                int cur = dest;
                while (cur != start) {
                    path.push_back(parentMove[cur]);
                    cur = parentTile[cur];
                }
                std::reverse(path.begin(), path.end());
                return path;
            }
            frontier.push_back(dest);
        }
    }
    return "";
}

std::vector<std::array<uint8_t, 4>> detectTunnelSquares(
    const std::vector<int>& interiorTilesList,
    const std::vector<uint8_t>& interiorTilesMask,
    const std::vector<uint8_t>& goalStateMask,
    const std::vector<uint8_t>& walls,
    int width,
    int height
) {
    int numTiles = width * height;
    std::vector<std::array<uint8_t, 4>> tunnelSquares(numTiles, {0, 0, 0, 0});
    auto isWall = [&](int r, int c) {
        if (r < 0 || r >= height || c < 0 || c >= width) return false;
        return walls[r * width + c] != 0;
    };

    for (int tile : interiorTilesList) {
        int r = tile / width;
        int c = tile % width;
        for (int d = 0; d < 4; ++d) {
            int deltaRow = DIRECTION[d][0];
            int deltaCol = DIRECTION[d][1];
            int prevRow = r - deltaRow;
            int prevCol = c - deltaCol;
            if (prevRow < 0 || prevRow >= height || prevCol < 0 || prevCol >= width) continue;
            int prevTile = prevRow * width + prevCol;
            int sideRow = -deltaCol;
            int sideCol = deltaRow;

            if (interiorTilesMask[prevTile] && !goalStateMask[tile] &&
                isWall(r + sideRow, c + sideCol) && isWall(r - sideRow, c - sideCol) &&
                isWall(prevRow + sideRow, prevCol + sideCol) && isWall(prevRow - sideRow, prevCol - sideCol)) {
                tunnelSquares[tile][d] = 1;
            }
        }
    }
    return tunnelSquares;
}

void generateHotspotTable(
    const std::vector<int>& goalState,
    const std::vector<int>& interiorTilesList,
    const std::vector<std::vector<int>>& neighbors,
    int width,
    int height,
    std::vector<uint8_t>& validBoxTilesOut,
    std::vector<std::vector<int>>& hotspotListOut
) {
    int numTiles = width * height;
    std::vector<int> frontier;
    frontier.reserve(numTiles * 4);
    std::vector<uint8_t> explored(numTiles * numTiles, 0);
    std::vector<int> exploredIndices;
    exploredIndices.reserve(numTiles * 4);
    std::vector<uint8_t> reachedMask(numTiles, 0);

    auto getReachableGoalsCount = [&](int goal, int hotspotTile, std::vector<int>& reachedOut, std::vector<uint8_t>* visitedByGoalOut) {
        //Check if a box can be pulled from a goalstate to a certain tile
        //BFS ALGO
        //explored is now tracking the player and box states for this specific target
        frontier.clear();
        for (int idx : exploredIndices) explored[idx] = 0;
        exploredIndices.clear();
        for (int rTile : reachedOut) reachedMask[rTile] = 0;
        reachedOut.clear();

        for (int dest : neighbors[goal]) {
            if (dest != hotspotTile) {
                int startState = dest * numTiles + goal;
                frontier.push_back(startState);
                explored[startState] = 1;
                exploredIndices.push_back(startState);
            }
        }

        //we are trying to get the player to pull the 
        for (std::size_t i = 0; i < frontier.size(); ++i) {
            int state = frontier[i];
            int currentPlayerPos = state / numTiles;
            int currentBoxCoords = state % numTiles;

            if (!reachedMask[currentBoxCoords]) {
                reachedMask[currentBoxCoords] = 1;
                reachedOut.push_back(currentBoxCoords);
            }
            if (visitedByGoalOut) {
                (*visitedByGoalOut)[currentPlayerPos] = 1;
                (*visitedByGoalOut)[currentBoxCoords] = 1;
            }

            int pRow = currentPlayerPos / width;
            int pCol = currentPlayerPos % width;
            int bRow = currentBoxCoords / width;
            int bCol = currentBoxCoords % width;

            // for each direction from the box, put the player in that direction and check if its valid
            int deltaRow = pRow - bRow;
            int deltaCol = pCol - bCol;
            // delta row and col is the direction of the player relative to the box
            // we want to move the player 1 backwards
            // the code below is the new state after moving the box
            if (std::abs(deltaRow) + std::abs(deltaCol) == 1) {
                int dRow = pRow + deltaRow;
                int dCol = pCol + deltaCol;
                if (dRow >= 0 && dRow < height && dCol >= 0 && dCol < width) {
                    int dest = dRow * width + dCol;
                    bool isNeighbor = false;
                    for (int nb : neighbors[currentPlayerPos]) {
                        if (nb == dest) {
                            isNeighbor = true;
                            break;
                        }
                    }
                    if (isNeighbor && dest != hotspotTile) {
                        int newState = dest * numTiles + currentPlayerPos;
                        if (!explored[newState]) {
                            explored[newState] = 1;
                            exploredIndices.push_back(newState);
                            frontier.push_back(newState);
                        }
                    }
                }
            }
            //move the player to the 4 adjacent tiles from where they are standing and check if its valid
            for (int destination : neighbors[currentPlayerPos]) {
                if (destination != currentBoxCoords && destination != hotspotTile) {
                    int startState = destination * numTiles + currentBoxCoords;
                    if (!explored[startState]) {
                        explored[startState] = 1;
                        exploredIndices.push_back(startState);
                        frontier.push_back(startState);
                    }
                }
            }
        }
    };

    // get the number of reachable goals if there are no hotspots
    std::vector<int> reachableGoalCount(numTiles, 0);
    std::vector<std::vector<int>> goalReached(goalState.size());
    std::vector<std::vector<uint8_t>> goalReachedMask(goalState.size(), std::vector<uint8_t>(numTiles, 0));
    std::vector<std::vector<uint8_t>> visitedByGoal(goalState.size(), std::vector<uint8_t>(numTiles, 0));
    std::vector<int> goalIndexByTile(numTiles, -1);

    std::vector<int> tempReached;
    for (std::size_t gIdx = 0; gIdx < goalState.size(); ++gIdx) {
        int goal = goalState[gIdx];
        goalIndexByTile[goal] = static_cast<int>(gIdx);
        getReachableGoalsCount(goal, -1, tempReached, &visitedByGoal[gIdx]);
        goalReached[gIdx] = tempReached;
        for (int tile : tempReached) {
            goalReachedMask[gIdx][tile] = 1;
            reachableGoalCount[tile] += 1;
        }
    }

    //every tile a box can be pushed onto without walking into a wall or a simple deadlock
    validBoxTilesOut.assign(numTiles, 0);
    std::vector<int> validBoxTilesList;
    for (int tile : interiorTilesList) {
        if (reachableGoalCount[tile] > 0) {
            validBoxTilesOut[tile] = 1;
            validBoxTilesList.push_back(tile);
        }
    }

    //map a tuple coordinate acting as a hotspot to a set of trapped tuple coordinates
    hotspotListOut.assign(numTiles, {});

    //test every valid floor tile as an immovable hotspot
    std::vector<int> hotspotGoalCount(numTiles, 0);
    for (int hotspotTile : validBoxTilesList) {
        for (int tile : validBoxTilesList) {
            hotspotGoalCount[tile] = 0;
        }
        for (std::size_t gIdx = 0; gIdx < goalState.size(); ++gIdx) {
            int goal = goalState[gIdx];
            if (goal == hotspotTile) {
                continue;
            }
            if (visitedByGoal[gIdx][hotspotTile]) {
                getReachableGoalsCount(goal, hotspotTile, tempReached, nullptr);
                for (int tile : tempReached) {
                    if (validBoxTilesOut[tile]) {
                        hotspotGoalCount[tile] += 1;
                    }
                }
            } else {
                for (int tile : goalReached[gIdx]) {
                    if (validBoxTilesOut[tile]) {
                        hotspotGoalCount[tile] += 1;
                    }
                }
            }
        }

        int hGoalIdx = goalIndexByTile[hotspotTile];
        for (int tile : validBoxTilesList) {
            if (tile == hotspotTile) {
                continue;
            }
            int expectedCount = reachableGoalCount[tile];
            if (hGoalIdx != -1 && goalReachedMask[hGoalIdx][tile]) {
                expectedCount -= 1;
            }
            //if the number of reachable goals dropped, the hotspot tile is blocking boxes on this tile from getting into our goal room
            if (hotspotGoalCount[tile] < expectedCount) {
                hotspotListOut[hotspotTile].push_back(tile);
            }
        }
    }
}

int getPackedBoxes(const std::vector<int>& boxCoordinates, const std::vector<uint8_t>& goalStateMask) {
    int packed = 0;
    for (int box : boxCoordinates) {
        if (goalStateMask[box]) {
            packed += 1;
        }
    }
    return packed;
}

int countRegions(
    const std::vector<int>& interiorTilesList,
    const std::vector<uint8_t>& boxGrid,
    const std::vector<std::vector<int>>& neighbors,
    int numTiles
) {
    std::vector<uint8_t> explored(numTiles, 0);
    int rooms = 0;
    std::vector<int> frontier;
    frontier.reserve(interiorTilesList.size());

    //For every single node in the map
    for (int currentPos : interiorTilesList) {
        //If its a wall, a box, or already explored, then skip it
        if (boxGrid[currentPos] || explored[currentPos]) {
            continue;
        }

        //Else, new "room" so basically fill the reigon then increment our room counter
        rooms += 1;

        //everything below is just standard flood fill bfs
        frontier.clear();
        frontier.push_back(currentPos);
        explored[currentPos] = 1;

        for (std::size_t idx = 0; idx < frontier.size(); ++idx) {
            int current = frontier[idx];
            for (int dest : neighbors[current]) {
                if (!boxGrid[dest] && !explored[dest]) {
                    explored[dest] = 1;
                    frontier.push_back(dest);
                }
            }
        }
    }

    // Return the number of rooms found
    return rooms;
}

void generateRoomsAndGraph(
    const std::vector<int>& interiorTilesList,
    const std::vector<uint8_t>& interiorTilesMask,
    const std::vector<std::vector<int>>& neighbors,
    int width,
    int height,
    std::vector<std::vector<int>>& roomsOut,
    std::vector<std::vector<int>>& roomGraphOut,
    std::vector<int>& tileToRoomOut
) {
    int numTiles = width * height;
    std::vector<uint8_t> roomTilesMask(numTiles, 0);

    auto isInterior = [&](int r, int c) {
        if (r < 0 || r >= height || c < 0 || c >= width) return false;
        return interiorTilesMask[r * width + c] != 0;
    };

    //find 2x3 and 3x2 rectangles
    for (int tile : interiorTilesList) {
        int row = tile / width;
        int col = tile % width;

        if (isInterior(row, col) && isInterior(row, col + 1) && isInterior(row, col + 2) &&
            isInterior(row + 1, col) && isInterior(row + 1, col + 1) && isInterior(row + 1, col + 2)) {
            for (int dr = 0; dr < 2; ++dr) {
                for (int dc = 0; dc < 3; ++dc) {
                    roomTilesMask[(row + dr) * width + (col + dc)] = 1;
                }
            }
        }

        if (isInterior(row, col) && isInterior(row + 1, col) && isInterior(row + 2, col) &&
            isInterior(row, col + 1) && isInterior(row + 1, col + 1) && isInterior(row + 2, col + 1)) {
            for (int dr = 0; dr < 3; ++dr) {
                for (int dc = 0; dc < 2; ++dc) {
                    roomTilesMask[(row + dr) * width + (col + dc)] = 1;
                }
            }
        }
    }

    roomsOut.clear();
    tileToRoomOut.assign(numTiles, -1);

    //destructively popping from roomTiles means we dont need an explored set
    for (int tile = 0; tile < numTiles; ++tile) {
        if (!roomTilesMask[tile]) continue;
        roomTilesMask[tile] = 0;
        int roomId = static_cast<int>(roomsOut.size());
        std::vector<int> frontier = {tile};
        tileToRoomOut[tile] = roomId;

        for (std::size_t idx = 0; idx < frontier.size(); ++idx) {
            int currentPos = frontier[idx];
            for (int dest : neighbors[currentPos]) {
                if (roomTilesMask[dest]) {
                    roomTilesMask[dest] = 0;
                    tileToRoomOut[dest] = roomId;
                    frontier.push_back(dest);
                }
            }
        }
        roomsOut.push_back(std::move(frontier));
    }

    roomGraphOut.assign(roomsOut.size(), {});
    std::vector<std::set<int>> roomGraphSets(roomsOut.size());

    for (std::size_t roomId = 0; roomId < roomsOut.size(); ++roomId) {
        std::vector<int> frontier = roomsOut[roomId];
        std::vector<uint8_t> explored(numTiles, 0);
        for (int t : frontier) explored[t] = 1;

        for (std::size_t idx = 0; idx < frontier.size(); ++idx) {
            int currentPos = frontier[idx];
            for (int dest : neighbors[currentPos]) {
                if (!explored[dest]) {
                    explored[dest] = 1;
                    if (tileToRoomOut[dest] != -1) {
                        int neighborRoom = tileToRoomOut[dest];
                        roomGraphSets[roomId].insert(neighborRoom);
                        roomGraphSets[neighborRoom].insert(static_cast<int>(roomId));
                    } else {
                        frontier.push_back(dest);
                    }
                }
            }
        }
    }

    for (std::size_t roomId = 0; roomId < roomsOut.size(); ++roomId) {
        roomGraphOut[roomId].assign(roomGraphSets[roomId].begin(), roomGraphSets[roomId].end());
    }
}

int countRoomConnectivity(
    const std::vector<uint8_t>& boxGrid,
    const std::vector<std::vector<int>>& rooms,
    const std::vector<std::vector<int>>& roomGraph,
    const std::vector<int>& tileToRoom,
    const std::vector<std::vector<int>>& neighbors,
    int numTiles
) {
    int brokenEdges = 0;
    std::vector<uint8_t> explored(numTiles, 0);
    std::vector<uint8_t> connectedRoomsMask(rooms.size(), 0);
    std::vector<int> frontier;
    frontier.reserve(numTiles);

    for (std::size_t roomId = 0; roomId < rooms.size(); ++roomId) {
        if (roomGraph[roomId].empty()) {
            continue;
        }

        std::fill(explored.begin(), explored.end(), 0);
        std::fill(connectedRoomsMask.begin(), connectedRoomsMask.end(), 0);
        std::size_t connectedCount = 0;

        for (int startPos : rooms[roomId]) {
            if (boxGrid[startPos] || explored[startPos]) {
                continue;
            }

            frontier.clear();
            frontier.push_back(startPos);
            explored[startPos] = 1;

            for (std::size_t idx = 0; idx < frontier.size(); ++idx) {
                int currentPos = frontier[idx];
                for (int dest : neighbors[currentPos]) {
                    if (!boxGrid[dest] && !explored[dest]) {
                        explored[dest] = 1;

                        int neighborRoom = tileToRoom[dest];
                        if (neighborRoom != -1) {
                            if (neighborRoom == static_cast<int>(roomId)) {
                                frontier.push_back(dest);
                            } else if (!connectedRoomsMask[neighborRoom]) {
                                connectedRoomsMask[neighborRoom] = 1;
                                connectedCount += 1;
                            }
                        } else {
                            frontier.push_back(dest);
                        }
                    }
                }
            }

            if (connectedCount == roomGraph[roomId].size()) {
                break;
            }
        }

        for (int neighborRoom : roomGraph[roomId]) {
            if (static_cast<int>(roomId) < neighborRoom && !connectedRoomsMask[neighborRoom]) {
                brokenEdges += 1;
            }
        }
    }

    return brokenEdges;
}

int countMobility(
    const std::vector<int>& boxCoordinates,
    const std::vector<uint8_t>& reachable,
    const std::vector<std::vector<int>>& neighbors
) {
    int mobility = 0;
    for (int box : boxCoordinates) {
        for (int dest : neighbors[box]) {
            if (reachable[dest]) {
                mobility += 1;
            }
        }
    }
    return mobility;
}

std::pair<FeatureSpace, FeatureSpace> getProjectedFeatureSpace(
    const FeatureSpace& rawFeatureSpace,
    const FeatureSpace& bestAncestorFeatureSpace
) {
    if (rawFeatureSpace < bestAncestorFeatureSpace) {
        return {rawFeatureSpace, rawFeatureSpace};
    }
    return {bestAncestorFeatureSpace, bestAncestorFeatureSpace};
}

struct ParentRecord {
    int previousStateId;
    int standingTile;
    int pushDirIdx;
    int pushCount;
};

struct StateNode {
    int playerPosition;
    std::vector<int> boxCoordinates;
    std::vector<uint8_t> reachable;
    FeatureSpace bestAncestor;
    ParentRecord parent;
};

std::string solveSokobanPuzzleCpp(
    int width,
    int height,
    const std::vector<std::string>& mapData,
    const std::vector<std::string>& itemsData,
    double timeLimitSeconds = 0.0
) {
    auto startTime = std::chrono::steady_clock::now();
    int numTiles = width * height;

    std::vector<int> goalState;
    std::vector<uint8_t> goalStateMask(numTiles, 0);
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            if (mapData[row][col] == '.') {
                int tile = row * width + col;
                goalState.push_back(tile);
                goalStateMask[tile] = 1;
            }
        }
    }
    std::sort(goalState.begin(), goalState.end());

    std::vector<int> boxCoordinates;
    int playerPosition = 0;
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            if (itemsData[row][col] == '$') {
                boxCoordinates.push_back(row * width + col);
            } else if (itemsData[row][col] == '@') {
                playerPosition = row * width + col;
            }
        }
    }
    std::sort(boxCoordinates.begin(), boxCoordinates.end());

    std::vector<uint8_t> walls(numTiles, 0);
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            if (mapData[row][col] == '#') {
                walls[row * width + col] = 1;
            }
        }
    }

    //walls never move, so get the neighbors of every floor tile once instead of every bfs step
    std::vector<std::vector<int>> neighbors(numTiles);
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            int tile = row * width + col;
            if (walls[tile]) {
                continue;
            }
            for (int d = 0; d < 4; ++d) {
                int nr = row + DIRECTION[d][0];
                int nc = col + DIRECTION[d][1];
                if (nr >= 0 && nr < height && nc >= 0 && nc < width && !walls[nr * width + nc]) {
                    neighbors[tile].push_back(nr * width + nc);
                }
            }
        }
    }

    std::vector<uint8_t> emptyBoxGrid(numTiles, 0);
    std::vector<uint8_t> interiorTilesMask = reachablePlayerTiles(playerPosition, emptyBoxGrid, neighbors, numTiles);
    std::vector<int> interiorTilesList;
    for (int t = 0; t < numTiles; ++t) {
        if (interiorTilesMask[t]) {
            interiorTilesList.push_back(t);
        }
    }

    std::vector<uint8_t> validBoxTiles;
    std::vector<std::vector<int>> hotspotDictionary;
    generateHotspotTable(goalState, interiorTilesList, neighbors, width, height, validBoxTiles, hotspotDictionary);

    std::vector<std::vector<int>> distances = generateDistanceMatrices(goalState, height, width, validBoxTiles, neighbors);
    std::vector<std::vector<int>> tileDistances = buildDistanceStack(distances, goalState, numTiles);
    std::vector<std::vector<std::vector<int>>> freezeSubset(numTiles);

    std::vector<std::vector<int>> rooms;
    std::vector<std::vector<int>> roomGraph;
    std::vector<int> tileToRoom;
    generateRoomsAndGraph(interiorTilesList, interiorTilesMask, neighbors, width, height, rooms, roomGraph, tileToRoom);

    std::unordered_map<std::vector<int>, int, VectorHash> heuristicCache;
    std::unordered_map<std::vector<int>, FeatureSpace, VectorHash> featureSpaceCache;

    std::vector<uint8_t> startBoxGrid(numTiles, 0);
    for (int b : boxCoordinates) startBoxGrid[b] = 1;
    std::vector<uint8_t> startReachable = reachablePlayerTiles(playerPosition, startBoxGrid, neighbors, numTiles);
    auto tunnelTiles = detectTunnelSquares(interiorTilesList, interiorTilesMask, goalStateMask, walls, width, height);

    auto getFeatureSpace = [&](const std::vector<int>& boxes, const std::vector<uint8_t>& boxGrid) -> FeatureSpace {
        auto it = featureSpaceCache.find(boxes);
        if (it != featureSpaceCache.end()) {
            return it->second;
        }
        int packed = getPackedBoxes(boxes, goalStateMask);
        int connectivity = countRegions(interiorTilesList, boxGrid, neighbors, numTiles);
        int roomConnectivity = countRoomConnectivity(boxGrid, rooms, roomGraph, tileToRoom, neighbors, numTiles);
        FeatureSpace rawFeatureSpace = {-packed, connectivity, roomConnectivity};
        featureSpaceCache[boxes] = rawFeatureSpace;
        return rawFeatureSpace;
    };

    std::vector<StateNode> states;
    states.reserve(65536);

    std::unordered_map<std::vector<int>, std::vector<std::vector<uint8_t>>, VectorHash> exploredReigons;
    exploredReigons[boxCoordinates].push_back(startReachable);

    FeatureSpace startFeatureSpace = getFeatureSpace(boxCoordinates, startBoxGrid);
    int heuristic = calculateHeuristic(boxCoordinates, tileDistances, heuristicCache, hotspotDictionary);
    int mobility = countMobility(boxCoordinates, startReachable, neighbors);

    states.push_back(StateNode{
        playerPosition,
        boxCoordinates,
        startReachable,
        startFeatureSpace,
        ParentRecord{-1, -1, -1, 0}
    });

    using HeapItem = std::tuple<int, int, int, int>;
    using MinHeap = std::priority_queue<HeapItem, std::vector<HeapItem>, std::greater<HeapItem>>;
    std::map<FeatureSpace, MinHeap> featureSpaceQueues;
    std::vector<FeatureSpace> activeFeatureSpaces;

    int stateCounter = 0;
    featureSpaceQueues[startFeatureSpace].push({heuristic, -mobility, -stateCounter, 0});
    activeFeatureSpaces.push_back(startFeatureSpace);
    stateCounter += 1;

    std::vector<uint8_t> currentBoxGrid(numTiles, 0);
    std::vector<uint8_t> macroBoxGrid(numTiles, 0);
    std::vector<uint8_t> exploredCorral(numTiles, 0);
    std::vector<uint8_t> corralMask(numTiles, 0);
    std::vector<uint8_t> corralBoxMask(numTiles, 0);
    std::vector<int> corralTiles;
    std::vector<int> corralBoxes;
    std::vector<int8_t> freezeExplored(numTiles, -1);
    std::vector<uint8_t> exploring(numTiles, 0);
    std::vector<int> frozenBoxes;

    while (!activeFeatureSpaces.empty()) {
        std::vector<FeatureSpace> roundRobinKeys = activeFeatureSpaces;
        for (const auto& featureSpaceKey : roundRobinKeys) {
            if (timeLimitSeconds > 0.0) {
                double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count();
                if (elapsed > timeLimitSeconds) {
                    return "";
                }
            }

            auto qIt = featureSpaceQueues.find(featureSpaceKey);
            if (qIt == featureSpaceQueues.end() || qIt->second.empty()) {
                continue;
            }
            int currentStateId = std::get<3>(qIt->second.top());
            qIt->second.pop();
            if (qIt->second.empty()) {
                featureSpaceQueues.erase(qIt);
                activeFeatureSpaces.erase(
                    std::remove(activeFeatureSpaces.begin(), activeFeatureSpaces.end(), featureSpaceKey),
                    activeFeatureSpaces.end()
                );
            }

            std::vector<int> currentBoxCoords = states[currentStateId].boxCoordinates;
            std::vector<uint8_t> reachable = std::move(states[currentStateId].reachable);
            FeatureSpace currentBestAncestor = states[currentStateId].bestAncestor;

            if (currentBoxCoords == goalState) {
                std::vector<ParentRecord> transitions;
                int key = currentStateId;
                while (states[key].parent.previousStateId != -1) {
                    transitions.push_back(states[key].parent);
                    key = states[key].parent.previousStateId;
                }
                std::reverse(transitions.begin(), transitions.end());

                std::string winPath;
                for (const auto& tr : transitions) {
                    const auto& prevState = states[tr.previousStateId];
                    std::string walk = findWalkingPath(
                        prevState.playerPosition,
                        tr.standingTile,
                        prevState.boxCoordinates,
                        neighbors,
                        width,
                        numTiles
                    );
                    winPath.append(walk);
                    char pushChar = getDirectionChar(DIRECTION[tr.pushDirIdx][0], DIRECTION[tr.pushDirIdx][1]);
                    winPath.append(tr.pushCount, pushChar);
                }

                std::cout << "Total states explored: " << states.size() << "\n";
                std::cout << "Total unique box configurations explored: " << exploredReigons.size() << std::endl;
                return winPath;
            }

            std::fill(currentBoxGrid.begin(), currentBoxGrid.end(), 0);
            for (int b : currentBoxCoords) currentBoxGrid[b] = 1;

            std::vector<int> boxesToExplore = currentBoxCoords;
            std::fill(exploredCorral.begin(), exploredCorral.end(), 0);

            //only start a corral from the tile right behind a box the player can push right now
            //a pi corral always has one of these, so flooding any other pocket is just wasted work
            std::vector<int> corralSeeds;
            for (int box : currentBoxCoords) {
                int bRow = box / width;
                int bCol = box % width;
                for (int d = 0; d < 4; ++d) {
                    int destRow = DIRECTION[d][0];
                    int destCol = DIRECTION[d][1];
                    int sRow = bRow - destRow;
                    int sCol = bCol - destCol;
                    int dRow = bRow + destRow;
                    int dCol = bCol + destCol;
                    if (sRow < 0 || sRow >= height || sCol < 0 || sCol >= width) continue;
                    if (dRow < 0 || dRow >= height || dCol < 0 || dCol >= width) continue;
                    int standingTile = sRow * width + sCol;
                    int boxDestination = dRow * width + dCol;
                    if (reachable[standingTile] && validBoxTiles[boxDestination]) {
                        if (!reachable[boxDestination] && !currentBoxGrid[boxDestination]) {
                            if (std::find(corralSeeds.begin(), corralSeeds.end(), boxDestination) == corralSeeds.end()) {
                                corralSeeds.push_back(boxDestination);
                            }
                        }
                    }
                }
            }

            for (int currentPos : corralSeeds) {
                if (exploredCorral[currentPos]) {
                    continue;
                }
                std::fill(corralMask.begin(), corralMask.end(), 0);
                std::fill(corralBoxMask.begin(), corralBoxMask.end(), 0);
                getCorralFromTile(
                    currentPos, currentBoxGrid, neighbors, numTiles,
                    corralMask, corralTiles, corralBoxes, corralBoxMask
                );
                for (int ct : corralTiles) exploredCorral[ct] = 1;

                //no boxes or every box in the corral means restricting to it prunes nothing
                if (corralBoxes.empty() || corralBoxes.size() == currentBoxCoords.size()) {
                    continue;
                }

                //every box is already on a goal and theres no empty goal inside, so nothing to do here
                bool allBoxesOnGoal = true;
                for (int cb : corralBoxes) {
                    if (!goalStateMask[cb]) {
                        allBoxesOnGoal = false;
                        break;
                    }
                }
                bool corralHasGoal = false;
                if (allBoxesOnGoal) {
                    for (int ct : corralTiles) {
                        if (goalStateMask[ct]) {
                            corralHasGoal = true;
                            break;
                        }
                    }
                    if (!corralHasGoal) {
                        continue;
                    }
                }

                if (isPICorral(
                    corralMask, corralBoxes, corralBoxMask, reachable,
                    currentBoxGrid, walls, validBoxTiles, width, height
                )) {
                    boxesToExplore = corralBoxes;
                    break;
                }
            }

            struct MacroEntry {
                int parentStateId;
                int currentBox;
                std::vector<int> boxCoords;
                std::vector<uint8_t> reachable;
            };

            for (int box : boxesToExplore) {
                std::vector<MacroEntry> macroFrontier;
                macroFrontier.push_back({currentStateId, box, currentBoxCoords, reachable});

                for (std::size_t mIdx = 0; mIdx < macroFrontier.size(); ++mIdx) {
                    int macroParentStateId = macroFrontier[mIdx].parentStateId;
                    int currentBox = macroFrontier[mIdx].currentBox;
                    std::vector<int> macroBoxCoords = macroFrontier[mIdx].boxCoords;
                    std::vector<uint8_t> macroReachable = macroFrontier[mIdx].reachable;

                    std::fill(macroBoxGrid.begin(), macroBoxGrid.end(), 0);
                    for (int b : macroBoxCoords) macroBoxGrid[b] = 1;

                    int curBoxRow = currentBox / width;
                    int curBoxCol = currentBox % width;

                    for (int d = 0; d < 4; ++d) {
                        int destRow = DIRECTION[d][0];
                        int destCol = DIRECTION[d][1];
                        int sRow = curBoxRow - destRow;
                        int sCol = curBoxCol - destCol;
                        int bRow = curBoxRow + destRow;
                        int bCol = curBoxCol + destCol;
                        if (sRow < 0 || sRow >= height || sCol < 0 || sCol >= width) continue;
                        if (bRow < 0 || bRow >= height || bCol < 0 || bCol >= width) continue;

                        int standingTile = sRow * width + sCol;
                        //If we cant reach the position of the player after pushing the box, cant push so prune
                        if (!macroReachable[standingTile]) {
                            continue;
                        }

                        int boxDestination = bRow * width + bCol;
                        //if pushing into a wall, a simple deadlock or another box, prune
                        if (!validBoxTiles[boxDestination] || macroBoxGrid[boxDestination]) {
                            continue;
                        }

                        //after the push the player ends up where the box was
                        int newPlayerPos = currentBox;
                        int pushCount = 1;
                        int movingBox = boxDestination;
                        //keep pushing the box through the tunnel until it comes out the other side or hits something
                        while (tunnelTiles[movingBox][d]) {
                            int nextRow = (movingBox / width) + destRow;
                            int nextCol = (movingBox % width) + destCol;
                            if (nextRow < 0 || nextRow >= height || nextCol < 0 || nextCol >= width) break;
                            int nextBox = nextRow * width + nextCol;
                            if (!validBoxTiles[nextBox] || macroBoxGrid[nextBox]) {
                                break;
                            }
                            newPlayerPos = movingBox;
                            movingBox = nextBox;
                            pushCount += 1;
                        }
                        boxDestination = movingBox;

                        std::vector<int> newBoxCoords = macroBoxCoords;
                        auto bIt = std::lower_bound(newBoxCoords.begin(), newBoxCoords.end(), currentBox);
                        newBoxCoords.erase(bIt);
                        newBoxCoords.insert(
                            std::lower_bound(newBoxCoords.begin(), newBoxCoords.end(), boxDestination),
                            boxDestination
                        );

                        bool alreadyExplored = false;
                        //if the boxes positions have been seen before
                        auto regIt = exploredReigons.find(newBoxCoords);
                        if (regIt != exploredReigons.end()) {
                            //if the player is in the same "room/reigon" then its the exact same state, so prune 
                            for (const auto& exploredRegion : regIt->second) {
                                if (exploredRegion[newPlayerPos]) {
                                    alreadyExplored = true;
                                    break;
                                }
                            }
                        }
                        if (alreadyExplored) {
                            continue;
                        }

                        macroBoxGrid[currentBox] = 0;
                        macroBoxGrid[boxDestination] = 1;

                        int newHeuristic = 0;
                        auto hIt = heuristicCache.find(newBoxCoords);
                        if (hIt != heuristicCache.end()) {
                            newHeuristic = hIt->second;
                        } else {
                            //check the freeze deadlocks first so we dont waste time on the heuristic of a dead state
                            if (hasFreezeSubset(boxDestination, newBoxCoords, freezeSubset)) {
                                macroBoxGrid[boxDestination] = 0;
                                macroBoxGrid[currentBox] = 1;
                                continue;
                            }

                            std::fill(freezeExplored.begin(), freezeExplored.end(), -1);
                            std::fill(exploring.begin(), exploring.end(), 0);
                            if (detectBlockedAxes(
                                boxDestination, width, height, validBoxTiles, walls,
                                freezeExplored, macroBoxGrid, exploring, frozenBoxes
                            )) {
                                bool anyNotOnGoal = false;
                                for (int fb : frozenBoxes) {
                                    if (!goalStateMask[fb]) {
                                        anyNotOnGoal = true;
                                        break;
                                    }
                                }
                                if (anyNotOnGoal) {
                                    if (frozenBoxes.size() <= 4) {
                                        std::sort(frozenBoxes.begin(), frozenBoxes.end());
                                        //save the pattern under every box in it so hasFreezeSubset only has to check the pushed box
                                        for (int frozenBox : frozenBoxes) {
                                            auto& list = freezeSubset[frozenBox];
                                            if (std::find(list.begin(), list.end(), frozenBoxes) == list.end()) {
                                                list.push_back(frozenBoxes);
                                            }
                                        }
                                    }
                                    macroBoxGrid[boxDestination] = 0;
                                    macroBoxGrid[currentBox] = 1;
                                    continue;
                                }
                            }

                            newHeuristic = calculateHeuristic(newBoxCoords, tileDistances, heuristicCache, hotspotDictionary);
                        }

                        if (newHeuristic >= 999999) {
                            macroBoxGrid[boxDestination] = 0;
                            macroBoxGrid[currentBox] = 1;
                            continue;
                        }

                        std::vector<uint8_t> newReachable = reachablePlayerTiles(newPlayerPos, macroBoxGrid, neighbors, numTiles);

                        //add the new reachable region to the list of explored regions for this box configuration
                        exploredReigons[newBoxCoords].push_back(newReachable);

                        FeatureSpace rawFeatureSpace = getFeatureSpace(newBoxCoords, macroBoxGrid);
                        auto projPair = getProjectedFeatureSpace(rawFeatureSpace, currentBestAncestor);
                        FeatureSpace projectedFeatureSpace = projPair.first;
                        FeatureSpace newBestAncestor = projPair.second;
                        int newMobility = countMobility(newBoxCoords, newReachable, neighbors);

                        macroBoxGrid[boxDestination] = 0;
                        macroBoxGrid[currentBox] = 1;

                        int newStateId = static_cast<int>(states.size());
                        states.push_back(StateNode{
                            newPlayerPos,
                            newBoxCoords,
                            newReachable,
                            newBestAncestor,
                            ParentRecord{macroParentStateId, standingTile, d, pushCount}
                        });

                        //negative counter so ties pop the newest state first, keeps following 1 push line instead of jumping around
                        auto& q = featureSpaceQueues[projectedFeatureSpace];
                        if (q.empty()) {
                            activeFeatureSpaces.push_back(projectedFeatureSpace);
                        }
                        q.push({newHeuristic, -newMobility, -stateCounter, newStateId});
                        stateCounter += 1;

                        macroFrontier.push_back({newStateId, boxDestination, std::move(newBoxCoords), std::move(newReachable)});
                    }
                }
            }
        }
    }

    return "";
}

static std::string lastSolutionBuffer;

extern "C" {
#ifdef _WIN32
__declspec(dllexport)
#endif
const char* solve_puzzle_c(int width, int height, const char* mapFlat, const char* itemsFlat, double timeLimitSeconds) {
    std::vector<std::string> mapData(height, std::string(width, ' '));
    std::vector<std::string> itemsData(height, std::string(width, ' '));
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            mapData[r][c] = mapFlat[r * width + c];
            itemsData[r][c] = itemsFlat[r * width + c];
        }
    }
    lastSolutionBuffer = solveSokobanPuzzleCpp(width, height, mapData, itemsData, timeLimitSeconds);
    return lastSolutionBuffer.c_str();
}
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int width = 0, height = 0;
    double timeLimitSeconds = 0.0;
    if (!(std::cin >> width >> height >> timeLimitSeconds)) {
        return 0;
    }
    std::string dummy;
    std::getline(std::cin, dummy);

    std::vector<std::string> mapData(height);
    std::vector<std::string> itemsData(height);
    for (int r = 0; r < height; ++r) {
        std::getline(std::cin, mapData[r]);
        if (static_cast<int>(mapData[r].size()) < width) {
            mapData[r].resize(width, ' ');
        }
    }
    for (int r = 0; r < height; ++r) {
        std::getline(std::cin, itemsData[r]);
        if (static_cast<int>(itemsData[r].size()) < width) {
            itemsData[r].resize(width, ' ');
        }
    }

    std::string solution = solveSokobanPuzzleCpp(width, height, mapData, itemsData, timeLimitSeconds);
    std::cout << "SOLUTION:" << solution << "\n";
    return 0;
}
