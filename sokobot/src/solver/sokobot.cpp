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

using FeatureSpace = std::tuple<int, int, int, int>;
using FullFeatures = std::tuple<int, int, int, int, int>;

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
    const std::vector<uint8_t>& boxGrid,
    const std::vector<std::vector<int>>& hotspotList
) {
    int hotspots = 0;
    for (int box : boxCoordinates) {
        const auto& trappedTiles = hotspotList[box];
        for (int trapped : trappedTiles) {
            if (boxGrid[trapped]) {
                hotspots += 1;
                break;
            }
        }
    }
    return hotspots;
}

int calculateHeuristic(
    const std::vector<int>& boxCoordinates,
    const std::vector<uint8_t>& boxGrid,
    const std::vector<std::vector<int>>& tileDistances,
    std::unordered_map<std::vector<int>, int, VectorHash>& heuristicCache,
    const std::vector<std::vector<int>>& hotspotList
) {
    int n = static_cast<int>(boxCoordinates.size());
    thread_local std::vector<int> u, v, p, way, minv;
    thread_local std::vector<char> used;
    u.assign(n + 1, 0);
    v.assign(n + 1, 0);
    p.assign(n + 1, 0);
    way.assign(n + 1, 0);

    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        minv.assign(n + 1, 2000000000);
        used.assign(n + 1, 0);
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

    int hotspots = countHotspots(boxCoordinates, boxGrid, hotspotList);
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
    thread_local std::vector<int> frontier;
    frontier.clear();
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

void generateBasinsAndPlan(
    const std::vector<int>& goalState,
    const std::vector<uint8_t>& goalStateMask,
    const std::vector<int>& boxCoordinates,
    const std::vector<uint8_t>& validBoxTiles,
    const std::vector<std::vector<int>>& rooms,
    const std::vector<std::vector<int>>& neighbors,
    int width,
    int height,
    std::vector<uint8_t>& sinkBasinMaskOut,
    std::vector<int>& packingPlanOut,
    std::vector<std::vector<int>>& mustBeFilledBeforeOut
) {
    int numTiles = width * height;
    std::vector<std::vector<uint8_t>> basins(goalState.size(), std::vector<uint8_t>(numTiles, 0));

    std::vector<int> frontier;
    frontier.reserve(numTiles * 4);
    std::vector<uint8_t> explored(numTiles * numTiles, 0);
    std::vector<int> exploredIndices;
    exploredIndices.reserve(numTiles * 4);

    for (std::size_t gIdx = 0; gIdx < goalState.size(); ++gIdx) {
        int goal = goalState[gIdx];
        auto& basin = basins[gIdx];
        basin[goal] = 1;

        frontier.clear();
        for (int idx : exploredIndices) explored[idx] = 0;
        exploredIndices.clear();

        for (int dest : neighbors[goal]) {
            if (!goalStateMask[dest]) {
                int startState = dest * numTiles + goal;
                frontier.push_back(startState);
                explored[startState] = 1;
                exploredIndices.push_back(startState);
            }
        }

        for (std::size_t i = 0; i < frontier.size(); ++i) {
            int state = frontier[i];
            int currentPlayerPos = state / numTiles;
            int currentBoxCoords = state % numTiles;

            int pRow = currentPlayerPos / width;
            int pCol = currentPlayerPos % width;
            int bRow = currentBoxCoords / width;
            int bCol = currentBoxCoords % width;
            int deltaRow = pRow - bRow;
            int deltaCol = pCol - bCol;

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
                    if (isNeighbor && !goalStateMask[dest] && validBoxTiles[currentPlayerPos]) {
                        int newState = dest * numTiles + currentPlayerPos;
                        if (!explored[newState]) {
                            explored[newState] = 1;
                            exploredIndices.push_back(newState);
                            basin[currentPlayerPos] = 1;
                            frontier.push_back(newState);
                        }
                    }
                }
            }

            for (int destination : neighbors[currentPlayerPos]) {
                if (destination != currentBoxCoords && !goalStateMask[destination]) {
                    int nextState = destination * numTiles + currentBoxCoords;
                    if (!explored[nextState]) {
                        explored[nextState] = 1;
                        exploredIndices.push_back(nextState);
                        frontier.push_back(nextState);
                    }
                }
            }
        }
    }

    sinkBasinMaskOut.assign(numTiles, 0);
    int bestBoxCount = -1;
    int bestBasinSize = -1;
    int bestRoomId = -1;

    for (std::size_t roomId = 0; roomId < rooms.size(); ++roomId) {
        const auto& roomTiles = rooms[roomId];
        std::vector<uint8_t> associatedBasin(numTiles, 0);
        for (int t : roomTiles) associatedBasin[t] = 1;

        for (std::size_t gIdx = 0; gIdx < goalState.size(); ++gIdx) {
            bool intersects = false;
            for (int t : roomTiles) {
                if (basins[gIdx][t]) {
                    intersects = true;
                    break;
                }
            }
            if (intersects) {
                for (int t = 0; t < numTiles; ++t) {
                    if (basins[gIdx][t]) {
                        associatedBasin[t] = 1;
                    }
                }
            }
        }

        int boxCount = 0;
        for (int b : boxCoordinates) {
            if (associatedBasin[b]) boxCount += 1;
        }
        int basinSize = 0;
        for (int t = 0; t < numTiles; ++t) {
            if (associatedBasin[t]) basinSize += 1;
        }

        if (boxCount > bestBoxCount || (boxCount == bestBoxCount && basinSize > bestBasinSize)) {
            bestBoxCount = boxCount;
            bestBasinSize = basinSize;
            bestRoomId = static_cast<int>(roomId);
            sinkBasinMaskOut = associatedBasin;
        }
    }

    if (bestBasinSize <= 0) {
        for (std::size_t gIdx = 0; gIdx < goalState.size(); ++gIdx) {
            for (int t = 0; t < numTiles; ++t) {
                if (basins[gIdx][t]) {
                    sinkBasinMaskOut[t] = 1;
                }
            }
        }
    }

    std::vector<uint8_t> sinkRoomMask = sinkBasinMaskOut;
    for (int g : goalState) {
        sinkRoomMask[g] = 0;
    }

    std::vector<int> goalDistToSink(numTiles, 999999);
    {
        std::vector<int> bfsQ;
        bfsQ.reserve(numTiles);
        for (int t = 0; t < numTiles; ++t) {
            if (sinkRoomMask[t]) {
                goalDistToSink[t] = 0;
                bfsQ.push_back(t);
            }
        }
        for (std::size_t idx = 0; idx < bfsQ.size(); ++idx) {
            int curr = bfsQ[idx];
            int nextDist = goalDistToSink[curr] + 1;
            for (int nb : neighbors[curr]) {
                if (goalDistToSink[nb] > nextDist) {
                    goalDistToSink[nb] = nextDist;
                    bfsQ.push_back(nb);
                }
            }
        }
    }

    std::vector<uint8_t> playerReachFromSink(numTiles, 0);
    std::vector<int> playerReachQueue;
    playerReachQueue.reserve(numTiles);

    auto pullReachability = [&](int goal, const std::vector<uint8_t>& blockedMask, std::vector<uint8_t>& reachedOut, bool requireSinkPlayer) -> std::pair<bool, int> {
        frontier.clear();
        for (int idx : exploredIndices) explored[idx] = 0;
        exploredIndices.clear();
        std::fill(reachedOut.begin(), reachedOut.end(), 0);
        reachedOut[goal] = 1;
        int reachedCount = 1;
        bool reachesSink = false;

        if (requireSinkPlayer) {
            std::fill(playerReachFromSink.begin(), playerReachFromSink.end(), 0);
            playerReachQueue.clear();
            for (int t = 0; t < numTiles; ++t) {
                if (sinkRoomMask[t] && !blockedMask[t] && t != goal) {
                    playerReachFromSink[t] = 1;
                    playerReachQueue.push_back(t);
                }
            }
            for (std::size_t idx = 0; idx < playerReachQueue.size(); ++idx) {
                int curr = playerReachQueue[idx];
                for (int nb : neighbors[curr]) {
                    if (!playerReachFromSink[nb] && !blockedMask[nb] && nb != goal) {
                        playerReachFromSink[nb] = 1;
                        playerReachQueue.push_back(nb);
                    }
                }
            }
            for (int dest : neighbors[goal]) {
                if (!blockedMask[dest] && playerReachFromSink[dest]) {
                    int startState = dest * numTiles + goal;
                    frontier.push_back(startState);
                    explored[startState] = 1;
                    exploredIndices.push_back(startState);
                }
            }
        } else {
            for (int dest : neighbors[goal]) {
                if (!blockedMask[dest]) {
                    int startState = dest * numTiles + goal;
                    frontier.push_back(startState);
                    explored[startState] = 1;
                    exploredIndices.push_back(startState);
                }
            }
        }

        for (std::size_t i = 0; i < frontier.size(); ++i) {
            int state = frontier[i];
            int currentPlayerPos = state / numTiles;
            int currentBoxCoords = state % numTiles;

            int pRow = currentPlayerPos / width;
            int pCol = currentPlayerPos % width;
            int bRow = currentBoxCoords / width;
            int bCol = currentBoxCoords % width;
            int deltaRow = pRow - bRow;
            int deltaCol = pCol - bCol;

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
                    if (isNeighbor && !blockedMask[dest] && validBoxTiles[currentPlayerPos]) {
                        int newState = dest * numTiles + currentPlayerPos;
                        if (!explored[newState]) {
                            explored[newState] = 1;
                            exploredIndices.push_back(newState);
                            if (!reachedOut[currentPlayerPos]) {
                                reachedOut[currentPlayerPos] = 1;
                                reachedCount += 1;
                                if (sinkRoomMask[currentPlayerPos]) {
                                    reachesSink = true;
                                }
                            }
                            frontier.push_back(newState);
                        }
                    }
                }
            }

            for (int destination : neighbors[currentPlayerPos]) {
                if (destination != currentBoxCoords && !blockedMask[destination]) {
                    int nextState = destination * numTiles + currentBoxCoords;
                    if (!explored[nextState]) {
                        explored[nextState] = 1;
                        exploredIndices.push_back(nextState);
                        frontier.push_back(nextState);
                    }
                }
            }
        }

        return {reachesSink, reachedCount};
    };

    std::vector<int> remainingGoals = goalState;
    std::vector<int> pulledOrder;
    std::vector<uint8_t> blockedMask(numTiles, 0);
    std::vector<uint8_t> reachedTiles(numTiles, 0);

    while (!remainingGoals.empty()) {
        int bestGoal = -1;
        std::tuple<int, int, int, int> bestScore = {-1, -999999, -1, -1};

        for (int pass = 0; pass < 2; ++pass) {
            bool requireSinkPlayer = (pass == 0);
            for (int goal : remainingGoals) {
                std::fill(blockedMask.begin(), blockedMask.end(), 0);
                for (int other : remainingGoals) {
                    if (other != goal) blockedMask[other] = 1;
                }
                auto res = pullReachability(goal, blockedMask, reachedTiles, requireSinkPlayer);
                if (pass == 0 && !res.first) {
                    continue;
                }
                int openNeighbors = static_cast<int>(neighbors[goal].size());
                std::tuple<int, int, int, int> score;
                if (res.first) {
                    score = {1, -goalDistToSink[goal], openNeighbors, res.second};
                } else {
                    score = {0, res.second, -goalDistToSink[goal], openNeighbors};
                }
                if (score > bestScore) {
                    bestScore = score;
                    bestGoal = goal;
                }
            }
            if (bestGoal != -1) {
                break;
            }
        }

        pulledOrder.push_back(bestGoal);
        remainingGoals.erase(std::remove(remainingGoals.begin(), remainingGoals.end(), bestGoal), remainingGoals.end());
    }

    packingPlanOut.assign(pulledOrder.rbegin(), pulledOrder.rend());

    mustBeFilledBeforeOut.assign(numTiles, {});
    for (std::size_t i = 0; i < packingPlanOut.size(); ++i) {
        int deeperGoal = packingPlanOut[i];
        std::fill(blockedMask.begin(), blockedMask.end(), 0);
        for (std::size_t k = 0; k < i; ++k) {
            blockedMask[packingPlanOut[k]] = 1;
        }
        bool useSinkPlayer = true;
        auto prefixRes = pullReachability(deeperGoal, blockedMask, reachedTiles, true);
        if (!prefixRes.first) {
            std::fill(blockedMask.begin(), blockedMask.end(), 0);
            useSinkPlayer = false;
            auto baseRes = pullReachability(deeperGoal, blockedMask, reachedTiles, false);
            if (!baseRes.first) {
                continue;
            }
        }

        for (std::size_t j = i + 1; j < packingPlanOut.size(); ++j) {
            int laterGoal = packingPlanOut[j];
            blockedMask[laterGoal] = 1;
            auto blockedRes = pullReachability(deeperGoal, blockedMask, reachedTiles, useSinkPlayer);
            if (!blockedRes.first) {
                if (std::find(mustBeFilledBeforeOut[laterGoal].begin(), mustBeFilledBeforeOut[laterGoal].end(), deeperGoal) == mustBeFilledBeforeOut[laterGoal].end()) {
                    mustBeFilledBeforeOut[laterGoal].push_back(deeperGoal);
                }
                std::vector<uint8_t> cutComp(numTiles, 0);
                std::vector<int> cutQ = {laterGoal};
                cutComp[laterGoal] = 1;
                for (std::size_t qIdx = 0; qIdx < cutQ.size(); ++qIdx) {
                    int curr = cutQ[qIdx];
                    int cRow = curr / width;
                    int cCol = curr % width;
                    for (int dr = -1; dr <= 1; ++dr) {
                        for (int dc = -1; dc <= 1; ++dc) {
                            if (dr == 0 && dc == 0) continue;
                            int nr = cRow + dr;
                            int nc = cCol + dc;
                            if (nr >= 0 && nr < height && nc >= 0 && nc < width) {
                                int nb = nr * width + nc;
                                if (blockedMask[nb] && !cutComp[nb]) {
                                    cutComp[nb] = 1;
                                    cutQ.push_back(nb);
                                }
                            }
                        }
                    }
                }
                for (std::size_t m = i + 1; m < j; ++m) {
                    int partnerGoal = packingPlanOut[m];
                    if (blockedMask[partnerGoal] && cutComp[partnerGoal]) {
                        blockedMask[partnerGoal] = 0;
                        auto partnerRes = pullReachability(deeperGoal, blockedMask, reachedTiles, useSinkPlayer);
                        if (partnerRes.first) {
                            if (std::find(mustBeFilledBeforeOut[partnerGoal].begin(), mustBeFilledBeforeOut[partnerGoal].end(), deeperGoal) == mustBeFilledBeforeOut[partnerGoal].end()) {
                                mustBeFilledBeforeOut[partnerGoal].push_back(deeperGoal);
                            }
                        }
                        blockedMask[partnerGoal] = 1;
                    }
                }
                blockedMask[laterGoal] = 0;
            }
        }
    }

    for (int g : goalState) {
        sinkBasinMaskOut[g] = 0;
    }
}

void computePlanPackingAndOOP(
    const std::vector<int>& boxCoordinates,
    const std::vector<uint8_t>& boxGrid,
    const std::vector<uint8_t>& goalStateMask,
    const std::vector<uint8_t>& sinkBasinMask,
    const std::vector<int>& packingPlan,
    const std::vector<std::vector<int>>& mustBeFilledBefore,
    int& packedOut,
    int& oopOut
) {
    int packed = 0;
    thread_local std::vector<uint8_t> isPackedGoal;
    isPackedGoal.assign(boxGrid.size(), 0);

    for (int goal : packingPlan) {
        if (!goalStateMask[goal]) {
            bool allSubsequentGoalsPacked = true;
            for (int g : packingPlan) {
                if (goalStateMask[g]) {
                    const auto& deps = mustBeFilledBefore[g];
                    if (std::find(deps.begin(), deps.end(), goal) != deps.end() && !isPackedGoal[g]) {
                        allSubsequentGoalsPacked = false;
                        break;
                    }
                }
            }
            if (boxGrid[goal] || allSubsequentGoalsPacked) {
                isPackedGoal[goal] = 1;
                if (boxGrid[goal] && !allSubsequentGoalsPacked) {
                    packed += 1;
                }
            }
            continue;
        }
        if (!boxGrid[goal]) {
            continue;
        }
        bool dependenciesMet = true;
        for (int reqGoal : mustBeFilledBefore[goal]) {
            if (!isPackedGoal[reqGoal]) {
                dependenciesMet = false;
                break;
            }
        }
        if (dependenciesMet) {
            isPackedGoal[goal] = 1;
            packed += 1;
        }
    }

    int oop = 0;
    for (int box : boxCoordinates) {
        if (!isPackedGoal[box] && !sinkBasinMask[box]) {
            oop += 1;
        }
    }

    packedOut = packed;
    oopOut = oop;
}

int countRegions(
    const std::vector<int>& interiorTilesList,
    const std::vector<uint8_t>& boxGrid,
    const std::vector<std::vector<int>>& neighbors,
    int numTiles
) {
    thread_local std::vector<uint8_t> explored;
    explored.assign(numTiles, 0);
    int rooms = 0;
    thread_local std::vector<int> frontier;

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
    thread_local std::vector<uint8_t> explored;
    thread_local std::vector<uint8_t> connectedRoomsMask;
    thread_local std::vector<int> frontier;
    explored.resize(numTiles);
    connectedRoomsMask.resize(rooms.size());

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

std::vector<std::vector<std::vector<int>>> generateDeadlockPatterns(
    const std::vector<int>& interiorTilesList,
    const std::vector<uint8_t>& validBoxTiles,
    const std::vector<uint8_t>& goalStateMask,
    const std::vector<uint8_t>& walls,
    int width,
    int height
) {
    int numTiles = width * height;
    std::vector<std::vector<std::vector<int>>> deadlockPatterns(numTiles);

    for (int tile : interiorTilesList) {
        if (!validBoxTiles[tile]) continue;
        int r = tile / width;
        int c = tile % width;
        for (int dr = -1; dr <= 0; ++dr) {
            for (int dc = -1; dc <= 0; ++dc) {
                std::vector<int> requiredBoxes;
                bool validQuad = true;
                for (int qr = 0; qr <= 1; ++qr) {
                    for (int qc = 0; qc <= 1; ++qc) {
                        int nr = r + dr + qr;
                        int nc = c + dc + qc;
                        if (nr < 0 || nr >= height || nc < 0 || nc >= width) {
                            validQuad = false;
                            break;
                        }
                        int qTile = nr * width + nc;
                        if (!walls[qTile]) {
                            requiredBoxes.push_back(qTile);
                        }
                    }
                    if (!validQuad) break;
                }
                if (!validQuad) continue;
                bool anyNotOnGoal = false;
                for (int req : requiredBoxes) {
                    if (!goalStateMask[req]) {
                        anyNotOnGoal = true;
                        break;
                    }
                }
                if (requiredBoxes.size() >= 2 && anyNotOnGoal) {
                    std::sort(requiredBoxes.begin(), requiredBoxes.end());
                    deadlockPatterns[tile].push_back(std::move(requiredBoxes));
                }
            }
        }
    }
    return deadlockPatterns;
}

bool isPatternDeadlock(
    int boxDestination,
    const std::vector<uint8_t>& boxGrid,
    const std::vector<std::vector<std::vector<int>>>& deadlockPatterns
) {
    for (const auto& requiredBoxes : deadlockPatterns[boxDestination]) {
        bool allPresent = true;
        for (int req : requiredBoxes) {
            if (!boxGrid[req]) {
                allPresent = false;
                break;
            }
        }
        if (allPresent) {
            return true;
        }
    }
    return false;
}

bool isRetrogradeDeadlock(
    const std::vector<uint8_t>& boxGrid,
    const std::vector<int>& frozenBoxes,
    const std::vector<int>& goalState,
    const std::vector<std::vector<int>>& neighbors,
    int width,
    int height
) {
    int numTiles = width * height;
    std::vector<uint8_t> frozenMask(numTiles, 0);
    for (int fb : frozenBoxes) frozenMask[fb] = 1;

    std::vector<uint8_t> remainingGoalsMask(numTiles, 0);
    std::vector<int> remainingGoals;
    bool hasEmptyGoal = false;
    for (int g : goalState) {
        if (!boxGrid[g]) {
            hasEmptyGoal = true;
        }
        if (!frozenMask[g]) {
            remainingGoalsMask[g] = 1;
            remainingGoals.push_back(g);
        }
    }
    if (!hasEmptyGoal) {
        return false;
    }

    bool changed = true;
    while (changed && !remainingGoals.empty()) {
        changed = false;
        for (std::size_t i = 0; i < remainingGoals.size(); ) {
            int goal = remainingGoals[i];
            int gRow = goal / width;
            int gCol = goal % width;
            bool canPull = false;

            for (int stand : neighbors[goal]) {
                if (frozenMask[stand] || remainingGoalsMask[stand]) {
                    continue;
                }
                int sRow = stand / width;
                int sCol = stand % width;
                int dr = sRow - gRow;
                int dc = sCol - gCol;
                int pRow = sRow + dr;
                int pCol = sCol + dc;
                if (pRow < 0 || pRow >= height || pCol < 0 || pCol >= width) continue;
                int pullDest = pRow * width + pCol;

                bool isNeighbor = false;
                for (int nb : neighbors[stand]) {
                    if (nb == pullDest) {
                        isNeighbor = true;
                        break;
                    }
                }
                if (isNeighbor && !frozenMask[pullDest] && !remainingGoalsMask[pullDest]) {
                    canPull = true;
                    break;
                }
            }

            if (canPull) {
                remainingGoalsMask[goal] = 0;
                remainingGoals[i] = remainingGoals.back();
                remainingGoals.pop_back();
                changed = true;
            } else {
                ++i;
            }
        }
    }

    for (int g : remainingGoals) {
        if (!boxGrid[g]) {
            return true;
        }
    }
    return false;
}

bool isRoomDeadlock(
    const std::vector<uint8_t>& boxGrid,
    const std::vector<int>& frozenBoxes,
    const std::vector<std::vector<int>>& rooms,
    const std::vector<int>& tileToRoom,
    const std::vector<uint8_t>& goalStateMask,
    const std::vector<std::vector<int>>& neighbors,
    int numTiles
) {
    if (frozenBoxes.empty()) {
        return false;
    }
    std::vector<uint8_t> frozenMask(numTiles, 0);
    for (int fb : frozenBoxes) frozenMask[fb] = 1;

    for (std::size_t roomId = 0; roomId < rooms.size(); ++roomId) {
        const auto& roomTiles = rooms[roomId];
        int boxesInRoom = 0;
        int goalsInRoom = 0;
        for (int tile : roomTiles) {
            if (boxGrid[tile]) boxesInRoom += 1;
            if (goalStateMask[tile]) goalsInRoom += 1;
        }
        if (boxesInRoom > goalsInRoom) {
            bool hasOpenExit = false;
            for (int tile : roomTiles) {
                for (int dest : neighbors[tile]) {
                    if (tileToRoom[dest] != static_cast<int>(roomId) && !frozenMask[dest]) {
                        hasOpenExit = true;
                        break;
                    }
                }
                if (hasOpenExit) break;
            }
            if (!hasOpenExit) {
                return true;
            }
        }
    }
    return false;
}

bool isDynamicBipartiteDeadlock(
    const std::vector<int>& boxCoordinates,
    const std::vector<int>& frozenBoxes,
    const std::vector<int>& goalState,
    const std::vector<std::vector<int>>& neighbors,
    const std::vector<uint8_t>& validBoxTiles,
    int width,
    int height
) {
    if (frozenBoxes.empty()) {
        return false;
    }
    int numTiles = width * height;
    thread_local std::vector<uint8_t> frozenMask;
    frozenMask.assign(numTiles, 0);
    for (int fb : frozenBoxes) frozenMask[fb] = 1;

    thread_local std::vector<int> boxList;
    boxList.clear();
    for (int b : boxCoordinates) {
        if (!frozenMask[b]) boxList.push_back(b);
    }
    thread_local std::vector<int> goalList;
    goalList.clear();
    for (int g : goalState) {
        if (!frozenMask[g]) goalList.push_back(g);
    }
    if (boxList.empty() || boxList.size() != goalList.size()) {
        return false;
    }

    int n = static_cast<int>(boxList.size());
    thread_local std::vector<std::vector<int>> adj;
    adj.assign(n, {});
    thread_local std::vector<uint8_t> visited;
    thread_local std::vector<int> frontier;

    for (int gIdx = 0; gIdx < n; ++gIdx) {
        int goal = goalList[gIdx];
        visited.assign(numTiles, 0);
        frontier.clear();
        frontier.push_back(goal);
        visited[goal] = 1;

        for (std::size_t fIdx = 0; fIdx < frontier.size(); ++fIdx) {
            int curr = frontier[fIdx];
            int cRow = curr / width;
            int cCol = curr % width;

            for (int dest : neighbors[curr]) {
                if (frozenMask[dest] || !validBoxTiles[dest] || visited[dest]) {
                    continue;
                }
                int dRow = dest / width;
                int dCol = dest % width;
                int dr = dRow - cRow;
                int dc = dCol - cCol;
                int pRow = dRow + dr;
                int pCol = dCol + dc;
                if (pRow < 0 || pRow >= height || pCol < 0 || pCol >= width) continue;
                int playerStand = pRow * width + pCol;

                bool canStand = false;
                for (int nb : neighbors[dest]) {
                    if (nb == playerStand) {
                        canStand = true;
                        break;
                    }
                }
                if (canStand && !frozenMask[playerStand]) {
                    visited[dest] = 1;
                    frontier.push_back(dest);
                }
            }
        }

        for (int bIdx = 0; bIdx < n; ++bIdx) {
            int bTile = boxList[bIdx];
            if (visited[bTile]) {
                adj[bIdx].push_back(gIdx);
            }
        }
    }

    thread_local std::vector<int> matchG;
    matchG.assign(n, -1);
    thread_local std::vector<uint8_t> seenG;

    auto dfsMatch = [&](auto& self, int bIdx) -> bool {
        for (int gIdx : adj[bIdx]) {
            if (seenG[gIdx]) continue;
            seenG[gIdx] = 1;
            if (matchG[gIdx] == -1 || self(self, matchG[gIdx])) {
                matchG[gIdx] = bIdx;
                return true;
            }
        }
        return false;
    };

    for (int bIdx = 0; bIdx < n; ++bIdx) {
        if (adj[bIdx].empty()) {
            return true;
        }
        seenG.assign(n, 0);
        if (!dfsMatch(dfsMatch, bIdx)) {
            return true;
        }
    }
    return false;
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
    int nodeWeight;
    int parentNodeId;
    ParentRecord parent;
};

using FeatureTuple = std::tuple<int, int, int, int, int, int, int>;

struct CandidateMove {
    FeatureTuple featureTuple;
    int pushedBox;
    int boxDestination;
    int newStateId;
};

void selectAdvisedMoves(
    const std::vector<CandidateMove>& candidateMoves,
    const std::vector<StateNode>& states,
    const FullFeatures& parentFeatures,
    const std::vector<uint8_t>& parentReachable,
    const std::vector<int>& currentBoxCoords,
    const std::vector<uint8_t>& currentBoxGrid,
    const std::vector<std::vector<int>>& hotspotList,
    const std::vector<uint8_t>& sinkBasinMask,
    const std::vector<uint8_t>& validBoxTiles,
    int width,
    int height,
    std::vector<uint8_t>& advisedFlagsOut
) {
    advisedFlagsOut.assign(candidateMoves.size(), 0);
    if (candidateMoves.empty()) {
        return;
    }

    int parentOOP = std::get<0>(parentFeatures);
    int parentNegPacked = std::get<1>(parentFeatures);
    int parentConn = std::get<2>(parentFeatures);
    int parentRoomConn = std::get<3>(parentFeatures);
    int parentHotspots = std::get<4>(parentFeatures);

    // 1. Packing Advisor
    int bestIdx = -1;
    for (std::size_t i = 0; i < candidateMoves.size(); ++i) {
        if (std::get<1>(candidateMoves[i].featureTuple) < parentNegPacked) {
            if (bestIdx == -1 || candidateMoves[i].featureTuple < candidateMoves[bestIdx].featureTuple) {
                bestIdx = static_cast<int>(i);
            }
        }
    }
    if (bestIdx != -1) advisedFlagsOut[bestIdx] = 1;

    // 2. Connectivity Advisor
    bestIdx = -1;
    for (std::size_t i = 0; i < candidateMoves.size(); ++i) {
        if (std::get<2>(candidateMoves[i].featureTuple) < parentConn) {
            if (bestIdx == -1 || candidateMoves[i].featureTuple < candidateMoves[bestIdx].featureTuple) {
                bestIdx = static_cast<int>(i);
            }
        }
    }
    if (bestIdx != -1) advisedFlagsOut[bestIdx] = 1;

    // 3. Room-Connectivity Advisor
    bestIdx = -1;
    for (std::size_t i = 0; i < candidateMoves.size(); ++i) {
        if (std::get<3>(candidateMoves[i].featureTuple) < parentRoomConn) {
            if (bestIdx == -1 || candidateMoves[i].featureTuple < candidateMoves[bestIdx].featureTuple) {
                bestIdx = static_cast<int>(i);
            }
        }
    }
    if (bestIdx != -1) advisedFlagsOut[bestIdx] = 1;

    // 4. Hotspots Advisor
    bestIdx = -1;
    for (std::size_t i = 0; i < candidateMoves.size(); ++i) {
        if (std::get<4>(candidateMoves[i].featureTuple) < parentHotspots) {
            if (bestIdx == -1 || candidateMoves[i].featureTuple < candidateMoves[bestIdx].featureTuple) {
                bestIdx = static_cast<int>(i);
            }
        }
    }
    if (bestIdx != -1) advisedFlagsOut[bestIdx] = 1;

    // 5. Explorer Advisor
    struct LockedSide {
        int otherBox;
        int standing;
        int dest;
    };
    thread_local std::vector<LockedSide> lockedSides;
    lockedSides.clear();
    int numBoxes = static_cast<int>(currentBoxCoords.size());
    for (int bIdx = 0; bIdx < numBoxes; ++bIdx) {
        int box = currentBoxCoords[bIdx];
        int bRow = box / width;
        int bCol = box % width;
        for (int d = 0; d < 4; ++d) {
            int dr = DIRECTION[d][0];
            int dc = DIRECTION[d][1];
            int sRow = bRow - dr;
            int sCol = bCol - dc;
            int dRow = bRow + dr;
            int dCol = bCol + dc;
            if (sRow < 0 || sRow >= height || sCol < 0 || sCol >= width) continue;
            if (dRow < 0 || dRow >= height || dCol < 0 || dCol >= width) continue;
            int standing = sRow * width + sCol;
            int dest = dRow * width + dCol;
            if (!validBoxTiles[dest]) continue;
            if (!(parentReachable[standing] && !currentBoxGrid[dest])) {
                lockedSides.push_back({box, standing, dest});
            }
        }
    }

    bestIdx = -1;
    if (!lockedSides.empty()) {
        for (std::size_t i = 0; i < candidateMoves.size(); ++i) {
            const auto& move = candidateMoves[i];
            const auto& childState = states[move.newStateId];
            bool openedNewSide = false;
            for (const auto& ls : lockedSides) {
                if (ls.otherBox == move.pushedBox) continue;
                if (childState.reachable[ls.standing] && !containsBox(childState.boxCoordinates, ls.dest)) {
                    openedNewSide = true;
                    break;
                }
            }
            if (openedNewSide) {
                if (bestIdx == -1 || move.featureTuple < candidateMoves[bestIdx].featureTuple) {
                    bestIdx = static_cast<int>(i);
                }
            }
        }
    }
    if (bestIdx != -1) advisedFlagsOut[bestIdx] = 1;

    // 6. Opener Advisor
    int bestHotspotBox = -1;
    int maxTrapped = 0;
    for (int box : currentBoxCoords) {
        int trappedCount = 0;
        for (int trapped : hotspotList[box]) {
            if (currentBoxGrid[trapped]) {
                trappedCount += 1;
            }
        }
        if (trappedCount > maxTrapped) {
            maxTrapped = trappedCount;
            bestHotspotBox = box;
        }
    }

    if (bestHotspotBox != -1) {
        bestIdx = -1;
        for (std::size_t i = 0; i < candidateMoves.size(); ++i) {
            const auto& move = candidateMoves[i];
            if (move.pushedBox == bestHotspotBox && std::get<2>(move.featureTuple) <= parentConn) {
                if (bestIdx == -1 || move.featureTuple < candidateMoves[bestIdx].featureTuple) {
                    bestIdx = static_cast<int>(i);
                }
            }
        }
        if (bestIdx == -1) {
            int hRow = bestHotspotBox / width;
            int hCol = bestHotspotBox % width;
            for (std::size_t i = 0; i < candidateMoves.size(); ++i) {
                const auto& move = candidateMoves[i];
                if (std::get<2>(move.featureTuple) <= parentConn) {
                    int pRow = move.pushedBox / width;
                    int pCol = move.pushedBox % width;
                    int dRow = move.boxDestination / width;
                    int dCol = move.boxDestination % width;
                    int oldDist = std::abs(pRow - hRow) + std::abs(pCol - hCol);
                    int newDist = std::abs(dRow - hRow) + std::abs(dCol - hCol);
                    if (oldDist <= 3 && newDist > oldDist) {
                        if (bestIdx == -1 || move.featureTuple < candidateMoves[bestIdx].featureTuple) {
                            bestIdx = static_cast<int>(i);
                        }
                    }
                }
            }
        }
        if (bestIdx != -1) advisedFlagsOut[bestIdx] = 1;
    }

    // 7. OOP Advisor
    if (parentOOP > 0) {
        bestIdx = -1;
        for (std::size_t i = 0; i < candidateMoves.size(); ++i) {
            const auto& move = candidateMoves[i];
            if ((!sinkBasinMask[move.pushedBox] && sinkBasinMask[move.boxDestination] && std::get<2>(move.featureTuple) <= parentConn) ||
                std::get<0>(move.featureTuple) < parentOOP) {
                if (bestIdx == -1 || move.featureTuple < candidateMoves[bestIdx].featureTuple) {
                    bestIdx = static_cast<int>(i);
                }
            }
        }
        if (bestIdx != -1) advisedFlagsOut[bestIdx] = 1;
    }
}

void propagateDeadlock(
    int nodeId,
    const std::vector<int>& parentNodeMap,
    std::vector<int>& aliveChildren,
    std::vector<uint8_t>& deadlockedNodes
) {
    int curr = nodeId;
    while (curr != -1 && !deadlockedNodes[curr]) {
        if (aliveChildren[curr] <= 0) {
            deadlockedNodes[curr] = 1;
            int parent = parentNodeMap[curr];
            if (parent != -1) {
                aliveChildren[parent] -= 1;
            }
            curr = parent;
        } else {
            break;
        }
    }
}

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

    std::vector<uint8_t> sinkBasinMask;
    std::vector<int> packingPlan;
    std::vector<std::vector<int>> mustBeFilledBefore;
    generateBasinsAndPlan(
        goalState, goalStateMask, boxCoordinates, validBoxTiles, rooms, neighbors,
        width, height, sinkBasinMask, packingPlan, mustBeFilledBefore
    );

    auto deadlockPatterns = generateDeadlockPatterns(interiorTilesList, validBoxTiles, goalStateMask, walls, width, height);

    std::unordered_map<std::vector<int>, int, VectorHash> heuristicCache;
    std::unordered_map<std::vector<int>, FullFeatures, VectorHash> featureSpaceCache;

    std::vector<uint8_t> startBoxGrid(numTiles, 0);
    for (int b : boxCoordinates) startBoxGrid[b] = 1;
    std::vector<uint8_t> startReachable = reachablePlayerTiles(playerPosition, startBoxGrid, neighbors, numTiles);
    auto tunnelTiles = detectTunnelSquares(interiorTilesList, interiorTilesMask, goalStateMask, walls, width, height);

    auto getFeatureSpace = [&](const std::vector<int>& boxes, const std::vector<uint8_t>& boxGrid) -> FullFeatures {
        auto it = featureSpaceCache.find(boxes);
        if (it != featureSpaceCache.end()) {
            return it->second;
        }
        int packed = 0;
        int oop = 0;
        computePlanPackingAndOOP(boxes, boxGrid, goalStateMask, sinkBasinMask, packingPlan, mustBeFilledBefore, packed, oop);
        int connectivity = countRegions(interiorTilesList, boxGrid, neighbors, numTiles);
        int roomConnectivity = countRoomConnectivity(boxGrid, rooms, roomGraph, tileToRoom, neighbors, numTiles);
        int hotspots = countHotspots(boxes, boxGrid, hotspotDictionary);
        FullFeatures fullFeatures = {oop, -packed, connectivity, roomConnectivity, hotspots};
        featureSpaceCache[boxes] = fullFeatures;
        return fullFeatures;
    };

    std::vector<StateNode> states;
    states.reserve(65536);
    std::vector<int> parentNodeMap;
    parentNodeMap.reserve(65536);
    std::vector<int> aliveChildren;
    aliveChildren.reserve(65536);
    std::vector<uint8_t> deadlockedNodes;
    deadlockedNodes.reserve(65536);

    std::unordered_map<std::vector<int>, std::vector<uint8_t>, VectorHash> exploredReigons;

    FullFeatures startFullFeatures = getFeatureSpace(boxCoordinates, startBoxGrid);
    FeatureSpace startFeatureSpace = {
        std::get<0>(startFullFeatures),
        std::get<1>(startFullFeatures),
        std::get<2>(startFullFeatures),
        std::get<3>(startFullFeatures)
    };
    int heuristic = calculateHeuristic(boxCoordinates, startBoxGrid, tileDistances, heuristicCache, hotspotDictionary);
    int mobility = countMobility(boxCoordinates, startReachable, neighbors);

    states.push_back(StateNode{
        playerPosition,
        boxCoordinates,
        startReachable,
        startFeatureSpace,
        0,
        -1,
        ParentRecord{-1, -1, -1, 0}
    });
    parentNodeMap.push_back(-1);
    aliveChildren.push_back(0);
    deadlockedNodes.push_back(0);

    using HeapItem = std::tuple<int, int, int, int, int, int, int, int, int, int, int>;
    using MinHeap = std::priority_queue<HeapItem, std::vector<HeapItem>, std::greater<HeapItem>>;
    std::map<FeatureSpace, MinHeap> featureSpaceQueues;
    std::vector<FeatureSpace> activeFeatureSpaces;

    int stateCounter = 0;
    featureSpaceQueues[startFeatureSpace].push({
        0,
        heuristic,
        std::get<0>(startFullFeatures),
        std::get<1>(startFullFeatures),
        std::get<2>(startFullFeatures),
        std::get<3>(startFullFeatures),
        std::get<4>(startFullFeatures),
        heuristic,
        -mobility,
        -stateCounter,
        0
    });
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
    std::vector<uint8_t> localMacroVisited(numTiles * numTiles, 0);
    std::vector<int> localMacroVisitedIndices;
    localMacroVisitedIndices.reserve(numTiles * 4);
    std::vector<CandidateMove> candidateMoves;
    std::vector<uint8_t> advisedFlags;

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

            int currentStateId = -1;
            while (!qIt->second.empty()) {
                int candId = std::get<10>(qIt->second.top());
                qIt->second.pop();
                int pNodeId = parentNodeMap[candId];
                if (pNodeId != -1 && deadlockedNodes[pNodeId]) {
                    continue;
                }
                const auto& candBoxCoords = states[candId].boxCoordinates;
                int candPlayerPos = states[candId].playerPosition;
                auto regIt = exploredReigons.find(candBoxCoords);
                if (regIt != exploredReigons.end() && regIt->second[candPlayerPos]) {
                    if (pNodeId != -1) {
                        aliveChildren[pNodeId] -= 1;
                    }
                    continue;
                }
                currentStateId = candId;
                break;
            }

            if (qIt->second.empty()) {
                featureSpaceQueues.erase(qIt);
                activeFeatureSpaces.erase(
                    std::remove(activeFeatureSpaces.begin(), activeFeatureSpaces.end(), featureSpaceKey),
                    activeFeatureSpaces.end()
                );
            }
            if (currentStateId == -1) {
                continue;
            }

            std::vector<int> currentBoxCoords = states[currentStateId].boxCoordinates;
            std::vector<uint8_t> reachable = std::move(states[currentStateId].reachable);
            FeatureSpace currentBestAncestor = states[currentStateId].bestAncestor;
            int parentWeight = states[currentStateId].nodeWeight;

            auto& regionMask = exploredReigons[currentBoxCoords];
            if (regionMask.empty()) {
                regionMask = reachable;
            } else {
                for (int t = 0; t < numTiles; ++t) {
                    if (reachable[t]) regionMask[t] = 1;
                }
            }

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

            FullFeatures parentFeatures = getFeatureSpace(currentBoxCoords, currentBoxGrid);

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

            candidateMoves.clear();

            for (int box : boxesToExplore) {
                for (int idx : localMacroVisitedIndices) {
                    localMacroVisited[idx] = 0;
                }
                localMacroVisitedIndices.clear();

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

                        if (localMacroVisited[boxDestination * numTiles + newPlayerPos]) {
                            continue;
                        }

                        macroBoxGrid[currentBox] = 0;
                        macroBoxGrid[boxDestination] = 1;

                        if (isPatternDeadlock(boxDestination, macroBoxGrid, deadlockPatterns)) {
                            macroBoxGrid[boxDestination] = 0;
                            macroBoxGrid[currentBox] = 1;
                            continue;
                        }

                        std::vector<int> newBoxCoords = macroBoxCoords;
                        auto bIt = std::lower_bound(newBoxCoords.begin(), newBoxCoords.end(), currentBox);
                        newBoxCoords.erase(bIt);
                        newBoxCoords.insert(
                            std::lower_bound(newBoxCoords.begin(), newBoxCoords.end(), boxDestination),
                            boxDestination
                        );

                        //if the boxes positions have been seen before
                        auto regIt = exploredReigons.find(newBoxCoords);
                        if (regIt != exploredReigons.end() && regIt->second[newPlayerPos]) {
                            //if the player is in the same "room/reigon" then its the exact same state, so prune 
                            macroBoxGrid[boxDestination] = 0;
                            macroBoxGrid[currentBox] = 1;
                            continue;
                        }

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
                                if (isRetrogradeDeadlock(macroBoxGrid, frozenBoxes, goalState, neighbors, width, height)) {
                                    macroBoxGrid[boxDestination] = 0;
                                    macroBoxGrid[currentBox] = 1;
                                    continue;
                                }
                                if (isRoomDeadlock(macroBoxGrid, frozenBoxes, rooms, tileToRoom, goalStateMask, neighbors, numTiles)) {
                                    macroBoxGrid[boxDestination] = 0;
                                    macroBoxGrid[currentBox] = 1;
                                    continue;
                                }
                                if (isDynamicBipartiteDeadlock(newBoxCoords, frozenBoxes, goalState, neighbors, validBoxTiles, width, height)) {
                                    macroBoxGrid[boxDestination] = 0;
                                    macroBoxGrid[currentBox] = 1;
                                    continue;
                                }
                            }

                            newHeuristic = calculateHeuristic(newBoxCoords, macroBoxGrid, tileDistances, heuristicCache, hotspotDictionary);
                        }

                        if (newHeuristic >= 999999) {
                            macroBoxGrid[boxDestination] = 0;
                            macroBoxGrid[currentBox] = 1;
                            continue;
                        }

                        std::vector<uint8_t> newReachable = reachablePlayerTiles(newPlayerPos, macroBoxGrid, neighbors, numTiles);
                        int baseMacroOffset = boxDestination * numTiles;
                        for (int t = 0; t < numTiles; ++t) {
                            if (newReachable[t] && !localMacroVisited[baseMacroOffset + t]) {
                                localMacroVisited[baseMacroOffset + t] = 1;
                                localMacroVisitedIndices.push_back(baseMacroOffset + t);
                            }
                        }

                        FullFeatures fullFeatures = getFeatureSpace(newBoxCoords, macroBoxGrid);
                        int newMobility = countMobility(newBoxCoords, newReachable, neighbors);

                        macroBoxGrid[boxDestination] = 0;
                        macroBoxGrid[currentBox] = 1;

                        int newStateId = static_cast<int>(states.size());
                        states.push_back(StateNode{
                            newPlayerPos,
                            newBoxCoords,
                            newReachable,
                            currentBestAncestor,
                            parentWeight,
                            currentStateId,
                            ParentRecord{macroParentStateId, standingTile, d, pushCount}
                        });
                        parentNodeMap.push_back(currentStateId);
                        aliveChildren.push_back(0);
                        deadlockedNodes.push_back(0);
                        aliveChildren[currentStateId] += 1;

                        FeatureTuple featureTuple = {
                            std::get<0>(fullFeatures),
                            std::get<1>(fullFeatures),
                            std::get<2>(fullFeatures),
                            std::get<3>(fullFeatures),
                            std::get<4>(fullFeatures),
                            newHeuristic,
                            -newMobility
                        };

                        candidateMoves.push_back(CandidateMove{
                            featureTuple,
                            box,
                            boxDestination,
                            newStateId
                        });

                        macroFrontier.push_back({newStateId, boxDestination, std::move(newBoxCoords), std::move(newReachable)});
                    }
                }
            }

            if (candidateMoves.empty()) {
                propagateDeadlock(currentStateId, parentNodeMap, aliveChildren, deadlockedNodes);
                continue;
            }

            selectAdvisedMoves(
                candidateMoves,
                states,
                parentFeatures,
                reachable,
                currentBoxCoords,
                currentBoxGrid,
                hotspotDictionary,
                sinkBasinMask,
                validBoxTiles,
                width,
                height,
                advisedFlags
            );

            for (std::size_t i = 0; i < candidateMoves.size(); ++i) {
                const auto& move = candidateMoves[i];
                int moveWeight = advisedFlags[i] ? 0 : 1;
                int childWeight = parentWeight + moveWeight;
                states[move.newStateId].nodeWeight = childWeight;

                FeatureSpace rawFeatureSpace = {
                    std::get<0>(move.featureTuple),
                    std::get<1>(move.featureTuple),
                    std::get<2>(move.featureTuple),
                    std::get<3>(move.featureTuple)
                };
                auto projPair = getProjectedFeatureSpace(rawFeatureSpace, currentBestAncestor);
                FeatureSpace projectedFeatureSpace = projPair.first;
                states[move.newStateId].bestAncestor = projPair.second;
                int newHeuristic = std::get<5>(move.featureTuple);

                //negative counter so ties pop the newest state first, keeps following 1 push line instead of jumping around
                auto& q = featureSpaceQueues[projectedFeatureSpace];
                if (q.empty()) {
                    activeFeatureSpaces.push_back(projectedFeatureSpace);
                }
                q.push({
                    moveWeight,
                    newHeuristic,
                    std::get<0>(move.featureTuple),
                    std::get<1>(move.featureTuple),
                    std::get<2>(move.featureTuple),
                    std::get<3>(move.featureTuple),
                    std::get<4>(move.featureTuple),
                    std::get<5>(move.featureTuple),
                    std::get<6>(move.featureTuple),
                    -stateCounter,
                    move.newStateId
                });
                stateCounter += 1;
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
