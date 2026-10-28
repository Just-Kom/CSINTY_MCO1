#include "sokobot.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <deque>
#include <functional>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <optional>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace std;
namespace {

struct Position {
    int row = 0;
    int col = 0;
    bool operator==(const Position&) const = default;
    bool operator<(const Position& other) const {
        return tie(row, col) < tie(other.row, other.col);
    }
};


Position operator+(Position a, Position b) { return {a.row + b.row, a.col + b.col}; }
Position operator-(Position a, Position b) { return {a.row - b.row, a.col - b.col}; }

size_t hashCombine(size_t a, size_t b) {
    return a ^ (b + static_cast<size_t>(0x9e3779b97f4a7c15ULL) + (a << 6U) + (a >> 2U));
}

struct PositionHash {
    size_t operator()(Position p) const {
        return hashCombine(hash<int>{}(p.row), hash<int>{}(p.col));
    }
};

using TileSet = unordered_set<Position, PositionHash>;
using Boxes = vector<Position>; 
using Neighbors = unordered_map<Position, vector<Position>, PositionHash>;
using Matrix = vector<vector<int>>;
using DistanceMatrices = unordered_map<Position, Matrix, PositionHash>;
using TileDistances = unordered_map<Position, vector<int>, PositionHash>;
using HotspotDictionary = unordered_map<Position, TileSet, PositionHash>;
using FreezeSubset = unordered_map<Position, set<Boxes>, PositionHash>;

struct BoxesHash {
    size_t operator()(const Boxes& boxes) const {
        size_t result = 0;
        for (Position p : boxes) result = hashCombine(result, PositionHash{}(p));
        return result;
    }
};

struct State {
    Position player;
    Boxes boxes;
    bool operator==(const State&) const = default;
};

struct StateHash {
    size_t operator()(const State& s) const {
        return hashCombine(PositionHash{}(s.player), BoxesHash{}(s.boxes));
    }
};

struct Parent {
    State previousKey;
    Position standingTile;
    Position pushDirection;
    int pushCount;
};

constexpr array<Position, 4> DIRECTION = {{{0, -1}, {0, 1}, {1, 0}, {-1, 0}}};
constexpr array<array<Position, 2>, 2> AXES = {{{{{0, -1}, {0, 1}}},
                                                            {{{1, 0}, {-1, 0}}}}};

bool contains(const TileSet& set, Position p) { return set.find(p) != set.end(); }
bool contains(const Boxes& boxes, Position p) {
    return binary_search(boxes.begin(), boxes.end(), p);
}

Boxes canonicalBoxes(Boxes result) {
    sort(result.begin(), result.end());
    result.erase(unique(result.begin(), result.end()), result.end());
    return result;
}

TileSet asTileSet(const Boxes& boxes) { return TileSet(boxes.begin(), boxes.end()); }

bool subsetOf(const Boxes& subset, const Boxes& superset) {
    return includes(superset.begin(), superset.end(), subset.begin(), subset.end());
}
bool subsetOf(const TileSet& subset, const TileSet& superset) {
    for (Position p : subset) if (!contains(superset, p)) return false;
    return true;
}
bool disjoint(const TileSet& a, const TileSet& b) {
    for (Position p : a) if (contains(b, p)) return false;
    return true;
}

Boxes pushedBoxes(const Boxes& boxes, Position source, Position dest) {
    Boxes result = boxes;
    auto it = lower_bound(result.begin(), result.end(), source);
    if (it == result.end() || *it != source) throw logic_error("Missing pushed box");
    result.erase(it);
    result.insert(lower_bound(result.begin(), result.end(), dest), dest);
    return result;
}

char directionChar(Position p) {
    if (p == Position{0, -1}) return 'l';
    if (p == Position{0, 1}) return 'r';
    if (p == Position{1, 0}) return 'd';
    if (p == Position{-1, 0}) return 'u';
    throw logic_error("Invalid movement direction");
}

bool hasFreezeSubset(Position boxDestination, const Boxes& boxCoordinates,
                     const FreezeSubset& freezeSubset) {
    auto it = freezeSubset.find(boxDestination);
    if (it == freezeSubset.end()) return false;
    for (const Boxes& subset : it->second) {
        if (subsetOf(subset, boxCoordinates)) return true;
    }
    return false;
}

optional<TileSet> detectBlockedAxes(
    Position boxCoordinate, const TileSet& validBoxTiles, const TileSet& walls,
    unordered_map<Position, bool, PositionHash>& explored,
    const Boxes& boxCoordinates, TileSet& exploring) {

    function<bool(Position)> isFrozen = [&](Position currentBox) -> bool {
        if (auto it = explored.find(currentBox); it != explored.end()) return it->second;
        if (contains(exploring, currentBox)) return true;

        array<bool, 2> blockedAxes = {false, false};
        array<vector<Position>, 2> adjBoxesPerAxis;

        for (size_t axisIndex = 0; axisIndex < AXES.size(); ++axisIndex) {
            int numOfSimpleDeadlocks = 0;
            for (Position direction : AXES[axisIndex]) {
                Position dest = currentBox + direction;
                if (contains(walls, dest)) {
                    blockedAxes[axisIndex] = true;
                    break;
                }
                if (!contains(validBoxTiles, dest)) ++numOfSimpleDeadlocks;
                if (contains(boxCoordinates, dest)) {
                    if (contains(exploring, dest)) {
                        blockedAxes[axisIndex] = true;
                        break;
                    }
                    adjBoxesPerAxis[axisIndex].push_back(dest);
                }
            }
            if (!blockedAxes[axisIndex] && numOfSimpleDeadlocks == 2)
                blockedAxes[axisIndex] = true;
            if (!blockedAxes[axisIndex] && adjBoxesPerAxis[axisIndex].empty()) {
                explored[currentBox] = false;
                return false;
            }
        }

        auto savedExplored = explored;
        exploring.insert(currentBox);
        for (size_t axisIndex = 0; axisIndex < adjBoxesPerAxis.size(); ++axisIndex) {
            if (!blockedAxes[axisIndex]) {
                for (Position adjBox : adjBoxesPerAxis[axisIndex]) {
                    if (isFrozen(adjBox)) {
                        blockedAxes[axisIndex] = true;
                        break;
                    }
                }
            }
            if (!blockedAxes[axisIndex]) {
                exploring.erase(currentBox);
                explored = move(savedExplored);
                explored[currentBox] = false;
                return false;
            }
        }
        exploring.erase(currentBox);
        explored[currentBox] = true;
        return true;
    };

    if (!isFrozen(boxCoordinate)) return nullopt;
    TileSet frozen;
    for (const auto& [box, value] : explored) if (value) frozen.insert(box);
    return frozen;
}

DistanceMatrices generateDistanceMatrices(
    const Boxes& goalState, int height, int width, const TileSet& validBoxTiles,
    const Neighbors& neighbors) {
    DistanceMatrices matrices;
    for (Position goalVal : goalState) {
        Matrix matrix(height, vector<int>(width, -1));
        matrix[goalVal.row][goalVal.col] = 0;
        deque<pair<Position, int>> frontier = {{goalVal, 0}};
        TileSet explored = {goalVal};
        while (!frontier.empty()) {
            auto [currentPos, currentCtr] = frontier.front();
            frontier.pop_front();
            for (Position direction : DIRECTION) {
                Position dest = currentPos + direction;
                Position playerStand = dest + direction;
                if (!contains(explored, dest) && contains(validBoxTiles, dest) &&
                    find(neighbors.at(dest).begin(), neighbors.at(dest).end(), playerStand) != neighbors.at(dest).end()) {
                    explored.insert(dest);
                    int newCtr = currentCtr + 1;
                    frontier.emplace_back(dest, newCtr);
                    matrix[dest.row][dest.col] = newCtr;
                }
            }
        }
        matrices.emplace(goalVal, move(matrix));
    }
    return matrices;
}

TileDistances buildDistanceStack(const DistanceMatrices& distanceMatrices,
                                 const Boxes& goalState, const TileSet& validBoxTiles) {
    TileDistances tileDistances;
    for (Position tile : validBoxTiles) {
        vector<int> distances;
        distances.reserve(goalState.size());
        for (Position goalVal : goalState) {
            int val = distanceMatrices.at(goalVal)[tile.row][tile.col];
            distances.push_back(val == -1 ? 999999 : val);
        }
        tileDistances.emplace(tile, move(distances));
    }
    return tileDistances;
}


long long hungarianAssignmentCost(const vector<vector<int>>& costs) {
    const int n = static_cast<int>(costs.size());
    if (n == 0) return 0;
    const int m = static_cast<int>(costs[0].size());
    
    if (n > m) {
        vector<vector<int>> transposed(
            static_cast<size_t>(m), vector<int>(static_cast<size_t>(n)));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < m; ++j)
                transposed[j][i] = costs[i][j];
        return hungarianAssignmentCost(transposed);
    }
    constexpr long long INF = numeric_limits<long long>::max() / 4;
    vector<long long> u(n + 1), v(m + 1);
    vector<int> p(m + 1), way(m + 1);
    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        vector<long long> minv(m + 1, INF);
        vector<bool> used(m + 1);
        do {
            used[j0] = true;
            int i0 = p[j0];
            long long delta = INF;
            int j1 = 0;
            for (int j = 1; j <= m; ++j) {
                if (used[j]) continue;
                long long cur = static_cast<long long>(costs[i0 - 1][j - 1]) - u[i0] - v[j];
                if (cur < minv[j]) {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (minv[j] < delta) {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for (int j = 0; j <= m; ++j) {
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
        } while (j0 != 0);
    }
    long long result = 0;
    for (int j = 1; j <= m; ++j) {
        if (p[j] != 0) result += costs[p[j] - 1][j - 1];
    }
    return result;
}

int countHotspots(const Boxes& boxCoordinates, const HotspotDictionary& hotspotDictionary) {
    int hotspots = 0;
    for (Position box : boxCoordinates) {
        const TileSet& trapped = hotspotDictionary.at(box);
        for (Position other : boxCoordinates) {
            if (contains(trapped, other)) {
                ++hotspots;
                break;
            }
        }
    }
    return hotspots;
}

using HeuristicCache = unordered_map<Boxes, long long, BoxesHash>;
long long calculateHeuristic(const Boxes& boxCoordinates, const TileDistances& tileDistances,
                             HeuristicCache& heuristicCache, const HotspotDictionary& hotspotDictionary) {
    vector<vector<int>> costMatrix;
    costMatrix.reserve(boxCoordinates.size());
    for (Position box : boxCoordinates) costMatrix.push_back(tileDistances.at(box));
    long long heuristic = hungarianAssignmentCost(costMatrix) +
                          static_cast<long long>(countHotspots(boxCoordinates, hotspotDictionary)) * 10;
    heuristicCache[boxCoordinates] = heuristic;
    return heuristic;
}

TileSet reachablePlayerTiles(Position playerPosition, const Boxes& boxCoordinates,
                             const Neighbors& neighbors) {
    vector<Position> frontier = {playerPosition};
    TileSet reachable = {playerPosition};
    for (size_t i = 0; i < frontier.size(); ++i) {
        Position current = frontier[i];
        for (Position dest : neighbors.at(current)) {
            if (!contains(reachable, dest) && !contains(boxCoordinates, dest)) {
                reachable.insert(dest);
                frontier.push_back(dest);
            }
        }
    }
    return reachable;
}

pair<TileSet, TileSet> getCorralFromTile(Position startPos, const Boxes& boxCoordinates,
                                              const Neighbors& neighbors) {
    vector<Position> frontier = {startPos};
    TileSet corral = {startPos};
    TileSet corralBoxes;
    for (size_t i = 0; i < frontier.size(); ++i) {
        Position current = frontier[i];
        for (Position dest : neighbors.at(current)) {
            if (contains(corral, dest)) continue;
            if (contains(boxCoordinates, dest)) {
                corralBoxes.insert(dest);
            } else {
                corral.insert(dest);
                frontier.push_back(dest);
            }
        }
    }
    return {move(corral), move(corralBoxes)};
}

bool isPICorral(const TileSet& corral, const TileSet& corralBoxes,
                const TileSet& reachable, const Boxes& boxCoordinates,
                const TileSet& walls, const TileSet& validBoxTiles) {
    bool hasValidPush = false;
    for (Position box : corralBoxes) {
        for (Position direction : DIRECTION) {
            Position playerPos = box - direction;
            Position boxDest = box + direction;
            if (contains(walls, playerPos) || !contains(validBoxTiles, boxDest)) continue;
            if (contains(corral, playerPos)) continue;
            if (contains(corralBoxes, playerPos) && contains(corralBoxes, boxDest)) continue;
            if (!contains(corral, boxDest)) return false;
            if (!contains(reachable, playerPos) && !contains(corralBoxes, playerPos)) return false;
            if (contains(reachable, playerPos) && !contains(boxCoordinates, boxDest)) hasValidPush = true;
        }
    }
    return hasValidPush;
}

optional<string> findWalkingPath(Position start, Position destination,
                                           const Boxes& boxCoordinates, const Neighbors& neighbors) {
    if (start == destination) return string{};
    deque<Position> frontier = {start};
    unordered_map<Position, optional<pair<Position, char>>, PositionHash> parents;
    parents.emplace(start, nullopt);
    while (!frontier.empty()) {
        Position current = frontier.front();
        frontier.pop_front();
        for (Position dest : neighbors.at(current)) {
            if (parents.find(dest) != parents.end() || contains(boxCoordinates, dest)) continue;
            Position direction = dest - current;
            parents.emplace(dest, pair{current, directionChar(direction)});
            if (dest == destination) {
                string path;
                while (!(dest == start)) {
                    const auto& [previous, move] = *parents.at(dest);
                    path.push_back(move);
                    dest = previous;
                }
                reverse(path.begin(), path.end());
                return path;
            }
            frontier.push_back(dest);
        }
    }
    return nullopt;
}

struct TunnelKey {
    Position tile;
    Position direction;
    bool operator==(const TunnelKey&) const = default;
};
struct TunnelKeyHash {
    size_t operator()(const TunnelKey& k) const {
        return hashCombine(PositionHash{}(k.tile), PositionHash{}(k.direction));
    }
};
using TunnelSquares = unordered_set<TunnelKey, TunnelKeyHash>;

TunnelSquares detectTunnelSquares(const TileSet& interiorTiles, const Boxes& goalState,
                                   const TileSet& walls) {
    TunnelSquares tunnelSquares;
    for (Position tile : interiorTiles) {
        for (Position direction : DIRECTION) {
            Position prevTile = tile - direction;
            Position side = {-direction.col, direction.row};
            Position tileSide1 = tile + side, tileSide2 = tile - side;
            Position prevSide1 = prevTile + side, prevSide2 = prevTile - side;
            if (contains(interiorTiles, prevTile) && !contains(goalState, tile) &&
                contains(walls, tileSide1) && contains(walls, tileSide2) &&
                contains(walls, prevSide1) && contains(walls, prevSide2)) {
                tunnelSquares.insert(TunnelKey{tile, direction});
            }
        }
    }
    return tunnelSquares;
}

struct PullState {
    Position player;
    Position box;
    bool operator==(const PullState&) const = default;
};
struct PullStateHash {
    size_t operator()(const PullState& v) const {
        return hashCombine(PositionHash{}(v.player), PositionHash{}(v.box));
    }
};

pair<TileSet, unordered_set<PullState, PullStateHash>>
getReachableGoalsCount(Position goal, optional<Position> hotspotTile, const Neighbors& neighbors) {
    vector<PullState> frontier;
    unordered_set<PullState, PullStateHash> explored;
    TileSet reached;
    for (Position dest : neighbors.at(goal)) {
        if (!hotspotTile || dest != *hotspotTile) {
            PullState startState = {dest, goal};
            frontier.push_back(startState);
            explored.insert(startState);
        }
    }
    for (size_t i = 0; i < frontier.size(); ++i) {
        auto [currentPlayerPos, currentBoxCoords] = frontier[i];
        reached.insert(currentBoxCoords);
        Position delta = currentPlayerPos - currentBoxCoords;
        if (abs(delta.row) + abs(delta.col) == 1) {
            Position dest = currentPlayerPos + delta;
            if (find(neighbors.at(currentPlayerPos).begin(), neighbors.at(currentPlayerPos).end(), dest) != neighbors.at(currentPlayerPos).end() &&
                (!hotspotTile || dest != *hotspotTile)) {
                PullState newState = {dest, currentPlayerPos};
                if (explored.insert(newState).second) frontier.push_back(newState);
            }
        }
        for (Position destination : neighbors.at(currentPlayerPos)) {
            if (destination != currentBoxCoords && (!hotspotTile || destination != *hotspotTile)) {
                PullState startState = {destination, currentBoxCoords};
                if (explored.insert(startState).second) frontier.push_back(startState);
            }
        }
    }
    return {move(reached), move(explored)};
}

pair<TileSet, HotspotDictionary> generateHotspotTable(
    const Boxes& goalState, const TileSet& interiorTiles, const Neighbors& neighbors) {
    unordered_map<Position, int, PositionHash> reachableGoalCount;
    for (Position tile : interiorTiles) reachableGoalCount[tile] = 0;
    unordered_map<Position, TileSet, PositionHash> goalReached;
    unordered_map<Position, TileSet, PositionHash> visitedByGoal;

    for (Position goal : goalState) {
        auto [reached, explored] = getReachableGoalsCount(goal, nullopt, neighbors);
        goalReached[goal] = reached;
        TileSet visited = reached;
        for (const PullState& state : explored) visited.insert(state.player);
        visitedByGoal[goal] = move(visited);
        for (Position tile : reached) ++reachableGoalCount.at(tile);
    }

    TileSet validBoxTiles;
    for (const auto& [tile, count] : reachableGoalCount)
        if (count > 0) validBoxTiles.insert(tile);

    HotspotDictionary hotspotDictionary;
    for (Position tile : interiorTiles) hotspotDictionary.emplace(tile, TileSet{});
    for (Position hotspotTile : validBoxTiles) {
        unordered_map<Position, int, PositionHash> hotspotGoalCount;
        for (Position tile : validBoxTiles) hotspotGoalCount[tile] = 0;
        for (Position goal : goalState) {
            if (goal == hotspotTile) continue;
            TileSet reached;
            if (contains(visitedByGoal.at(goal), hotspotTile)) {
                reached = getReachableGoalsCount(goal, hotspotTile, neighbors).first;
            } else {
                reached = goalReached.at(goal);
            }
            for (Position tile : reached) {
                auto it = hotspotGoalCount.find(tile);
                if (it != hotspotGoalCount.end()) ++it->second;
            }
        }
        for (Position tile : validBoxTiles) {
            if (tile == hotspotTile) continue;
            int expectedCount = reachableGoalCount.at(tile);
            if (goalReached.find(hotspotTile) != goalReached.end() &&
                contains(goalReached.at(hotspotTile), tile)) --expectedCount;
            if (hotspotGoalCount.at(tile) < expectedCount) hotspotDictionary[hotspotTile].insert(tile);
        }
    }
    return {move(validBoxTiles), move(hotspotDictionary)};
}

int getPackedBoxes(const Boxes& boxCoordinates, const Boxes& goalState) {
    int packed = 0;
    for (Position box : boxCoordinates) if (contains(goalState, box)) ++packed;
    return packed;
}

int countRegions(const TileSet& interiorTiles, const Boxes& boxes, const Neighbors& neighbors) {
    TileSet explored;
    int rooms = 0;
    for (Position currentPos : interiorTiles) {
        if (contains(boxes, currentPos) || contains(explored, currentPos)) continue;
        ++rooms;
        vector<Position> frontier = {currentPos};
        explored.insert(currentPos);
        for (size_t i = 0; i < frontier.size(); ++i) {
            Position current = frontier[i];
            for (Position dest : neighbors.at(current)) {
                if (!contains(boxes, dest) && !contains(explored, dest)) {
                    explored.insert(dest);
                    frontier.push_back(dest);
                }
            }
        }
    }
    return rooms;
}

struct RoomInfo {
    vector<TileSet> rooms;
    vector<set<int>> graphIds;
    unordered_map<Position, int, PositionHash> tileToRoom;
};

RoomInfo generateRoomsAndGraph(const TileSet& interiorTiles, const Neighbors& neighbors) {
    TileSet roomTiles;
    for (Position tile : interiorTiles) {
        auto [row, col] = pair{tile.row, tile.col};
        array<Position, 6> twoByThree = {{{row, col}, {row, col + 1}, {row, col + 2},
                                                {row + 1, col}, {row + 1, col + 1}, {row + 1, col + 2}}};
        array<Position, 6> threeByTwo = {{{row, col}, {row + 1, col}, {row + 2, col},
                                                {row, col + 1}, {row + 1, col + 1}, {row + 2, col + 1}}};
        bool first = true, second = true;
        for (Position p : twoByThree) if (!contains(interiorTiles, p)) first = false;
        for (Position p : threeByTwo) if (!contains(interiorTiles, p)) second = false;
        if (first) roomTiles.insert(twoByThree.begin(), twoByThree.end());
        if (second) roomTiles.insert(threeByTwo.begin(), threeByTwo.end());
    }

    RoomInfo result;
    while (!roomTiles.empty()) {
        Position startPos = *roomTiles.begin();
        roomTiles.erase(startPos);
        int roomId = static_cast<int>(result.rooms.size());
        TileSet currentRoom = {startPos};
        vector<Position> frontier = {startPos};
        result.tileToRoom[startPos] = roomId;
        for (size_t i = 0; i < frontier.size(); ++i) {
            Position currentPos = frontier[i];
            for (Position dest : neighbors.at(currentPos)) {
                if (roomTiles.erase(dest) > 0) {
                    currentRoom.insert(dest);
                    result.tileToRoom[dest] = roomId;
                    frontier.push_back(dest);
                }
            }
        }
        result.rooms.push_back(move(currentRoom));
    }

    result.graphIds.resize(result.rooms.size());
    for (size_t roomId = 0; roomId < result.rooms.size(); ++roomId) {
        vector<Position> frontier(result.rooms[roomId].begin(), result.rooms[roomId].end());
        TileSet explored = result.rooms[roomId];
        for (size_t i = 0; i < frontier.size(); ++i) {
            Position currentPos = frontier[i];
            for (Position dest : neighbors.at(currentPos)) {
                if (!explored.insert(dest).second) continue;
                auto it = result.tileToRoom.find(dest);
                if (it != result.tileToRoom.end()) {
                    int neighborRoom = it->second;
                    result.graphIds[roomId].insert(neighborRoom);
                    result.graphIds[neighborRoom].insert(static_cast<int>(roomId));
                } else {
                    frontier.push_back(dest);
                }
            }
        }
    }
    return result;
}

int countRoomConnectivity(const Boxes& boxes, const RoomInfo& roomInfo,
                          const Neighbors& neighbors) {
    int brokenEdges = 0;
    for (size_t roomId = 0; roomId < roomInfo.rooms.size(); ++roomId) {
        const auto& roomTiles = roomInfo.rooms[roomId];
        const auto& adjRoomIds = roomInfo.graphIds[roomId];
        if (adjRoomIds.empty()) continue;
        TileSet explored;
        set<int> connectedRooms;
        for (Position startPos : roomTiles) {
            if (contains(boxes, startPos) || contains(explored, startPos)) continue;
            vector<Position> frontier = {startPos};
            explored.insert(startPos);
            for (size_t i = 0; i < frontier.size(); ++i) {
                Position currentPos = frontier[i];
                for (Position dest : neighbors.at(currentPos)) {
                    if (contains(boxes, dest) || contains(explored, dest)) continue;
                    explored.insert(dest);
                    if (auto it = roomInfo.tileToRoom.find(dest); it != roomInfo.tileToRoom.end()) {
                        int neighborRoom = it->second;
                        if (neighborRoom == static_cast<int>(roomId)) frontier.push_back(dest);
                        else connectedRooms.insert(neighborRoom);
                    } else {
                        frontier.push_back(dest);
                    }
                }
            }
            if (connectedRooms.size() == adjRoomIds.size()) break;
        }
        for (int neighborRoom : adjRoomIds) {
            if (static_cast<int>(roomId) < neighborRoom &&
                connectedRooms.find(neighborRoom) == connectedRooms.end()) ++brokenEdges;
        }
    }
    return brokenEdges;
}

int countMobility(const Boxes& boxCoordinates, const TileSet& reachable,
                  const Neighbors& neighbors) {
    int mobility = 0;
    for (Position box : boxCoordinates) {
        for (Position dest : neighbors.at(box)) {
            if (contains(reachable, dest)) ++mobility;
        }
    }
    return mobility;
}

using FeatureSpace = array<int, 3>;
pair<FeatureSpace, FeatureSpace> getProjectedFeatureSpace(
    const FeatureSpace& rawFeatureSpace, const FeatureSpace& bestAncestorFeatureSpace) {
    if (rawFeatureSpace < bestAncestorFeatureSpace) return {rawFeatureSpace, rawFeatureSpace};
    return {bestAncestorFeatureSpace, bestAncestorFeatureSpace};
}

struct HeapEntry {
    long long heuristic;
    long long negativeMobility;
    long long negativeCounter;
    State key;
};
struct HeapCompare {
    bool operator()(const HeapEntry& a, const HeapEntry& b) const {
        return tie(a.heuristic, a.negativeMobility, a.negativeCounter) >
               tie(b.heuristic, b.negativeMobility, b.negativeCounter);
    }
};
using FeatureHeap = priority_queue<HeapEntry, vector<HeapEntry>, HeapCompare>;

struct MacroNode {
    State parent;
    Position currentBox;
    Boxes boxes;
    TileSet reachable;
};

}

string SokoBot::solveSokobanPuzzle(int width, int height, const vector<string>& mapData, const vector<string>& itemsData) {
    Boxes goalState;
    for (int row = 0; row < static_cast<int>(mapData.size()); ++row) {
        for (int col = 0; col < static_cast<int>(mapData[row].size()); ++col) {
            if (mapData[row][col] == '.') goalState.push_back({row, col});
        }
    }
    goalState = canonicalBoxes(move(goalState));
    const TileSet goalTiles = asTileSet(goalState);

    Boxes boxCoordinates;
    Position playerPosition{};
    for (int row = 0; row < static_cast<int>(itemsData.size()); ++row) {
        for (int col = 0; col < static_cast<int>(itemsData[row].size()); ++col) {
            if (itemsData[row][col] == '$') boxCoordinates.push_back({row, col});
            else if (itemsData[row][col] == '@') playerPosition = {row, col};
        }
    }
    boxCoordinates = canonicalBoxes(move(boxCoordinates));

    TileSet walls;
    for (int row = 0; row < height; ++row)
        for (int col = 0; col < width; ++col)
            if (mapData.at(row).at(col) == '#') walls.insert({row, col});

    Neighbors neighbors;
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            Position tile{row, col};
            if (contains(walls, tile)) continue;
            vector<Position> tileNeighbors;
            for (Position d : DIRECTION) {
                Position dest = tile + d;
                if (dest.row >= 0 && dest.row < height && dest.col >= 0 && dest.col < width &&
                    !contains(walls, dest)) tileNeighbors.push_back(dest);
            }
            neighbors.emplace(tile, move(tileNeighbors));
        }
    }

    TileSet interiorTiles = reachablePlayerTiles(playerPosition, Boxes{}, neighbors);
    auto [validBoxTiles, hotspotDictionary] = generateHotspotTable(goalState, interiorTiles, neighbors);
    DistanceMatrices distances = generateDistanceMatrices(goalState, height, width, validBoxTiles, neighbors);
    TileDistances tileDistances = buildDistanceStack(distances, goalState, validBoxTiles);
    FreezeSubset freezeSubset;
    RoomInfo roomInfo = generateRoomsAndGraph(interiorTiles, neighbors);
    HeuristicCache heuristicCache;
    unordered_map<Boxes, FeatureSpace, BoxesHash> featureSpaceCache;
    TileSet startReachable = reachablePlayerTiles(playerPosition, boxCoordinates, neighbors);
    TunnelSquares tunnelTiles = detectTunnelSquares(interiorTiles, goalState, walls);

    auto getFeatureSpace = [&](const Boxes& boxes) -> FeatureSpace {
        if (auto it = featureSpaceCache.find(boxes); it != featureSpaceCache.end()) return it->second;
        int packed = getPackedBoxes(boxes, goalState);
        int connectivity = countRegions(interiorTiles, boxes, neighbors);
        int roomConnectivity = countRoomConnectivity(boxes, roomInfo, neighbors);
        FeatureSpace rawFeatureSpace = {-packed, connectivity, roomConnectivity};
        featureSpaceCache[boxes] = rawFeatureSpace;
        return rawFeatureSpace;
    };

    State startKey{playerPosition, boxCoordinates};
    unordered_map<State, TileSet, StateHash> stateByKey;
    stateByKey.emplace(startKey, startReachable);
    unordered_map<State, optional<Parent>, StateHash> parents;
    parents.emplace(startKey, nullopt);
    unordered_map<Boxes, vector<TileSet>, BoxesHash> exploredReigons;
    exploredReigons[boxCoordinates].push_back(startReachable);

    map<FeatureSpace, FeatureHeap> featureSpaceQueues;
    list<FeatureSpace> featureOrder;
    long long stateCounter = 0;
    FeatureSpace startFeatureSpace = getFeatureSpace(boxCoordinates);
    unordered_map<State, FeatureSpace, StateHash> bestAncestorFeatureSpace;
    bestAncestorFeatureSpace[startKey] = startFeatureSpace;
    long long heuristic = calculateHeuristic(boxCoordinates, tileDistances, heuristicCache, hotspotDictionary);
    int mobility = countMobility(boxCoordinates, startReachable, neighbors);
    featureSpaceQueues[startFeatureSpace].push({heuristic, -mobility, -stateCounter, startKey});
    featureOrder.push_back(startFeatureSpace);
    ++stateCounter;

    while (!featureSpaceQueues.empty()) {
        vector<FeatureSpace> snapshot(featureOrder.begin(), featureOrder.end());
        for (FeatureSpace featureSpaceKey : snapshot) {
            auto heapIt = featureSpaceQueues.find(featureSpaceKey);
            if (heapIt == featureSpaceQueues.end()) continue;
            if (heapIt->second.empty()) {
                featureSpaceQueues.erase(heapIt);
                featureOrder.remove(featureSpaceKey);
                continue;
            }
            HeapEntry entry = heapIt->second.top();
            heapIt->second.pop();
            if (heapIt->second.empty()) {
                featureSpaceQueues.erase(heapIt);
                featureOrder.remove(featureSpaceKey);
            }

            State currentKey = move(entry.key);
            const Boxes& currentBoxCoords = currentKey.boxes;
            auto stateIt = stateByKey.find(currentKey);
            if (stateIt == stateByKey.end()) throw logic_error("Missing state reachable region");
            TileSet reachable = move(stateIt->second);
            stateByKey.erase(stateIt);
            FeatureSpace currentBestAncestor = bestAncestorFeatureSpace.at(currentKey);

            if (currentBoxCoords == goalState) {
                vector<Parent> transitions;
                State key = currentKey;
                while (parents.at(key).has_value()) {
                    Parent transition = *parents.at(key);
                    transitions.push_back(transition);
                    key = transition.previousKey;
                }
                reverse(transitions.begin(), transitions.end());
                string winPath;
                for (const Parent& transition : transitions) {
                    optional<string> walk = findWalkingPath(
                        transition.previousKey.player, transition.standingTile,
                        transition.previousKey.boxes, neighbors);
                    if (!walk) throw runtime_error("Invalid reconstructed push path");
                    winPath += *walk;
                    winPath.append(static_cast<size_t>(transition.pushCount),
                                   directionChar(transition.pushDirection));
                }
                cout << "Total states explored: " << parents.size() << '\n';
                cout << "Total unique box configurations explored: " << exploredReigons.size() << '\n';
                return winPath;
            }

            TileSet boxesToExplore = asTileSet(currentBoxCoords);
            TileSet exploredCorral;
            TileSet corralSeeds;
            for (Position box : currentBoxCoords) {
                for (Position direction : DIRECTION) {
                    Position standingTile = box - direction;
                    Position boxDestination = box + direction;
                    if (contains(reachable, standingTile) && contains(validBoxTiles, boxDestination)) {
                        if (!contains(reachable, boxDestination) && !contains(currentBoxCoords, boxDestination))
                            corralSeeds.insert(boxDestination);
                    }
                }
            }
            for (Position currentPos : corralSeeds) {
                if (contains(exploredCorral, currentPos)) continue;
                auto [corral, corralBoxes] = getCorralFromTile(currentPos, currentBoxCoords, neighbors);
                exploredCorral.insert(corral.begin(), corral.end());
                if (corralBoxes.empty() || corralBoxes.size() == currentBoxCoords.size()) continue;
                if (subsetOf(corralBoxes, goalTiles) && disjoint(corral, goalTiles)) continue;
                if (isPICorral(corral, corralBoxes, reachable, currentBoxCoords, walls, validBoxTiles)) {
                    boxesToExplore = move(corralBoxes);
                    break;
                }
            }

            for (Position box : boxesToExplore) {
                vector<MacroNode> macroFrontier = {{currentKey, box, currentBoxCoords, reachable}};
                for (size_t macroIndex = 0; macroIndex < macroFrontier.size(); ++macroIndex) {
                    MacroNode macroNode = move(macroFrontier[macroIndex]);
                    const State& macroParentKey = macroNode.parent;
                    Position currentBox = macroNode.currentBox;
                    const Boxes& macroBoxCoords = macroNode.boxes;
                    const TileSet& macroReachable = macroNode.reachable;
                    for (Position direction : DIRECTION) {
                        Position standingTile = currentBox - direction;
                        if (!contains(macroReachable, standingTile)) continue;
                        Position boxDestination = currentBox + direction;
                        if (!contains(validBoxTiles, boxDestination) || contains(macroBoxCoords, boxDestination))
                            continue;
                        Position newPlayerPos = currentBox;
                        Position pushDirection = direction;
                        int pushCount = 1;
                        Position movingBox = boxDestination;
                        while (tunnelTiles.find(TunnelKey{movingBox, pushDirection}) != tunnelTiles.end()) {
                            Position nextBox = movingBox + direction;
                            if (!contains(validBoxTiles, nextBox) || contains(macroBoxCoords, nextBox)) break;
                            newPlayerPos = movingBox;
                            movingBox = nextBox;
                            ++pushCount;
                        }
                        boxDestination = movingBox;
                        Boxes newBoxCoords = pushedBoxes(macroBoxCoords, currentBox, boxDestination);
                        bool alreadyExplored = false;
                        if (auto it = exploredReigons.find(newBoxCoords); it != exploredReigons.end()) {
                            for (const TileSet& exploredRegion : it->second) {
                                if (contains(exploredRegion, newPlayerPos)) {
                                    alreadyExplored = true;
                                    break;
                                }
                            }
                        }
                        if (alreadyExplored) continue;

                        long long newHeuristic;
                        if (auto it = heuristicCache.find(newBoxCoords); it != heuristicCache.end()) {
                            newHeuristic = it->second;
                        } else {
                            if (hasFreezeSubset(boxDestination, newBoxCoords, freezeSubset)) continue;
                            unordered_map<Position, bool, PositionHash> freezeExplored;
                            TileSet exploring;
                            optional<TileSet> frozenBoxes = detectBlockedAxes(
                                boxDestination, validBoxTiles, walls, freezeExplored, newBoxCoords, exploring);
                            bool nonGoalFrozen = false;
                            if (frozenBoxes) {
                                for (Position b : *frozenBoxes) {
                                    if (!contains(goalState, b)) {
                                        nonGoalFrozen = true;
                                        break;
                                    }
                                }
                            }
                            if (nonGoalFrozen) {
                                if (frozenBoxes->size() <= 4) {
                                    Boxes frozenPattern = canonicalBoxes(Boxes(frozenBoxes->begin(), frozenBoxes->end()));
                                    for (Position frozenBox : frozenPattern)
                                        freezeSubset[frozenBox].insert(frozenPattern);
                                }
                                continue;
                            }
                            newHeuristic = calculateHeuristic(newBoxCoords, tileDistances,
                                                              heuristicCache, hotspotDictionary);
                        }
                        if (newHeuristic >= 999999) continue;

                        TileSet newReachable = reachablePlayerTiles(newPlayerPos, newBoxCoords, neighbors);
                        exploredReigons[newBoxCoords].push_back(newReachable);
                        State newKey{newPlayerPos, newBoxCoords};
                        parents[newKey] = Parent{macroParentKey, standingTile, direction, pushCount};
                        stateByKey[newKey] = newReachable;

                        FeatureSpace rawFeatureSpace = getFeatureSpace(newBoxCoords);
                        auto [projectedFeatureSpace, newBestAncestor] =
                            getProjectedFeatureSpace(rawFeatureSpace, currentBestAncestor);
                        bestAncestorFeatureSpace[newKey] = newBestAncestor;
                        int newMobility = countMobility(newBoxCoords, newReachable, neighbors);

                        if (featureSpaceQueues.find(projectedFeatureSpace) == featureSpaceQueues.end())
                            featureOrder.push_back(projectedFeatureSpace);
                        featureSpaceQueues[projectedFeatureSpace].push(
                            {newHeuristic, -newMobility, -stateCounter, newKey});
                        ++stateCounter;
                        macroFrontier.push_back({newKey, boxDestination, newBoxCoords, move(newReachable)});
                    }
                }
            }
        }
    }
    return "";
}
int main() {
    int width, height;
    cin >> width >> height;
    cin.ignore(numeric_limits<streamsize>::max(), '\n');

    vector<string> mapData(height);
    vector<string> itemsData(height);

    for (int i = 0; i < height; i++) {
        getline(cin, mapData[i]);
    }

    for (int i = 0; i < height; i++) {
        getline(cin, itemsData[i]);
    }

    SokoBot solver;
    string solution = solver.solveSokobanPuzzle(
        width, height, mapData, itemsData
    );

    cout << solution << endl;

    return 0;
}
