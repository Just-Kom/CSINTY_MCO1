#include "sokobot.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
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

inline int toTileIndex(Position p, int width) {
    return p.row * width + p.col;
}

inline Position toPosition(int tileIndex, int width) {
    return {tileIndex / width, tileIndex % width};
}

size_t hashCombine(size_t a, size_t b) {
    return a ^ (b + static_cast<size_t>(0x9e3779b97f4a7c15ULL) + (a << 6U) + (a >> 2U));
}

struct PositionHash {
    size_t operator()(Position p) const {
        return hashCombine(hash<int>{}(p.row), hash<int>{}(p.col));
    }
};

struct TileMask {
    int width = 0;
    int totalTiles = 0;
    vector<uint64_t> words;

    TileMask() = default;
    TileMask(int widthVal, int totalTilesVal)
        : width(widthVal),
          totalTiles(totalTilesVal),
          words(static_cast<size_t>((totalTilesVal + 63) / 64), 0ULL) {}

    void insertIndex(int tileIndex) {
        words[static_cast<size_t>(tileIndex >> 6)] |= (1ULL << (tileIndex & 63));
    }

    void insert(Position p) {
        insertIndex(toTileIndex(p, width));
    }

    bool insertAndCheck(int tileIndex) {
        uint64_t mask = 1ULL << (tileIndex & 63);
        uint64_t& word = words[static_cast<size_t>(tileIndex >> 6)];
        if ((word & mask) != 0ULL) return false;
        word |= mask;
        return true;
    }

    bool insertAndCheck(Position p) {
        return insertAndCheck(toTileIndex(p, width));
    }

    bool erase(Position p) {
        int tileIndex = toTileIndex(p, width);
        uint64_t mask = 1ULL << (tileIndex & 63);
        uint64_t& word = words[static_cast<size_t>(tileIndex >> 6)];
        if ((word & mask) == 0ULL) return false;
        word &= ~mask;
        return true;
    }

    bool containsIndex(int tileIndex) const {
        return (words[static_cast<size_t>(tileIndex >> 6)] & (1ULL << (tileIndex & 63))) != 0ULL;
    }

    bool containsPos(Position p) const {
        return containsIndex(toTileIndex(p, width));
    }

    void insertAll(const TileMask& other) {
        for (size_t i = 0; i < words.size(); ++i) {
            words[i] |= other.words[i];
        }
    }

    void clear() {
        fill(words.begin(), words.end(), 0ULL);
    }

    bool empty() const {
        for (uint64_t w : words) {
            if (w != 0ULL) return false;
        }
        return true;
    }

    size_t size() const {
        size_t count = 0;
        for (uint64_t w : words) {
            count += static_cast<size_t>(popcount(w));
        }
        return count;
    }

    Position firstPosition() const {
        for (size_t i = 0; i < words.size(); ++i) {
            if (words[i] != 0ULL) {
                int bit = countr_zero(words[i]);
                int idx = static_cast<int>(i * 64) + bit;
                return toPosition(idx, width);
            }
        }
        return {0, 0};
    }

    struct Iterator {
        const TileMask* maskPtr = nullptr;
        size_t wordIndex = 0;
        uint64_t currentWord = 0ULL;

        void advanceToNextWord() {
            while (currentWord == 0ULL && maskPtr != nullptr && wordIndex < maskPtr->words.size()) {
                currentWord = maskPtr->words[wordIndex++];
            }
        }

        Position operator*() const {
            int bit = countr_zero(currentWord);
            int idx = static_cast<int>((wordIndex - 1) * 64) + bit;
            return toPosition(idx, maskPtr->width);
        }

        Iterator& operator++() {
            currentWord &= (currentWord - 1ULL);
            if (currentWord == 0ULL) {
                advanceToNextWord();
            }
            return *this;
        }

        bool operator==(const Iterator& other) const {
            return currentWord == other.currentWord && wordIndex == other.wordIndex;
        }

        bool operator!=(const Iterator& other) const {
            return !(*this == other);
        }
    };

    Iterator begin() const {
        Iterator it{this, 0, 0ULL};
        it.advanceToNextWord();
        return it;
    }

    Iterator end() const {
        return Iterator{this, words.size(), 0ULL};
    }
};

using TileSet = TileMask;
using Boxes = vector<Position>;
using Neighbors = vector<vector<Position>>;
using Matrix = vector<vector<int>>;
using DistanceMatrices = vector<Matrix>;
using TileDistances = vector<vector<int>>;
using HotspotDictionary = vector<TileMask>;
using FreezeSubset = vector<vector<Boxes>>;
using FeatureSpace = array<int, 3>;

struct BoxesHash {
    size_t operator()(const Boxes& boxes) const {
        size_t result = 0;
        for (Position p : boxes) {
            uint64_t packedPos = (static_cast<uint64_t>(static_cast<uint32_t>(p.row)) << 32U) |
                                 static_cast<uint64_t>(static_cast<uint32_t>(p.col));
            result = hashCombine(result, hash<uint64_t>{}(packedPos));
        }
        return result;
    }
};

struct State {
    Position player;
    Boxes boxes;
    bool operator==(const State&) const = default;
};

struct Parent {
    int previousStateId = -1;
    Position standingTile{};
    Position pushDirection{};
    int pushCount = 0;
};

struct StateRecord {
    State stateKey;
    TileSet reachable;
    FeatureSpace bestAncestor{};
    Parent parent{};
};

struct HungarianScratch {
    vector<long long> u;
    vector<long long> v;
    vector<int> p;
    vector<int> way;
    vector<long long> minv;
    vector<uint8_t> used;
    vector<const int*> rowPointers;

    void prepare(int n, int m) {
        if (static_cast<int>(u.size()) < n + 1) u.resize(static_cast<size_t>(n + 1));
        if (static_cast<int>(v.size()) < m + 1) v.resize(static_cast<size_t>(m + 1));
        if (static_cast<int>(p.size()) < m + 1) p.resize(static_cast<size_t>(m + 1));
        if (static_cast<int>(way.size()) < m + 1) way.resize(static_cast<size_t>(m + 1));
        if (static_cast<int>(minv.size()) < m + 1) minv.resize(static_cast<size_t>(m + 1));
        if (static_cast<int>(used.size()) < m + 1) used.resize(static_cast<size_t>(m + 1));
        fill(u.begin(), u.begin() + n + 1, 0LL);
        fill(v.begin(), v.begin() + m + 1, 0LL);
        fill(p.begin(), p.begin() + m + 1, 0);
        fill(way.begin(), way.begin() + m + 1, 0);
    }
};

struct BfsScratch {
    vector<int> visitStamp;
    int currentStamp = 0;
    vector<Position> frontier;
    vector<int> indexFrontier;
    vector<int> parentTile;
    vector<char> parentMove;

    explicit BfsScratch(int totalTiles)
        : visitStamp(static_cast<size_t>(totalTiles), 0),
          parentTile(static_cast<size_t>(totalTiles), -1),
          parentMove(static_cast<size_t>(totalTiles), 0) {
        frontier.reserve(static_cast<size_t>(totalTiles));
        indexFrontier.reserve(static_cast<size_t>(totalTiles));
    }

    int nextStamp() {
        ++currentStamp;
        if (currentStamp == numeric_limits<int>::max()) {
            fill(visitStamp.begin(), visitStamp.end(), 0);
            currentStamp = 1;
        }
        return currentStamp;
    }
};

constexpr array<Position, 4> DIRECTION = {{{0, -1}, {0, 1}, {1, 0}, {-1, 0}}};
constexpr array<array<Position, 2>, 2> AXES = {{{{{0, -1}, {0, 1}}},
                                                {{{1, 0}, {-1, 0}}}}};

bool contains(const TileSet& set, Position p) {
    return set.containsPos(p);
}

bool contains(const Boxes& boxes, Position p) {
    return binary_search(boxes.begin(), boxes.end(), p);
}

Boxes canonicalBoxes(Boxes result) {
    sort(result.begin(), result.end());
    result.erase(unique(result.begin(), result.end()), result.end());
    return result;
}

TileSet asTileSet(const Boxes& boxes, int width, int totalTiles) {
    TileSet mask(width, totalTiles);
    for (Position p : boxes) mask.insert(p);
    return mask;
}

bool subsetOf(const Boxes& subset, const Boxes& superset) {
    return includes(superset.begin(), superset.end(), subset.begin(), subset.end());
}

bool subsetOf(const TileSet& subset, const TileSet& superset) {
    for (size_t i = 0; i < subset.words.size(); ++i) {
        if ((subset.words[i] & ~superset.words[i]) != 0ULL) return false;
    }
    return true;
}

bool disjoint(const TileSet& a, const TileSet& b) {
    for (size_t i = 0; i < a.words.size(); ++i) {
        if ((a.words[i] & b.words[i]) != 0ULL) return false;
    }
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

int directionIndex(Position p) {
    if (p == Position{0, -1}) return 0;
    if (p == Position{0, 1}) return 1;
    if (p == Position{1, 0}) return 2;
    return 3;
}

bool hasFreezeSubset(Position boxDestination, const Boxes& boxCoordinates,
                     const FreezeSubset& freezeSubset, int width) {
    const auto& subsets = freezeSubset[static_cast<size_t>(toTileIndex(boxDestination, width))];
    for (const Boxes& subset : subsets) {
        if (subsetOf(subset, boxCoordinates)) return true;
    }
    return false;
}

bool detectBlockedAxes(
    Position boxCoordinate,
    const TileSet& validBoxTiles,
    const TileSet& walls,
    const TileSet& currentBoxesMask,
    int width,
    int height,
    vector<uint8_t>& freezeState,
    vector<int>& touchedFreezeTiles,
    vector<Position>& frozenBoxesOut) {

    for (int idx : touchedFreezeTiles) {
        freezeState[static_cast<size_t>(idx)] = 0;
    }
    touchedFreezeTiles.clear();
    frozenBoxesOut.clear();

    auto isFrozen = [&](auto& self, Position currentBox) -> bool {
        int currentIndex = toTileIndex(currentBox, width);
        uint8_t status = freezeState[static_cast<size_t>(currentIndex)];
        if (status == 2 || status == 1) return true;
        if (status == 3) return false;

        array<bool, 2> blockedAxes = {false, false};
        array<array<Position, 2>, 2> adjBoxesPerAxis{};
        array<int, 2> adjCount = {0, 0};

        for (size_t axisIndex = 0; axisIndex < AXES.size(); ++axisIndex) {
            int numOfSimpleDeadlocks = 0;
            for (Position direction : AXES[axisIndex]) {
                Position dest = currentBox + direction;
                if (dest.row < 0 || dest.row >= height || dest.col < 0 || dest.col >= width ||
                    contains(walls, dest)) {
                    blockedAxes[axisIndex] = true;
                    break;
                }
                if (!contains(validBoxTiles, dest)) ++numOfSimpleDeadlocks;
                if (contains(currentBoxesMask, dest)) {
                    int destIndex = toTileIndex(dest, width);
                    if (freezeState[static_cast<size_t>(destIndex)] == 1) {
                        blockedAxes[axisIndex] = true;
                        break;
                    }
                    adjBoxesPerAxis[axisIndex][static_cast<size_t>(adjCount[axisIndex]++)] = dest;
                }
            }
            if (!blockedAxes[axisIndex] && numOfSimpleDeadlocks == 2)
                blockedAxes[axisIndex] = true;
            if (!blockedAxes[axisIndex] && adjCount[axisIndex] == 0) {
                freezeState[static_cast<size_t>(currentIndex)] = 3;
                touchedFreezeTiles.push_back(currentIndex);
                return false;
            }
        }

        size_t savedTouchedSize = touchedFreezeTiles.size();
        freezeState[static_cast<size_t>(currentIndex)] = 1;
        touchedFreezeTiles.push_back(currentIndex);

        for (size_t axisIndex = 0; axisIndex < 2; ++axisIndex) {
            if (!blockedAxes[axisIndex]) {
                for (int i = 0; i < adjCount[axisIndex]; ++i) {
                    if (self(self, adjBoxesPerAxis[axisIndex][static_cast<size_t>(i)])) {
                        blockedAxes[axisIndex] = true;
                        break;
                    }
                }
            }
            if (!blockedAxes[axisIndex]) {
                while (touchedFreezeTiles.size() > savedTouchedSize) {
                    int revertIdx = touchedFreezeTiles.back();
                    touchedFreezeTiles.pop_back();
                    freezeState[static_cast<size_t>(revertIdx)] = 0;
                }
                freezeState[static_cast<size_t>(currentIndex)] = 3;
                touchedFreezeTiles.push_back(currentIndex);
                return false;
            }
        }

        freezeState[static_cast<size_t>(currentIndex)] = 2;
        return true;
    };

    if (!isFrozen(isFrozen, boxCoordinate)) return false;
    for (int idx : touchedFreezeTiles) {
        if (freezeState[static_cast<size_t>(idx)] == 2) {
            frozenBoxesOut.push_back(toPosition(idx, width));
        }
    }
    return true;
}

DistanceMatrices generateDistanceMatrices(
    const Boxes& goalState, int height, int width, const TileSet& validBoxTiles,
    const Neighbors& neighbors, BfsScratch& scratch) {
    DistanceMatrices matrices(goalState.size());
    for (size_t gIdx = 0; gIdx < goalState.size(); ++gIdx) {
        Position goalVal = goalState[gIdx];
        Matrix matrix(static_cast<size_t>(height), vector<int>(static_cast<size_t>(width), -1));
        matrix[static_cast<size_t>(goalVal.row)][static_cast<size_t>(goalVal.col)] = 0;
        int stamp = scratch.nextStamp();
        int goalIdx = toTileIndex(goalVal, width);
        scratch.visitStamp[static_cast<size_t>(goalIdx)] = stamp;
        scratch.frontier.clear();
        scratch.frontier.push_back(goalVal);

        for (size_t i = 0; i < scratch.frontier.size(); ++i) {
            Position currentPos = scratch.frontier[i];
            int currentCtr = matrix[static_cast<size_t>(currentPos.row)][static_cast<size_t>(currentPos.col)];
            for (Position direction : DIRECTION) {
                Position dest = currentPos + direction;
                Position playerStand = dest + direction;
                if (dest.row < 0 || dest.row >= height || dest.col < 0 || dest.col >= width) continue;
                int destIdx = toTileIndex(dest, width);
                if (scratch.visitStamp[static_cast<size_t>(destIdx)] == stamp || !contains(validBoxTiles, dest))
                    continue;
                const auto& destNeighbors = neighbors[static_cast<size_t>(destIdx)];
                if (find(destNeighbors.begin(), destNeighbors.end(), playerStand) != destNeighbors.end()) {
                    scratch.visitStamp[static_cast<size_t>(destIdx)] = stamp;
                    matrix[static_cast<size_t>(dest.row)][static_cast<size_t>(dest.col)] = currentCtr + 1;
                    scratch.frontier.push_back(dest);
                }
            }
        }
        matrices[gIdx] = move(matrix);
    }
    return matrices;
}

TileDistances buildDistanceStack(const DistanceMatrices& distanceMatrices,
                                 const Boxes& goalState, const TileSet& validBoxTiles,
                                 int width, int totalTiles) {
    TileDistances tileDistances(static_cast<size_t>(totalTiles));
    for (Position tile : validBoxTiles) {
        vector<int> distances(goalState.size());
        for (size_t gIdx = 0; gIdx < goalState.size(); ++gIdx) {
            int val = distanceMatrices[gIdx][static_cast<size_t>(tile.row)][static_cast<size_t>(tile.col)];
            distances[gIdx] = (val == -1 ? 999999 : val);
        }
        tileDistances[static_cast<size_t>(toTileIndex(tile, width))] = move(distances);
    }
    return tileDistances;
}

long long hungarianAssignmentCost(HungarianScratch& scratch, int n, int m) {
    if (n == 0) return 0;
    constexpr long long INF = numeric_limits<long long>::max() / 4;
    scratch.prepare(n, m);

    auto& u = scratch.u;
    auto& v = scratch.v;
    auto& p = scratch.p;
    auto& way = scratch.way;
    auto& minv = scratch.minv;
    auto& used = scratch.used;
    const auto& rowPointers = scratch.rowPointers;

    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        fill(minv.begin(), minv.begin() + m + 1, INF);
        fill(used.begin(), used.begin() + m + 1, static_cast<uint8_t>(0));
        do {
            used[static_cast<size_t>(j0)] = 1;
            int i0 = p[static_cast<size_t>(j0)];
            long long delta = INF;
            int j1 = 0;
            const int* costsRow = rowPointers[static_cast<size_t>(i0 - 1)];
            for (int j = 1; j <= m; ++j) {
                if (used[static_cast<size_t>(j)]) continue;
                long long cur = static_cast<long long>(costsRow[j - 1]) - u[static_cast<size_t>(i0)] - v[static_cast<size_t>(j)];
                if (cur < minv[static_cast<size_t>(j)]) {
                    minv[static_cast<size_t>(j)] = cur;
                    way[static_cast<size_t>(j)] = j0;
                }
                if (minv[static_cast<size_t>(j)] < delta) {
                    delta = minv[static_cast<size_t>(j)];
                    j1 = j;
                }
            }
            for (int j = 0; j <= m; ++j) {
                if (used[static_cast<size_t>(j)]) {
                    u[static_cast<size_t>(p[static_cast<size_t>(j)])] += delta;
                    v[static_cast<size_t>(j)] -= delta;
                } else {
                    minv[static_cast<size_t>(j)] -= delta;
                }
            }
            j0 = j1;
        } while (p[static_cast<size_t>(j0)] != 0);
        do {
            int j1 = way[static_cast<size_t>(j0)];
            p[static_cast<size_t>(j0)] = p[static_cast<size_t>(j1)];
            j0 = j1;
        } while (j0 != 0);
    }
    long long result = 0;
    for (int j = 1; j <= m; ++j) {
        int matchedRow = p[static_cast<size_t>(j)];
        if (matchedRow != 0) {
            result += rowPointers[static_cast<size_t>(matchedRow - 1)][j - 1];
        }
    }
    return result;
}

int countHotspots(const Boxes& boxCoordinates, const TileSet& currentBoxesMask,
                  const HotspotDictionary& hotspotDictionary, int width) {
    int hotspots = 0;
    for (Position box : boxCoordinates) {
        const TileSet& trapped = hotspotDictionary[static_cast<size_t>(toTileIndex(box, width))];
        if (!disjoint(trapped, currentBoxesMask)) {
            ++hotspots;
        }
    }
    return hotspots;
}

using HeuristicCache = unordered_map<Boxes, long long, BoxesHash>;

long long calculateHeuristic(const Boxes& boxCoordinates, const TileSet& currentBoxesMask,
                             const TileDistances& tileDistances, HeuristicCache& heuristicCache,
                             const HotspotDictionary& hotspotDictionary, int width, int numGoals,
                             HungarianScratch& hungarianScratch) {
    int n = static_cast<int>(boxCoordinates.size());
    hungarianScratch.rowPointers.resize(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        int tileIdx = toTileIndex(boxCoordinates[static_cast<size_t>(i)], width);
        hungarianScratch.rowPointers[static_cast<size_t>(i)] = tileDistances[static_cast<size_t>(tileIdx)].data();
    }
    long long assignmentCost = hungarianAssignmentCost(hungarianScratch, n, numGoals);
    long long heuristic = assignmentCost +
                          static_cast<long long>(countHotspots(boxCoordinates, currentBoxesMask, hotspotDictionary, width)) * 10;
    heuristicCache[boxCoordinates] = heuristic;
    return heuristic;
}

TileSet reachablePlayerTiles(Position playerPosition, const TileSet& currentBoxesMask,
                             const Neighbors& neighbors, int width, int totalTiles,
                             BfsScratch& scratch) {
    TileSet reachable(width, totalTiles);
    reachable.insert(playerPosition);
    scratch.frontier.clear();
    scratch.frontier.push_back(playerPosition);
    for (size_t i = 0; i < scratch.frontier.size(); ++i) {
        Position current = scratch.frontier[i];
        int currentIdx = toTileIndex(current, width);
        for (Position dest : neighbors[static_cast<size_t>(currentIdx)]) {
            if (!contains(currentBoxesMask, dest) && reachable.insertAndCheck(dest)) {
                scratch.frontier.push_back(dest);
            }
        }
    }
    return reachable;
}

pair<TileSet, TileSet> getCorralFromTile(Position startPos, const TileSet& currentBoxesMask,
                                         const Neighbors& neighbors, int width, int totalTiles,
                                         BfsScratch& scratch) {
    TileSet corral(width, totalTiles);
    TileSet corralBoxes(width, totalTiles);
    corral.insert(startPos);
    scratch.frontier.clear();
    scratch.frontier.push_back(startPos);
    for (size_t i = 0; i < scratch.frontier.size(); ++i) {
        Position current = scratch.frontier[i];
        int currentIdx = toTileIndex(current, width);
        for (Position dest : neighbors[static_cast<size_t>(currentIdx)]) {
            if (contains(corral, dest)) continue;
            if (contains(currentBoxesMask, dest)) {
                corralBoxes.insert(dest);
            } else {
                corral.insert(dest);
                scratch.frontier.push_back(dest);
            }
        }
    }
    return {move(corral), move(corralBoxes)};
}

bool isPICorral(const TileSet& corral, const TileSet& corralBoxes,
                const TileSet& reachable, const TileSet& currentBoxesMask,
                const TileSet& walls, const TileSet& validBoxTiles,
                int width, int height) {
    bool hasValidPush = false;
    for (Position box : corralBoxes) {
        for (Position direction : DIRECTION) {
            Position playerPos = box - direction;
            Position boxDest = box + direction;
            if (playerPos.row < 0 || playerPos.row >= height || playerPos.col < 0 || playerPos.col >= width)
                continue;
            if (boxDest.row < 0 || boxDest.row >= height || boxDest.col < 0 || boxDest.col >= width)
                continue;
            if (contains(walls, playerPos) || !contains(validBoxTiles, boxDest)) continue;
            if (contains(corral, playerPos)) continue;
            if (contains(corralBoxes, playerPos) && contains(corralBoxes, boxDest)) continue;
            if (!contains(corral, boxDest)) return false;
            if (!contains(reachable, playerPos) && !contains(corralBoxes, playerPos)) return false;
            if (contains(reachable, playerPos) && !contains(currentBoxesMask, boxDest)) hasValidPush = true;
        }
    }
    return hasValidPush;
}

optional<string> findWalkingPath(Position start, Position destination,
                                 const Boxes& boxCoordinates, const Neighbors& neighbors,
                                 int width, int totalTiles, BfsScratch& scratch) {
    if (start == destination) return string{};
    TileSet boxMask = asTileSet(boxCoordinates, width, totalTiles);
    int stamp = scratch.nextStamp();
    int startIdx = toTileIndex(start, width);
    int destTargetIdx = toTileIndex(destination, width);

    scratch.visitStamp[static_cast<size_t>(startIdx)] = stamp;
    scratch.parentTile[static_cast<size_t>(startIdx)] = -1;
    scratch.frontier.clear();
    scratch.frontier.push_back(start);

    for (size_t i = 0; i < scratch.frontier.size(); ++i) {
        Position current = scratch.frontier[i];
        int currentIdx = toTileIndex(current, width);
        for (Position dest : neighbors[static_cast<size_t>(currentIdx)]) {
            int nextIdx = toTileIndex(dest, width);
            if (scratch.visitStamp[static_cast<size_t>(nextIdx)] == stamp || contains(boxMask, dest))
                continue;
            Position direction = dest - current;
            scratch.visitStamp[static_cast<size_t>(nextIdx)] = stamp;
            scratch.parentTile[static_cast<size_t>(nextIdx)] = currentIdx;
            scratch.parentMove[static_cast<size_t>(nextIdx)] = directionChar(direction);
            if (nextIdx == destTargetIdx) {
                string path;
                int traceIdx = destTargetIdx;
                while (traceIdx != startIdx) {
                    path.push_back(scratch.parentMove[static_cast<size_t>(traceIdx)]);
                    traceIdx = scratch.parentTile[static_cast<size_t>(traceIdx)];
                }
                reverse(path.begin(), path.end());
                return path;
            }
            scratch.frontier.push_back(dest);
        }
    }
    return nullopt;
}

using TunnelSquares = vector<array<uint8_t, 4>>;

TunnelSquares detectTunnelSquares(const TileSet& interiorTiles, const TileSet& goalTiles,
                                  const TileSet& walls, int width, int height, int totalTiles) {
    TunnelSquares tunnelSquares(static_cast<size_t>(totalTiles), {0, 0, 0, 0});
    for (Position tile : interiorTiles) {
        int tileIdx = toTileIndex(tile, width);
        for (int d = 0; d < 4; ++d) {
            Position direction = DIRECTION[static_cast<size_t>(d)];
            Position prevTile = tile - direction;
            if (prevTile.row < 0 || prevTile.row >= height || prevTile.col < 0 || prevTile.col >= width)
                continue;
            Position side = {-direction.col, direction.row};
            Position tileSide1 = tile + side, tileSide2 = tile - side;
            Position prevSide1 = prevTile + side, prevSide2 = prevTile - side;
            auto isWallPos = [&](Position p) {
                return p.row < 0 || p.row >= height || p.col < 0 || p.col >= width || contains(walls, p);
            };
            if (contains(interiorTiles, prevTile) && !contains(goalTiles, tile) &&
                isWallPos(tileSide1) && isWallPos(tileSide2) &&
                isWallPos(prevSide1) && isWallPos(prevSide2)) {
                tunnelSquares[static_cast<size_t>(tileIdx)][static_cast<size_t>(d)] = 1;
            }
        }
    }
    return tunnelSquares;
}

struct PullState {
    Position player;
    Position box;
};

pair<TileSet, TileSet> getReachableGoalsCount(
    Position goal, optional<Position> hotspotTile, const Neighbors& neighbors,
    int width, int totalTiles, vector<PullState>& pullFrontier,
    vector<int>& pullStamp, int& currentPullStamp) {

    ++currentPullStamp;
    if (currentPullStamp == numeric_limits<int>::max()) {
        fill(pullStamp.begin(), pullStamp.end(), 0);
        currentPullStamp = 1;
    }

    pullFrontier.clear();
    TileSet reached(width, totalTiles);
    TileSet visited(width, totalTiles);
    int goalIdx = toTileIndex(goal, width);

    for (Position dest : neighbors[static_cast<size_t>(goalIdx)]) {
        if (!hotspotTile || dest != *hotspotTile) {
            int destIdx = toTileIndex(dest, width);
            int stateIdx = destIdx * totalTiles + goalIdx;
            pullStamp[static_cast<size_t>(stateIdx)] = currentPullStamp;
            pullFrontier.push_back({dest, goal});
        }
    }
    for (size_t i = 0; i < pullFrontier.size(); ++i) {
        auto [currentPlayerPos, currentBoxCoords] = pullFrontier[i];
        reached.insert(currentBoxCoords);
        visited.insert(currentBoxCoords);
        visited.insert(currentPlayerPos);
        int playerIdx = toTileIndex(currentPlayerPos, width);
        int boxIdx = toTileIndex(currentBoxCoords, width);
        const auto& playerNeighbors = neighbors[static_cast<size_t>(playerIdx)];

        Position delta = currentPlayerPos - currentBoxCoords;
        if (abs(delta.row) + abs(delta.col) == 1) {
            Position dest = currentPlayerPos + delta;
            if (find(playerNeighbors.begin(), playerNeighbors.end(), dest) != playerNeighbors.end() &&
                (!hotspotTile || dest != *hotspotTile)) {
                int destIdx = toTileIndex(dest, width);
                int newStateIdx = destIdx * totalTiles + playerIdx;
                if (pullStamp[static_cast<size_t>(newStateIdx)] != currentPullStamp) {
                    pullStamp[static_cast<size_t>(newStateIdx)] = currentPullStamp;
                    pullFrontier.push_back({dest, currentPlayerPos});
                }
            }
        }
        for (Position destination : playerNeighbors) {
            if (destination != currentBoxCoords && (!hotspotTile || destination != *hotspotTile)) {
                int destIdx = toTileIndex(destination, width);
                int nextStateIdx = destIdx * totalTiles + boxIdx;
                if (pullStamp[static_cast<size_t>(nextStateIdx)] != currentPullStamp) {
                    pullStamp[static_cast<size_t>(nextStateIdx)] = currentPullStamp;
                    pullFrontier.push_back({destination, currentBoxCoords});
                }
            }
        }
    }
    return {move(reached), move(visited)};
}

pair<TileSet, HotspotDictionary> generateHotspotTable(
    const Boxes& goalState, const TileSet& interiorTiles, const Neighbors& neighbors,
    int width, int totalTiles) {
    vector<int> reachableGoalCount(static_cast<size_t>(totalTiles), 0);
    vector<TileSet> goalReached(static_cast<size_t>(totalTiles));
    vector<TileSet> visitedByGoal(static_cast<size_t>(totalTiles));
    vector<uint8_t> isGoalTile(static_cast<size_t>(totalTiles), 0);

    vector<PullState> pullFrontier;
    pullFrontier.reserve(static_cast<size_t>(totalTiles * 4));
    vector<int> pullStamp(static_cast<size_t>(totalTiles * totalTiles), 0);
    int currentPullStamp = 0;

    for (Position goal : goalState) {
        int goalIdx = toTileIndex(goal, width);
        isGoalTile[static_cast<size_t>(goalIdx)] = 1;
        auto [reached, visited] = getReachableGoalsCount(
            goal, nullopt, neighbors, width, totalTiles, pullFrontier, pullStamp, currentPullStamp);
        for (Position tile : reached) {
            ++reachableGoalCount[static_cast<size_t>(toTileIndex(tile, width))];
        }
        goalReached[static_cast<size_t>(goalIdx)] = move(reached);
        visitedByGoal[static_cast<size_t>(goalIdx)] = move(visited);
    }

    TileSet validBoxTiles(width, totalTiles);
    for (Position tile : interiorTiles) {
        int tileIdx = toTileIndex(tile, width);
        if (reachableGoalCount[static_cast<size_t>(tileIdx)] > 0) {
            validBoxTiles.insertIndex(tileIdx);
        }
    }

    HotspotDictionary hotspotDictionary(static_cast<size_t>(totalTiles), TileSet(width, totalTiles));
    vector<int> hotspotGoalCount(static_cast<size_t>(totalTiles), 0);

    for (Position hotspotTile : validBoxTiles) {
        int hotspotIdx = toTileIndex(hotspotTile, width);
        fill(hotspotGoalCount.begin(), hotspotGoalCount.end(), 0);
        for (Position goal : goalState) {
            if (goal == hotspotTile) continue;
            int goalIdx = toTileIndex(goal, width);
            TileSet reached;
            if (contains(visitedByGoal[static_cast<size_t>(goalIdx)], hotspotTile)) {
                reached = getReachableGoalsCount(
                    goal, hotspotTile, neighbors, width, totalTiles, pullFrontier, pullStamp, currentPullStamp).first;
            } else {
                reached = goalReached[static_cast<size_t>(goalIdx)];
            }
            for (Position tile : reached) {
                int tileIdx = toTileIndex(tile, width);
                if (validBoxTiles.containsIndex(tileIdx)) {
                    ++hotspotGoalCount[static_cast<size_t>(tileIdx)];
                }
            }
        }
        for (Position tile : validBoxTiles) {
            if (tile == hotspotTile) continue;
            int tileIdx = toTileIndex(tile, width);
            int expectedCount = reachableGoalCount[static_cast<size_t>(tileIdx)];
            if (isGoalTile[static_cast<size_t>(hotspotIdx)] &&
                 goalReached[static_cast<size_t>(hotspotIdx)].containsIndex(tileIdx)) {
                --expectedCount;
            }
            if (hotspotGoalCount[static_cast<size_t>(tileIdx)] < expectedCount) {
                hotspotDictionary[static_cast<size_t>(hotspotIdx)].insertIndex(tileIdx);
            }
        }
    }
    return {move(validBoxTiles), move(hotspotDictionary)};
}

int getPackedBoxes(const Boxes& boxCoordinates, const TileSet& goalTiles) {
    int packed = 0;
    for (Position box : boxCoordinates) {
        if (contains(goalTiles, box)) ++packed;
    }
    return packed;
}

int countRegions(const vector<int>& interiorTileIndices, const TileSet& currentBoxesMask,
                 const vector<vector<int>>& neighborIndices, BfsScratch& scratch) {
    int stamp = scratch.nextStamp();
    int rooms = 0;
    for (int currentPos : interiorTileIndices) {
        if (currentBoxesMask.containsIndex(currentPos) ||
            scratch.visitStamp[static_cast<size_t>(currentPos)] == stamp) {
            continue;
        }
        ++rooms;
        scratch.indexFrontier.clear();
        scratch.indexFrontier.push_back(currentPos);
        scratch.visitStamp[static_cast<size_t>(currentPos)] = stamp;
        for (size_t i = 0; i < scratch.indexFrontier.size(); ++i) {
            int current = scratch.indexFrontier[i];
            for (int dest : neighborIndices[static_cast<size_t>(current)]) {
                if (!currentBoxesMask.containsIndex(dest) &&
                    scratch.visitStamp[static_cast<size_t>(dest)] != stamp) {
                    scratch.visitStamp[static_cast<size_t>(dest)] = stamp;
                    scratch.indexFrontier.push_back(dest);
                }
            }
        }
    }
    return rooms;
}

struct RoomInfo {
    vector<vector<int>> rooms;
    vector<vector<int>> graphIds;
    vector<int> tileToRoom;
};

RoomInfo generateRoomsAndGraph(const TileSet& interiorTiles, const Neighbors& neighbors,
                               int width, int totalTiles) {
    TileSet roomTiles(width, totalTiles);
    for (Position tile : interiorTiles) {
        auto [row, col] = pair{tile.row, tile.col};
        array<Position, 6> twoByThree = {{{row, col}, {row, col + 1}, {row, col + 2},
                                          {row + 1, col}, {row + 1, col + 1}, {row + 1, col + 2}}};
        array<Position, 6> threeByTwo = {{{row, col}, {row + 1, col}, {row + 2, col},
                                          {row, col + 1}, {row + 1, col + 1}, {row + 2, col + 1}}};
        bool first = true, second = true;
        for (Position p : twoByThree) if (!contains(interiorTiles, p)) first = false;
        for (Position p : threeByTwo) if (!contains(interiorTiles, p)) second = false;
        if (first) for (Position p : twoByThree) roomTiles.insert(p);
        if (second) for (Position p : threeByTwo) roomTiles.insert(p);
    }

    RoomInfo result;
    result.tileToRoom.assign(static_cast<size_t>(totalTiles), -1);
    while (!roomTiles.empty()) {
        Position startPos = roomTiles.firstPosition();
        roomTiles.erase(startPos);
        int roomId = static_cast<int>(result.rooms.size());
        int startIdx = toTileIndex(startPos, width);
        vector<int> currentRoom = {startIdx};
        vector<Position> frontier = {startPos};
        result.tileToRoom[static_cast<size_t>(startIdx)] = roomId;
        for (size_t i = 0; i < frontier.size(); ++i) {
            Position currentPos = frontier[i];
            int currentIdx = toTileIndex(currentPos, width);
            for (Position dest : neighbors[static_cast<size_t>(currentIdx)]) {
                if (roomTiles.erase(dest)) {
                    int destIdx = toTileIndex(dest, width);
                    currentRoom.push_back(destIdx);
                    result.tileToRoom[static_cast<size_t>(destIdx)] = roomId;
                    frontier.push_back(dest);
                }
            }
        }
        result.rooms.push_back(move(currentRoom));
    }

    vector<set<int>> graphSets(result.rooms.size());
    for (size_t roomId = 0; roomId < result.rooms.size(); ++roomId) {
        vector<Position> frontier;
        TileSet explored(width, totalTiles);
        for (int tileIdx : result.rooms[roomId]) {
            Position p = toPosition(tileIdx, width);
            frontier.push_back(p);
            explored.insertIndex(tileIdx);
        }
        for (size_t i = 0; i < frontier.size(); ++i) {
            Position currentPos = frontier[i];
            int currentIdx = toTileIndex(currentPos, width);
            for (Position dest : neighbors[static_cast<size_t>(currentIdx)]) {
                if (!explored.insertAndCheck(dest)) continue;
                int destIdx = toTileIndex(dest, width);
                int neighborRoom = result.tileToRoom[static_cast<size_t>(destIdx)];
                if (neighborRoom != -1) {
                    graphSets[roomId].insert(neighborRoom);
                    graphSets[static_cast<size_t>(neighborRoom)].insert(static_cast<int>(roomId));
                } else {
                    frontier.push_back(dest);
                }
            }
        }
    }
    result.graphIds.resize(result.rooms.size());
    for (size_t roomId = 0; roomId < result.rooms.size(); ++roomId) {
        result.graphIds[roomId].assign(graphSets[roomId].begin(), graphSets[roomId].end());
    }
    return result;
}

int countRoomConnectivity(const TileSet& currentBoxesMask, const RoomInfo& roomInfo,
                          const vector<vector<int>>& neighborIndices, BfsScratch& scratch,
                          vector<uint8_t>& connectedRoomsMask) {
    int brokenEdges = 0;
    for (size_t roomId = 0; roomId < roomInfo.rooms.size(); ++roomId) {
        const auto& roomTiles = roomInfo.rooms[roomId];
        const auto& adjRoomIds = roomInfo.graphIds[roomId];
        if (adjRoomIds.empty()) continue;

        int stamp = scratch.nextStamp();
        fill(connectedRoomsMask.begin(), connectedRoomsMask.end(), 0);
        size_t connectedCount = 0;

        for (int startPos : roomTiles) {
            if (currentBoxesMask.containsIndex(startPos) ||
                scratch.visitStamp[static_cast<size_t>(startPos)] == stamp) {
                continue;
            }
            scratch.indexFrontier.clear();
            scratch.indexFrontier.push_back(startPos);
            scratch.visitStamp[static_cast<size_t>(startPos)] = stamp;
            for (size_t i = 0; i < scratch.indexFrontier.size(); ++i) {
                int currentPos = scratch.indexFrontier[i];
                for (int dest : neighborIndices[static_cast<size_t>(currentPos)]) {
                    if (currentBoxesMask.containsIndex(dest) ||
                        scratch.visitStamp[static_cast<size_t>(dest)] == stamp) {
                        continue;
                    }
                    scratch.visitStamp[static_cast<size_t>(dest)] = stamp;
                    int neighborRoom = roomInfo.tileToRoom[static_cast<size_t>(dest)];
                    if (neighborRoom != -1) {
                        if (neighborRoom == static_cast<int>(roomId)) {
                            scratch.indexFrontier.push_back(dest);
                        } else if (!connectedRoomsMask[static_cast<size_t>(neighborRoom)]) {
                            connectedRoomsMask[static_cast<size_t>(neighborRoom)] = 1;
                            ++connectedCount;
                        }
                    } else {
                        scratch.indexFrontier.push_back(dest);
                    }
                }
            }
            if (connectedCount == adjRoomIds.size()) break;
        }
        for (int neighborRoom : adjRoomIds) {
            if (static_cast<int>(roomId) < neighborRoom &&
                !connectedRoomsMask[static_cast<size_t>(neighborRoom)]) {
                ++brokenEdges;
            }
        }
    }
    return brokenEdges;
}

int countMobility(const Boxes& boxCoordinates, const TileSet& reachable,
                  const Neighbors& neighbors, int width) {
    int mobility = 0;
    for (Position box : boxCoordinates) {
        int boxIdx = toTileIndex(box, width);
        for (Position dest : neighbors[static_cast<size_t>(boxIdx)]) {
            if (contains(reachable, dest)) ++mobility;
        }
    }
    return mobility;
}

pair<FeatureSpace, FeatureSpace> getProjectedFeatureSpace(
    const FeatureSpace& rawFeatureSpace, const FeatureSpace& bestAncestorFeatureSpace) {
    if (rawFeatureSpace < bestAncestorFeatureSpace) return {rawFeatureSpace, rawFeatureSpace};
    return {bestAncestorFeatureSpace, bestAncestorFeatureSpace};
}

struct HeapEntry {
    long long heuristic;
    long long negativeMobility;
    long long negativeCounter;
    int stateId;
};

struct HeapCompare {
    bool operator()(const HeapEntry& a, const HeapEntry& b) const {
        return tie(a.heuristic, a.negativeMobility, a.negativeCounter) >
               tie(b.heuristic, b.negativeMobility, b.negativeCounter);
    }
};

using FeatureHeap = priority_queue<HeapEntry, vector<HeapEntry>, HeapCompare>;

struct MacroNode {
    int parentStateId;
    Position currentBox;
    Boxes boxes;
    TileSet reachable;
};

}

string SokoBot::solveSokobanPuzzle(int width, int height, const vector<string>& mapData, const vector<string>& itemsData) {
    int totalTiles = width * height;
    BfsScratch bfsScratch(totalTiles);
    HungarianScratch hungarianScratch;

    Boxes goalState;
    for (int row = 0; row < static_cast<int>(mapData.size()); ++row) {
        for (int col = 0; col < static_cast<int>(mapData[static_cast<size_t>(row)].size()); ++col) {
            if (mapData[static_cast<size_t>(row)][static_cast<size_t>(col)] == '.')
                goalState.push_back({row, col});
        }
    }
    goalState = canonicalBoxes(move(goalState));
    const int numGoals = static_cast<int>(goalState.size());
    const TileSet goalTiles = asTileSet(goalState, width, totalTiles);

    Boxes boxCoordinates;
    Position playerPosition{};
    for (int row = 0; row < static_cast<int>(itemsData.size()); ++row) {
        for (int col = 0; col < static_cast<int>(itemsData[static_cast<size_t>(row)].size()); ++col) {
            char ch = itemsData[static_cast<size_t>(row)][static_cast<size_t>(col)];
            if (ch == '$') boxCoordinates.push_back({row, col});
            else if (ch == '@') playerPosition = {row, col};
        }
    }
    boxCoordinates = canonicalBoxes(move(boxCoordinates));

    TileSet walls(width, totalTiles);
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            if (mapData[static_cast<size_t>(row)][static_cast<size_t>(col)] == '#') {
                walls.insert({row, col});
            }
        }
    }

    Neighbors neighbors(static_cast<size_t>(totalTiles));
    vector<vector<int>> neighborIndices(static_cast<size_t>(totalTiles));
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            Position tile{row, col};
            if (contains(walls, tile)) continue;
            int tileIdx = toTileIndex(tile, width);
            vector<Position> tileNeighbors;
            vector<int> tileNeighborIndices;
            for (Position d : DIRECTION) {
                Position dest = tile + d;
                if (dest.row >= 0 && dest.row < height && dest.col >= 0 && dest.col < width &&
                    !contains(walls, dest)) {
                    tileNeighbors.push_back(dest);
                    tileNeighborIndices.push_back(toTileIndex(dest, width));
                }
            }
            neighbors[static_cast<size_t>(tileIdx)] = move(tileNeighbors);
            neighborIndices[static_cast<size_t>(tileIdx)] = move(tileNeighborIndices);
        }
    }

    TileSet emptyBoxesMask(width, totalTiles);
    TileSet interiorTiles = reachablePlayerTiles(playerPosition, emptyBoxesMask, neighbors, width, totalTiles, bfsScratch);
    vector<int> interiorTileIndices;
    for (Position tile : interiorTiles) {
        interiorTileIndices.push_back(toTileIndex(tile, width));
    }

    auto [validBoxTiles, hotspotDictionary] = generateHotspotTable(goalState, interiorTiles, neighbors, width, totalTiles);
    DistanceMatrices distances = generateDistanceMatrices(goalState, height, width, validBoxTiles, neighbors, bfsScratch);
    TileDistances tileDistances = buildDistanceStack(distances, goalState, validBoxTiles, width, totalTiles);
    FreezeSubset freezeSubset(static_cast<size_t>(totalTiles));
    RoomInfo roomInfo = generateRoomsAndGraph(interiorTiles, neighbors, width, totalTiles);
    vector<uint8_t> connectedRoomsMask(roomInfo.rooms.size(), 0);
    HeuristicCache heuristicCache;
    unordered_map<Boxes, FeatureSpace, BoxesHash> featureSpaceCache;

    TileSet startBoxesMask = asTileSet(boxCoordinates, width, totalTiles);
    TileSet startReachable = reachablePlayerTiles(playerPosition, startBoxesMask, neighbors, width, totalTiles, bfsScratch);
    TunnelSquares tunnelTiles = detectTunnelSquares(interiorTiles, goalTiles, walls, width, height, totalTiles);

    auto getFeatureSpace = [&](const Boxes& boxes, const TileSet& boxesMask) -> FeatureSpace {
        if (auto it = featureSpaceCache.find(boxes); it != featureSpaceCache.end()) return it->second;
        int packed = getPackedBoxes(boxes, goalTiles);
        int connectivity = countRegions(interiorTileIndices, boxesMask, neighborIndices, bfsScratch);
        int roomConnectivity = countRoomConnectivity(boxesMask, roomInfo, neighborIndices, bfsScratch, connectedRoomsMask);
        FeatureSpace rawFeatureSpace = {-packed, connectivity, roomConnectivity};
        featureSpaceCache[boxes] = rawFeatureSpace;
        return rawFeatureSpace;
    };

    State startKey{playerPosition, boxCoordinates};
    FeatureSpace startFeatureSpace = getFeatureSpace(boxCoordinates, startBoxesMask);
    long long heuristic = calculateHeuristic(
        boxCoordinates, startBoxesMask, tileDistances, heuristicCache, hotspotDictionary, width, numGoals, hungarianScratch);
    int mobility = countMobility(boxCoordinates, startReachable, neighbors, width);

    vector<StateRecord> statePool;
    statePool.reserve(65536);
    statePool.push_back(StateRecord{
        startKey,
        startReachable,
        startFeatureSpace,
        Parent{-1, Position{}, Position{}, 0}
    });

    unordered_map<Boxes, vector<TileSet>, BoxesHash> exploredReigons;
    exploredReigons[boxCoordinates].push_back(startReachable);

    map<FeatureSpace, FeatureHeap> featureSpaceQueues;
    long long stateCounter = 0;
    featureSpaceQueues[startFeatureSpace].push({heuristic, -mobility, -stateCounter, 0});
    ++stateCounter;

    vector<uint8_t> freezeState(static_cast<size_t>(totalTiles), 0);
    vector<int> touchedFreezeTiles;
    touchedFreezeTiles.reserve(static_cast<size_t>(totalTiles));
    vector<Position> frozenBoxes;
    frozenBoxes.reserve(goalState.size());
    vector<Position> boxesToExplore;
    boxesToExplore.reserve(boxCoordinates.size());
    vector<Position> corralSeeds;
    corralSeeds.reserve(boxCoordinates.size() * 4);
    vector<MacroNode> macroFrontier;

    while (!featureSpaceQueues.empty()) {
        for (auto heapIt = featureSpaceQueues.begin(); heapIt != featureSpaceQueues.end(); ) {
            if (heapIt->second.empty()) {
                heapIt = featureSpaceQueues.erase(heapIt);
                continue;
            }
            HeapEntry entry = heapIt->second.top();
            heapIt->second.pop();
            if (heapIt->second.empty()) {
                heapIt = featureSpaceQueues.erase(heapIt);
            } else {
                ++heapIt;
            }

            int currentStateId = entry.stateId;
            Boxes currentBoxCoords = statePool[static_cast<size_t>(currentStateId)].stateKey.boxes;
            TileSet reachable = move(statePool[static_cast<size_t>(currentStateId)].reachable);
            statePool[static_cast<size_t>(currentStateId)].reachable = TileSet{};
            FeatureSpace currentBestAncestor = statePool[static_cast<size_t>(currentStateId)].bestAncestor;

            if (currentBoxCoords == goalState) {
                vector<Parent> transitions;
                int traceStateId = currentStateId;
                while (statePool[static_cast<size_t>(traceStateId)].parent.previousStateId != -1) {
                    const Parent& transition = statePool[static_cast<size_t>(traceStateId)].parent;
                    transitions.push_back(transition);
                    traceStateId = transition.previousStateId;
                }
                reverse(transitions.begin(), transitions.end());
                string winPath;
                for (const Parent& transition : transitions) {
                    const State& prevState = statePool[static_cast<size_t>(transition.previousStateId)].stateKey;
                    optional<string> walk = findWalkingPath(
                        prevState.player, transition.standingTile,
                        prevState.boxes, neighbors, width, totalTiles, bfsScratch);
                    if (!walk) throw runtime_error("Invalid reconstructed push path");
                    winPath += *walk;
                    winPath.append(static_cast<size_t>(transition.pushCount),
                                   directionChar(transition.pushDirection));
                }
                cout << "Total states explored: " << statePool.size() << '\n';
                cout << "Total unique box configurations explored: " << exploredReigons.size() << '\n';
                return winPath;
            }

            TileSet currentBoxesMask = asTileSet(currentBoxCoords, width, totalTiles);
            boxesToExplore = currentBoxCoords;
            TileSet exploredCorral(width, totalTiles);
            TileSet corralSeedMask(width, totalTiles);
            corralSeeds.clear();

            for (Position box : currentBoxCoords) {
                for (Position direction : DIRECTION) {
                    Position standingTile = box - direction;
                    Position boxDestination = box + direction;
                    if (standingTile.row < 0 || standingTile.row >= height ||
                        standingTile.col < 0 || standingTile.col >= width) continue;
                    if (boxDestination.row < 0 || boxDestination.row >= height ||
                        boxDestination.col < 0 || boxDestination.col >= width) continue;
                    if (contains(reachable, standingTile) && contains(validBoxTiles, boxDestination)) {
                        if (!contains(reachable, boxDestination) && !contains(currentBoxesMask, boxDestination)) {
                            if (corralSeedMask.insertAndCheck(boxDestination)) {
                                corralSeeds.push_back(boxDestination);
                            }
                        }
                    }
                }
            }

            for (Position currentPos : corralSeeds) {
                if (contains(exploredCorral, currentPos)) continue;
                auto [corral, corralBoxes] = getCorralFromTile(
                    currentPos, currentBoxesMask, neighbors, width, totalTiles, bfsScratch);
                exploredCorral.insertAll(corral);
                size_t corralBoxCount = corralBoxes.size();
                if (corralBoxCount == 0 || corralBoxCount == currentBoxCoords.size()) continue;
                if (subsetOf(corralBoxes, goalTiles) && disjoint(corral, goalTiles)) continue;
                if (isPICorral(corral, corralBoxes, reachable, currentBoxesMask, walls, validBoxTiles, width, height)) {
                    boxesToExplore.clear();
                    for (Position cb : corralBoxes) boxesToExplore.push_back(cb);
                    break;
                }
            }

            for (Position box : boxesToExplore) {
                macroFrontier.clear();
                macroFrontier.push_back({currentStateId, box, currentBoxCoords, reachable});
                for (size_t macroIndex = 0; macroIndex < macroFrontier.size(); ++macroIndex) {
                    int macroParentId = macroFrontier[macroIndex].parentStateId;
                    Position currentBox = macroFrontier[macroIndex].currentBox;
                    Boxes macroBoxCoords = move(macroFrontier[macroIndex].boxes);
                    TileSet macroReachable = move(macroFrontier[macroIndex].reachable);
                    TileSet macroBoxesMask = asTileSet(macroBoxCoords, width, totalTiles);

                    for (int d = 0; d < 4; ++d) {
                        Position direction = DIRECTION[static_cast<size_t>(d)];
                        Position standingTile = currentBox - direction;
                        if (standingTile.row < 0 || standingTile.row >= height ||
                            standingTile.col < 0 || standingTile.col >= width) continue;
                        if (!contains(macroReachable, standingTile)) continue;

                        Position boxDestination = currentBox + direction;
                        if (boxDestination.row < 0 || boxDestination.row >= height ||
                            boxDestination.col < 0 || boxDestination.col >= width) continue;
                        if (!contains(validBoxTiles, boxDestination) || contains(macroBoxesMask, boxDestination))
                            continue;

                        Position newPlayerPos = currentBox;
                        int pushCount = 1;
                        Position movingBox = boxDestination;
                        while (tunnelTiles[static_cast<size_t>(toTileIndex(movingBox, width))][static_cast<size_t>(d)] != 0) {
                            Position nextBox = movingBox + direction;
                            if (nextBox.row < 0 || nextBox.row >= height ||
                                nextBox.col < 0 || nextBox.col >= width) break;
                            if (!contains(validBoxTiles, nextBox) || contains(macroBoxesMask, nextBox)) break;
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

                        TileSet newBoxesMask = macroBoxesMask;
                        newBoxesMask.erase(currentBox);
                        newBoxesMask.insert(boxDestination);

                        long long newHeuristic;
                        if (auto it = heuristicCache.find(newBoxCoords); it != heuristicCache.end()) {
                            newHeuristic = it->second;
                        } else {
                            if (hasFreezeSubset(boxDestination, newBoxCoords, freezeSubset, width)) continue;
                            bool isBoxFrozen = detectBlockedAxes(
                                boxDestination, validBoxTiles, walls, newBoxesMask,
                                width, height, freezeState, touchedFreezeTiles, frozenBoxes);
                            bool nonGoalFrozen = false;
                            if (isBoxFrozen) {
                                for (Position b : frozenBoxes) {
                                    if (!contains(goalTiles, b)) {
                                        nonGoalFrozen = true;
                                        break;
                                    }
                                }
                            }
                            if (nonGoalFrozen) {
                                if (frozenBoxes.size() <= 4) {
                                    Boxes frozenPattern = canonicalBoxes(frozenBoxes);
                                    for (Position frozenBox : frozenPattern) {
                                        auto& patternList = freezeSubset[static_cast<size_t>(toTileIndex(frozenBox, width))];
                                        if (find(patternList.begin(), patternList.end(), frozenPattern) == patternList.end()) {
                                            patternList.push_back(frozenPattern);
                                        }
                                    }
                                }
                                heuristicCache[newBoxCoords] = 999999;
                                continue;
                            }
                            newHeuristic = calculateHeuristic(
                                newBoxCoords, newBoxesMask, tileDistances,
                                heuristicCache, hotspotDictionary, width, numGoals, hungarianScratch);
                        }
                        if (newHeuristic >= 999999) continue;

                        TileSet newReachable = reachablePlayerTiles(
                            newPlayerPos, newBoxesMask, neighbors, width, totalTiles, bfsScratch);
                        exploredReigons[newBoxCoords].push_back(newReachable);

                        FeatureSpace rawFeatureSpace = getFeatureSpace(newBoxCoords, newBoxesMask);
                        auto [projectedFeatureSpace, newBestAncestor] =
                            getProjectedFeatureSpace(rawFeatureSpace, currentBestAncestor);
                        int newMobility = countMobility(newBoxCoords, newReachable, neighbors, width);

                        int newStateId = static_cast<int>(statePool.size());
                        statePool.push_back(StateRecord{
                            State{newPlayerPos, newBoxCoords},
                            newReachable,
                            newBestAncestor,
                            Parent{macroParentId, standingTile, direction, pushCount}
                        });

                        featureSpaceQueues[projectedFeatureSpace].push(
                            {newHeuristic, -newMobility, -stateCounter, newStateId});
                        ++stateCounter;
                        macroFrontier.push_back({newStateId, boxDestination, move(newBoxCoords), move(newReachable)});
                    }
                }
            }
        }
    }
    return "";
}

static string lastSolution;

extern "C" {
#ifdef _WIN32
__declspec(dllexport)
#endif
const char* solve_puzzle_c(int width, int height, const char* mapFlat, const char* itemsFlat, double timeLimit) {
    (void)timeLimit;
    vector<string> mapData(static_cast<size_t>(height), string(static_cast<size_t>(width), ' '));
    vector<string> itemsData(static_cast<size_t>(height), string(static_cast<size_t>(width), ' '));
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            mapData[static_cast<size_t>(row)][static_cast<size_t>(col)] = mapFlat[row * width + col];
            itemsData[static_cast<size_t>(row)][static_cast<size_t>(col)] = itemsFlat[row * width + col];
        }
    }
    SokoBot solver;
    lastSolution = solver.solveSokobanPuzzle(width, height, mapData, itemsData);
    return lastSolution.c_str();
}
}

int main() {
    int width, height;
    if (!(cin >> width >> height)) return 0;
    cin.ignore(numeric_limits<streamsize>::max(), '\n');

    vector<string> mapData(static_cast<size_t>(height));
    vector<string> itemsData(static_cast<size_t>(height));

    for (int i = 0; i < height; i++) {
        getline(cin, mapData[static_cast<size_t>(i)]);
    }

    for (int i = 0; i < height; i++) {
        getline(cin, itemsData[static_cast<size_t>(i)]);
    }

    SokoBot solver;
    string solution = solver.solveSokobanPuzzle(
        width, height, mapData, itemsData
    );

    cout << "SOLUTION:" << solution << endl;

    return 0;
}
