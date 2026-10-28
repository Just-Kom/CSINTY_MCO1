#pragma once

#include <string>
#include <vector>

// Faithful C++20 counterpart of the Python SokoBot class.
// mapData and itemsData are separate height-by-width grids, just as in Python.
class SokoBot {
public:
    std::string solveSokobanPuzzle(int width, int height,
                                   const std::vector<std::string>& mapData,
                                   const std::vector<std::string>& itemsData);
};
