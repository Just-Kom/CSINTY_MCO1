import ctypes
import os
import subprocess
import sys
import tempfile
from pathlib import Path



_BRIDGE = r'''
#include "sokobot.hpp"
#include <string>
#include <vector>
#include <exception>

extern "C" const char* solve_puzzle_c(int width, int height,
                                       const char* map_flat,
                                       const char* items_flat,
                                       double /*time_limit*/) {
    static thread_local std::string result;
    try {
        if (width <= 0 || height <= 0 || !map_flat || !items_flat) {
            result.clear();
            return result.c_str();
        }
        std::vector<std::string> map_data, items_data;
        map_data.reserve(height);
        items_data.reserve(height);
        for (int row = 0; row < height; ++row) {
            map_data.emplace_back(map_flat + static_cast<size_t>(row) * width,
                                  static_cast<size_t>(width));
            items_data.emplace_back(items_flat + static_cast<size_t>(row) * width,
                                    static_cast<size_t>(width));
        }
        SokoBot solver;
        result = solver.solveSokobanPuzzle(width, height, map_data, items_data);
    } catch (const std::exception&) {
        result.clear();
    } catch (...) {
        result.clear();
    }
    return result.c_str();
}
'''


class SokoBot:
    def __init__(self):
        self.solverDir = Path(__file__).resolve().parent
        self.cppPath = self.solverDir / "sokobot.cpp"
        self.hppPath = self.solverDir / "sokobot.hpp"
        self.libraryPath = self.solverDir / (
            "sokobot_cpp.dll" if sys.platform == "win32" else "sokobot_cpp.so"
        )
        self.cppLib = None
        self._loadSolver()

    def _loadSolver(self):
        if not self.cppPath.is_file() or not self.hppPath.is_file():
            raise FileNotFoundError("Both sokobot.cpp and sokobot.hpp must be in " + str(self.solverDir))

        needsCompile = (
            not self.libraryPath.exists()
            or max(self.cppPath.stat().st_mtime, self.hppPath.stat().st_mtime)
            > self.libraryPath.stat().st_mtime
        )
        if not needsCompile:
            try:
                lib = ctypes.CDLL(str(self.libraryPath))
                getattr(lib, "solve_puzzle_c")
            except (OSError, AttributeError):
                needsCompile = True

        if needsCompile:
            with tempfile.TemporaryDirectory(prefix="sokobot_bridge_") as temp_dir:
                bridge = Path(temp_dir) / "sokobot_bridge.cpp"
                bridge.write_text(_BRIDGE, encoding="utf-8")
                command = ["g++", "-O3", "-std=c++20", "-shared"]
                if sys.platform != "win32":
                    command.append("-fPIC")
                command += [
                    "-I", str(self.solverDir),
                    str(self.cppPath), str(bridge),
                    "-o", str(self.libraryPath),
                ]
                subprocess.run(command, cwd=self.solverDir, check=True)

        self.cppLib = ctypes.CDLL(str(self.libraryPath))
        self.cppLib.solve_puzzle_c.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_double
        ]
        self.cppLib.solve_puzzle_c.restype = ctypes.c_char_p

    def solveSokobanPuzzle(self, width, height, mapData, itemsData):
        timeLimit = float(os.environ.get("SOKOBOT_TIME_LIMIT", "0.0"))
        mapFlat = "".join("".join(row[:width]).ljust(width) for row in mapData[:height]).encode("ascii")
        itemsFlat = "".join("".join(row[:width]).ljust(width) for row in itemsData[:height]).encode("ascii")
        result = self.cppLib.solve_puzzle_c(width, height, mapFlat, itemsFlat, timeLimit)
        return result.decode("ascii") if result else ""
