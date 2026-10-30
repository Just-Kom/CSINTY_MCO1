import ctypes
import os
import shutil
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

        if sys.platform == "win32":
            self.libraryPath = self.solverDir / "sokobot_cpp.dll"
        elif sys.platform.startswith("linux") or sys.platform == "darwin":
            self.libraryPath = self.solverDir / "sokobot_cpp.so"
        else:
            raise RuntimeError("Unsupported operating system")

        self.cppLib = None
        self.errors = []
        self._loadSolver()

    def _tryLoad(self, path):
        try:
            if sys.platform == "win32":
                gpp = shutil.which("g++")
                if gpp and hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(str(Path(gpp).resolve().parent))
            lib = ctypes.CDLL(str(path))
            getattr(lib, "solve_puzzle_c")
            return lib
        except (OSError, AttributeError) as e:
            self.errors.append(f"load {path}: {e}")
            return None

    def _compile(self, outPath):
        with tempfile.TemporaryDirectory(prefix="sokobot_bridge_") as temp_dir:
            bridge = Path(temp_dir) / "sokobot_bridge.cpp"
            bridge.write_text(_BRIDGE, encoding="utf-8")
            command = ["g++", "-O3", "-std=c++20", "-shared"]
            if sys.platform == "win32":
                command += ["-static", "-static-libgcc", "-static-libstdc++"]
            else:
                command.append("-fPIC")
            command += [
                "-I", str(self.solverDir),
                str(self.cppPath), str(bridge),
                "-o", str(outPath),
            ]
            proc = subprocess.run(
                command, cwd=self.solverDir, capture_output=True, text=True
            )
            if proc.returncode != 0:
                raise RuntimeError(f"g++ failed:\n{proc.stderr}")

    def _loadSolver(self):
        if not self.cppPath.is_file() or not self.hppPath.is_file():
            raise FileNotFoundError(
                "Both sokobot.cpp and sokobot.hpp must be in " + str(self.solverDir)
            )


        if self.libraryPath.exists():
            sourcesTime = max(self.cppPath.stat().st_mtime, self.hppPath.stat().st_mtime)
            if self.libraryPath.stat().st_mtime >= sourcesTime:
                self.cppLib = self._tryLoad(self.libraryPath)

        if self.cppLib is None:
            try:
                self._compile(self.libraryPath)
                self.cppLib = self._tryLoad(self.libraryPath)
            except (OSError, RuntimeError) as e:
                self.errors.append(f"compile in {self.solverDir}: {e}")
                self.cppLib = None


            if self.cppLib is None:
                fallbackDir = Path(tempfile.mkdtemp(prefix="sokobot_lib_"))
                fallbackPath = fallbackDir / self.libraryPath.name
                try:
                    self._compile(fallbackPath)
                    self.cppLib = self._tryLoad(fallbackPath)
                except (OSError, RuntimeError) as e:
                    self.errors.append(f"compile in {fallbackDir}: {e}")

        if self.cppLib is None:
            raise RuntimeError(
                "Could not build or load the C++ solver library:\n"
                + "\n".join(self.errors)
            )

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
