import ctypes
import subprocess
import sys
from pathlib import Path


class SokoBot:
    def __init__(self):
        self.solverDir = Path(__file__).resolve().parent
        self.cppPath = self.solverDir / "sokobot.cpp"

        self.isWindows = sys.platform == "win32"
        self.isLinux = sys.platform.startswith("linux")

        if self.isWindows:
            self.libraryPath = self.solverDir / "sokobot_cpp.exe"
        elif self.isLinux:
            self.libraryPath = self.solverDir / "sokobot_cpp.so"
        else:
            raise RuntimeError("Unsupported operating system")

        self.cppLib = None
        self._loadSolver()

    def _compileSolver(self):
        if not self.cppPath.exists():
            raise FileNotFoundError(
                f"C++ source not found: {self.cppPath}"
            )

        if self.isWindows:
            command = [
                "g++",
                "-O3",
                "-std=c++20",
                str(self.cppPath),
                "-o",
                str(self.libraryPath)
            ]
        else:
            command = [
                "g++",
                "-O3",
                "-std=c++20",
                "-shared",
                "-fPIC",
                str(self.cppPath),
                "-o",
                str(self.libraryPath)
            ]

        print(f"Compiling C++ for {sys.platform}...")
        subprocess.run(
            command,
            cwd=self.solverDir,
            check=True
        )

    def _loadSolver(self):
        if not self.libraryPath.is_file():
            self._compileSolver()

        if self.isLinux:
            self.cppLib = ctypes.CDLL(str(self.libraryPath))
            self.cppLib.solve_puzzle_c.argtypes = [
                ctypes.c_char_p,
                ctypes.POINTER(ctypes.c_char),
                ctypes.c_size_t
            ]
            self.cppLib.solve_puzzle_c.restype = ctypes.c_int

    def solveSokobanPuzzle(self, width, height, mapData, itemsData):
        mapLines = ["".join(row) for row in mapData]
        itemLines = ["".join(row) for row in itemsData]
        puzzle = "\n".join(
            [f"{width} {height}"] + mapLines + itemLines
        ) + "\n"

        if self.isWindows:
            result = subprocess.run(
                [str(self.libraryPath)],
                input=puzzle,
                text=True,
                capture_output=True,
                cwd=self.solverDir,
                check=True
            )

            for line in result.stdout.splitlines():
                if line.startswith("SOLUTION:"):
                    return line.split("SOLUTION:", 1)[1].strip()

            raise RuntimeError("C++ solver did not return a solution")

        else:
            raise NotImplementedError(
                "Linux requires a C-compatible function exported from the .so"
            )