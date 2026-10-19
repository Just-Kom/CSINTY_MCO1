import ctypes
import os
import subprocess
from pathlib import Path


class SokoBot:
    def __init__(self):
        self.solverDir = Path(__file__).resolve().parent
        self.cppPath = self.solverDir / "sokobot.cpp"
        self.dllPath = self.solverDir / "sokobot_cpp.dll"
        self.exePath = self.solverDir / "sokobot_cpp.exe"
        self.cppLib = None
        self._loadSolver()

    def _loadSolver(self):
        if self.dllPath.exists():
            try:
                lib = ctypes.CDLL(str(self.dllPath))
                lib.solve_puzzle_c.argtypes = [
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_char_p,
                    ctypes.c_char_p,
                    ctypes.c_double
                ]
                lib.solve_puzzle_c.restype = ctypes.c_char_p
                self.cppLib = lib
                return
            except OSError:
                self.cppLib = None

        if not self.exePath.exists() and self.cppPath.exists():
            subprocess.run(
                ["g++", "-O3", "-std=c++14", "-static", str(self.cppPath), "-o", str(self.exePath)],
                check=False
            )

    def solveSokobanPuzzle(self, width, height, mapData, itemsData):
        timeLimit = float(os.environ.get("SOKOBOT_TIME_LIMIT", "0.0"))

        if self.cppLib is not None:
            mapFlat = "".join("".join(row[:width]).ljust(width) for row in mapData[:height]).encode("ascii")
            itemsFlat = "".join("".join(row[:width]).ljust(width) for row in itemsData[:height]).encode("ascii")
            result = self.cppLib.solve_puzzle_c(width, height, mapFlat, itemsFlat, timeLimit)
            return result.decode("ascii") if result else ""

        lines = [f"{width} {height} {timeLimit}"]
        for row in mapData[:height]:
            lines.append("".join(row[:width]).ljust(width))
        for row in itemsData[:height]:
            lines.append("".join(row[:width]).ljust(width))
        inputData = "\n".join(lines) + "\n"

        proc = subprocess.run(
            [str(self.exePath)],
            input=inputData,
            capture_output=True,
            text=True
        )
        solution = ""
        for line in proc.stdout.splitlines():
            if line.startswith("SOLUTION:"):
                solution = line[len("SOLUTION:"):].strip()
            else:
                print(line)
        return solution