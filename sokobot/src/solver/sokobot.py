import subprocess
import sys
from pathlib import Path


class SokoBot:
    def __init__(self):
        self.solverDir = Path(__file__).resolve().parent
        self.cppPath = self.solverDir / "sokobot.cpp"

        if sys.platform == "win32":
            self.exePath = self.solverDir / "sokobot_cpp.exe"
        elif sys.platform.startswith("linux") or sys.platform == "darwin":
            self.exePath = self.solverDir / "sokobot_cpp"
        else:
            raise RuntimeError("Unsupported operating system")

        self._ensureCompiled()

    def _ensureCompiled(self):
        if not self.cppPath.exists():
            raise FileNotFoundError(f"C++ source not found: {self.cppPath}")

        # Recompile if the binary is missing or older than the source files
        sources = [self.cppPath, self.solverDir / "sokobot.hpp"]
        newest = max(p.stat().st_mtime for p in sources if p.exists())
        if self.exePath.is_file() and self.exePath.stat().st_mtime >= newest:
            return

        command = [
            "g++", "-O3", "-std=c++20",
            str(self.cppPath),
            "-o", str(self.exePath),
        ]
        print(f"Compiling C++ for {sys.platform}...")
        subprocess.run(command, cwd=self.solverDir, check=True)

    def solveSokobanPuzzle(self, width, height, mapData, itemsData):
        mapLines = ["".join(row) for row in mapData]
        itemLines = ["".join(row) for row in itemsData]
        puzzle = "\n".join([f"{width} {height}"] + mapLines + itemLines) + "\n"

        result = subprocess.run(
            [str(self.exePath)],
            input=puzzle,
            text=True,
            capture_output=True,
            cwd=self.solverDir,
            check=True,
        )

        for line in result.stdout.splitlines():
            if line.startswith("SOLUTION:"):
                return line.split("SOLUTION:", 1)[1].strip()

        raise RuntimeError("C++ solver did not return a solution")