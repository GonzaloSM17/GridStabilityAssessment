"""Discovery of a Monte Carlo study and its simulation runs."""

from dataclasses import dataclass, field
from pathlib import Path

from run import MonteCarloRun


@dataclass
class MonteCarloContext:
    """Represents one Monte Carlo study directory."""

    context_path: Path | None = None
    runs: list[MonteCarloRun] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if self.context_path is None:
            self.select_directory()
        else:
            self.set_context_path(self.context_path)

    def set_context_path(self, context_path: str | Path) -> Path:
        """Validate a study directory and discover its published runs."""
        path = Path(context_path).expanduser().resolve()

        if not path.is_dir():
            raise NotADirectoryError(
                f"The context path must be an existing directory: {path}"
            )

        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Study manifest not found: {manifest_path}")

        runs_path = path / "runs"
        if not runs_path.is_dir():
            raise NotADirectoryError(f"Study runs directory not found: {runs_path}")

        self.context_path = path
        self.runs = [
            MonteCarloRun.from_directory(run_path)
            for run_path in sorted(
                (
                    child
                    for child in runs_path.iterdir()
                    if child.is_dir() and child.name.startswith("run_")
                ),
                key=lambda child: child.name,
            )
        ]
        return path

    def select_directory(self) -> Path | None:
        """Open a Tk directory picker and use the selected study directory."""
        selected_path = self.choose_directory()
        if selected_path is None:
            return None

        return self.set_context_path(selected_path)

    @staticmethod
    def choose_directory() -> Path | None:
        """Open a Tk directory picker without constructing a study context."""
        from tkinter import Tk, filedialog

        window = Tk()
        window.withdraw()
        window.attributes("-topmost", True)

        try:
            selected_path = filedialog.askdirectory(
                title="Select the Monte Carlo study directory"
            )
        finally:
            window.destroy()

        if not selected_path:
            return None

        return Path(selected_path)


if __name__ == "__main__":

    context = MonteCarloContext()

    if context.context_path is None:
        print("No Monte Carlo study was selected.")
    else:
        print(f"Context: {context.context_path}")
        print(f"Runs: {len(context.runs)}")
