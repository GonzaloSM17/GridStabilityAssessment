"""Discovery of a Monte Carlo study and its simulation runs."""

from dataclasses import dataclass, field
from pathlib import Path

from data.run import MonteCarloRun


@dataclass
class MonteCarloContext:
    """Represents one Monte Carlo study directory."""

    context_path: Path
    runs: list[MonteCarloRun] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
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
