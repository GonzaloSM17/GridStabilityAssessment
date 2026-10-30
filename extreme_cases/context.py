"""Validated source context for a collection of deterministic extreme cases."""

from dataclasses import dataclass
from pathlib import Path

from extreme_cases.models import EXTREME_CASE_DEFINITIONS, ExtremeCaseDefinition
from extreme_cases.net_catalog import NetSignalCatalog
from extreme_cases.npy_signals import NpySignalMatrix


@dataclass(frozen=True)
class ExtremeCase:
    """One directory containing one NET model and its raw signal result."""

    definition: ExtremeCaseDefinition
    directory: Path
    net_catalog: NetSignalCatalog
    signals: NpySignalMatrix
    net_variations_path: Path


class ExtremeCasesContext:
    """Open and validate the three deterministic extreme-controller cases."""

    def __init__(
        self,
        root_path: str | Path | None = None,
        definitions: tuple[ExtremeCaseDefinition, ...] = EXTREME_CASE_DEFINITIONS,
    ) -> None:
        self.root_path = self._resolve_root_path(root_path)
        self.definitions = definitions
        self.cases = tuple(self._open_case(definition) for definition in definitions)
        self._validate_common_layout()

    @staticmethod
    def _resolve_root_path(root_path: str | Path | None) -> Path:
        if root_path is None:
            from tkinter import Tk, filedialog

            dialog = Tk()
            dialog.withdraw()
            dialog.attributes("-topmost", True)
            selected = filedialog.askdirectory(title="Select extreme cases directory")
            dialog.destroy()
            if not selected:
                raise ValueError("No extreme-cases directory was selected.")
            root_path = selected
        path = Path(root_path).expanduser().resolve()
        if not path.is_dir():
            raise NotADirectoryError(f"Extreme-cases directory not found: {path}")
        return path

    def _open_case(self, definition: ExtremeCaseDefinition) -> ExtremeCase:
        directory = self.root_path / definition.directory_name
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Extreme case '{definition.label}' not found: {directory}"
            )
        net_files = tuple(directory.glob("*.net"))
        if len(net_files) != 1:
            raise ValueError(
                f"Extreme case '{definition.label}' must contain exactly one NET file."
            )
        variations = directory / "net_variations.npy"
        if not variations.is_file():
            raise FileNotFoundError(f"NET variations not found: {variations}")
        return ExtremeCase(
            definition=definition,
            directory=directory,
            net_catalog=NetSignalCatalog.from_net(net_files[0]),
            signals=NpySignalMatrix.from_file(directory / "signals.npy"),
            net_variations_path=variations,
        )

    def _validate_common_layout(self) -> None:
        reference = self.cases[0]
        for case in self.cases[1:]:
            if case.net_catalog.signature != reference.net_catalog.signature:
                raise ValueError(
                    f"NET meter catalog differs between '{reference.definition.label}' "
                    f"and '{case.definition.label}'."
                )
            if case.signals.schema_signature != reference.signals.schema_signature:
                raise ValueError(
                    f"Signals NPY layout differs between '{reference.definition.label}' "
                    f"and '{case.definition.label}'."
                )

    @property
    def reference_case(self) -> ExtremeCase:
        return self.cases[0]

    @property
    def expected_signal_names(self) -> tuple[str, ...]:
        return self.reference_case.net_catalog.expected_signal_names

