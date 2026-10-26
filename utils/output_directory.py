"""Select and validate the destination for generated analysis files."""
from pathlib import Path


def select_output_directory(path: str | Path | None = None) -> Path:
    """Use an explicit destination or request it once through the Tk picker."""
    if path is None:
        from tkinter import Tk, filedialog
        window = Tk()
        window.withdraw()
        window.attributes("-topmost", True)
        try:
            selected = filedialog.askdirectory(title="Select the output directory")
        finally:
            window.destroy()
        if not selected:
            raise RuntimeError("Output directory selection was cancelled.")
        path = selected
    destination = Path(path).expanduser().resolve()
    if destination.exists() and not destination.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    return destination
