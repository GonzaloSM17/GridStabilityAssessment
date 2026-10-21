"""Remove EMTP .m and .mda source files contained in Monte Carlo runs."""

from collections.abc import Callable
from pathlib import Path
from tkinter import Label, Tk, Toplevel, filedialog, messagebox

SOURCE_SUFFIXES = {".m", ".mda"}
ProgressCallback = Callable[[int, int, str], None]


def find_run_directories(root_path: Path) -> list[Path]:
    """Return run directories, preferring the study's direct runs directory."""
    if root_path.name.startswith("run_"):
        return [root_path]

    runs_path = root_path / "runs"
    if runs_path.is_dir():
        root_path = runs_path

    if root_path.name == "runs":
        return sorted(
            (
                path
                for path in root_path.iterdir()
                if path.is_dir() and path.name.startswith("run_")
            ),
            key=lambda path: path.name,
        )

    return sorted(
        (path for path in root_path.rglob("run_*") if path.is_dir()),
        key=lambda path: path.name,
    )


def find_source_files(
    run_directories: list[Path],
    report_progress: ProgressCallback | None = None,
) -> list[Path]:
    """Return .m and .mda files found inside the supplied run directories."""
    source_files: list[Path] = []
    total_runs = len(run_directories)
    for index, run_directory in enumerate(run_directories, start=1):
        if report_progress is not None:
            report_progress(index, total_runs, "Searching")
        source_files.extend(
            path
            for path in run_directory.rglob("*")
            if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES
        )
    return source_files


def delete_files(
    source_files: list[Path],
    report_progress: ProgressCallback | None = None,
) -> tuple[int, list[tuple[Path, OSError]]]:
    """Delete files and return the number deleted together with any failures."""
    deleted_count = 0
    failures: list[tuple[Path, OSError]] = []
    total_files = len(source_files)
    for index, source_file in enumerate(source_files, start=1):
        if report_progress is not None:
            report_progress(index, total_files, "Deleting")
        try:
            source_file.unlink()
            deleted_count += 1
        except OSError as error:
            failures.append((source_file, error))
    return deleted_count, failures


def create_progress_dialog(window: Tk) -> tuple[Toplevel, Label]:
    """Create a compact status window for long network-directory scans."""
    dialog = Toplevel(window)
    dialog.title("Run source cleanup")
    dialog.transient(window)
    dialog.resizable(False, False)
    dialog.protocol("WM_DELETE_WINDOW", lambda: None)

    label = Label(dialog, padx=24, pady=18, text="Preparing cleanup...")
    label.pack()
    dialog.update_idletasks()
    return dialog, label


def update_progress(
    dialog: Toplevel,
    label: Label,
    action: str,
) -> ProgressCallback:
    """Build a callback that displays the current cleanup progress."""

    def report(current: int, total: int, _: str) -> None:
        label.config(text=f"{action} {current} of {total}...")
        dialog.update_idletasks()

    return report


def main() -> None:
    """Select a study, preview the deletion, and clean its run source files."""
    window = Tk()
    window.withdraw()
    window.attributes("-topmost", True)

    try:
        selected_path = filedialog.askdirectory(
            parent=window,
            title="Select a Monte Carlo study or runs directory",
        )
        if not selected_path:
            return

        root_path = Path(selected_path)
        progress_dialog, progress_label = create_progress_dialog(window)
        progress_label.config(text="Finding simulated runs...")
        progress_dialog.update_idletasks()
        run_directories = find_run_directories(root_path)

        if not run_directories:
            progress_dialog.destroy()
            messagebox.showinfo(
                "No runs found",
                "No directories named run_* were found below the selected directory.",
                parent=window,
            )
            return

        source_files = find_source_files(
            run_directories,
            update_progress(progress_dialog, progress_label, "Searching run"),
        )
        progress_dialog.destroy()

        if not source_files:
            messagebox.showinfo(
                "Nothing to remove",
                "No .m or .mda files were found inside the selected runs.",
                parent=window,
            )
            return

        should_delete = messagebox.askyesno(
            "Confirm cleanup",
            f"Found {len(source_files)} .m/.mda files in {len(run_directories)} runs.\n\n"
            "Delete them permanently?",
            icon="warning",
            parent=window,
        )
        if not should_delete:
            return

        progress_dialog, progress_label = create_progress_dialog(window)
        deleted_count, failures = delete_files(
            source_files,
            update_progress(progress_dialog, progress_label, "Deleting file"),
        )
        progress_dialog.destroy()

        message = f"Deleted {deleted_count} files."
        if failures:
            message += f"\nFailed to delete {len(failures)} files."
        messagebox.showinfo("Cleanup complete", message, parent=window)
    finally:
        window.destroy()


if __name__ == "__main__":
    main()
