from __future__ import annotations
from pathlib import Path
from jupytext import read, write


def copy_notebook_to_py(
    notebook_file_path: Path, output_dir: Path | None = None
) -> Path:
    """
    Convert a .ipynb or .pct.py file to .py in the py:percent format (cells delimited by `# %%`).

    Parameters
    ----------
    notebook_file_path : Path
        Path to source file
    output_dir : Path | None, optional
        Directory to put the.py version, by default None.
        If None, it is placed in the same directory as `notebook_file_path`.

    Returns
    -------
    Path
        Path where the .py file was written.
    """
    if output_dir is None:
        output_dir = notebook_file_path.parent
    # with_suffix("") clears the suffix int he case where it has two suffixes (e.g. .pct.py)
    # since with_suffix only replaces the last one
    output_filename = notebook_file_path.with_suffix("").with_suffix(".py").name
    output_file_path = output_dir / output_filename
    with notebook_file_path.open("r") as fp:
        content = read(fp)
    with output_file_path.open("w"):
        write(content, output_file_path, fmt="py:percent")
    return output_file_path


def main():
    # just a test
    p = Path(__file__).parents[1] / "examples" / "resonator_quality_vs_power.ipynb"
    copy_notebook_to_py(p)


if __name__ == "__main__":
    main()
