from __future__ import annotations
from typing import Any
from pathlib import Path

from docutils import nodes
from sphinx.addnodes import download_reference
from sphinx.util.docutils import ReferenceRole
from sphinx.application import Sphinx

from notebook_to_py import copy_notebook_to_py


class NbAsPyDownloadRole(ReferenceRole):
    EXPORT_DIR = "notebook_conversions"
    """
    Role to convert a .ipynb notebook to a .py file in the "py:percent" notebook format
    and provide a download link.

    This custom Role API isn't terribly well documented.
    See `ReferenceRole` and `SphinxRole` classes in 
    https://github.com/sphinx-doc/sphinx/blob/master/sphinx/util/docutils.py
    on what this class is doing. It's a wrapper for the docutils role function API:
    https://docutils.sourceforge.io/docs/howto/rst-roles.html

    MYST_NB does something similar for their `nb-download` role:
    https://github.com/executablebooks/MyST-NB/blob/master/myst_nb/ext/download.py
    """

    def run(self):
        """
        Converts the notebook into a .py file in the build directory, and links.
        """
        notebook_path = Path(self.get_source_info()[0])
        build_dir = Path(
            self.env.doctreedir
        ).parent  # is there a clearer way to get this?
        output_dir = build_dir / self.EXPORT_DIR
        output_dir.mkdir(exist_ok=True)
        py_path = copy_notebook_to_py(notebook_path, output_dir=output_dir)
        # retarget the download link
        new_title = py_path.name
        reftarget = str(py_path)
        node = download_reference(self.rawtext, reftarget=reftarget)
        title = self.title if self.has_explicit_title else new_title
        node += nodes.literal(self.rawtext, title, classes=["xref", "download"])

        return [node], []


def setup(app: Sphinx) -> dict[str, Any]:
    role_name = "nb-download-as-py"
    app.add_role(role_name, NbAsPyDownloadRole())

    return {
        "version": "0.0.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
