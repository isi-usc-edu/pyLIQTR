"""
An overload of autoattribute that handles BaseDescriptors more nicely,
giving a good default docstring.

See https://www.sphinx-doc.org/en/master/development/tutorials/autodoc_ext.html
and
https://github.com/sphinx-doc/sphinx/blob/master/doc/development/tutorials/examples/autodoc_intenum.py
for a tutorial.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sphinx.ext.autodoc import AttributeDocumenter
from sphinx.application import Sphinx


if TYPE_CHECKING:
    from docutils.statemachine import StringList
    from sphinx.application import Sphinx


class DescriptorDocumentor(AttributeDocumenter):
    """
    I hoped I would be able to do this by elevated priority rather than fully replacing AttributeDocumenter,
    but it didn't seem to work.
    """

    directivetype = AttributeDocumenter.objtype
    priority = 10 + AttributeDocumenter.priority

    def add_content(
        self,
        more_content: StringList | None,
    ) -> None:
        source_name = self.get_sourcename()
        # if isinstance(self.object, BaseDescriptor):
        #     # if the attribute does not have a docstring, it grabs it from the concrete implementation
        #     # of BaseDescriptor, which we do NOT want here, as it may be big and irrelevant
        #     # It seems like the implementation of AttributeDocumenter.get_doc should deal with this
        #     # but it doesn't seem to
        #     doc_comment = self.get_attribute_comment(self.parent, self.objpath[-1])
        #     if doc_comment is not None:
        #         super().add_content(more_content)
        #         self.add_line("", source_name)
        #         self.add_line("", source_name)
        #     self.add_line(self.object.make_default_docstring(), source_name)
        # else:
        super().add_content(more_content)


def setup(app: Sphinx) -> None:
    app.setup_extension("sphinx.ext.autodoc")  # Require autodoc extension
    app.add_autodocumenter(DescriptorDocumentor, override=True)
