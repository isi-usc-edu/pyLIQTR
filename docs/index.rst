.. pyLIQTR documentation master file, created by
   sphinx-quickstart on Fri May 31 08:52:40 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyLIQTR's documentation!
===================================

**pyLIQTR** is a MIT Lincoln Laboratory built library for building quantum circuits derived from quantum algorithms and generating Clifford+T resource estimates.


User Documentation
===================================

.. toctree::
   :maxdepth: 1

   Installation <install> 
   Documentation <document>
   Release strategy <release>
   Contributing <contribute>
   Debugger <debugger>
   Best Practices <best_practice>
   .. Example notebooks <notebooks>
   User guides <guides/index>.


API Reference
=============

.. toctree::
   :maxdepth: 2

   api
   api-internal


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   pyLIQTR