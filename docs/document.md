# Documentation

Documentation is incredibly important for collaboration, knowledge sharing, onboarding, continuity, etc.


## Guidelines

Any contributed code should contain docstrings -- we are partial to the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) style -- and a brief example script or jupyter notebook within docs/examples is preferred.
Please reach out to learn how to add any examples to the Sphinx index to that automatic documentation generation can find it. 


## Building the docs

Documentation is hosted in the github and will be periodically regenerated. You can generate the docs locally and view them in a web browser.

To build them:

1. Navigate to the docs folder:

    ```sh
    cd ./docs
    ```

2. Use the make file to clean the docs

    ```sh
    make clean
    ```

    And then build them

    ```sh
    sphinx-apidoc --force -o ./docs/_modules ../src
    make html
    ```

    Because previous builds and temporary files are cleaned up, this can be a little slow.

    **N.B.** - depending on your system, you might get an error that `Pandoc wasn't found` when trying to build Jupyter notebooks.
    If this happens, follow the instructions to [install Pandoc](https://pandoc.org/installing.html) on your machine.

3. Navigate to `docs/_build/html` and open `index.html` in your preferred browser.

These get built in a git-ignored directory, `docs/_build`.
Because the generated doc files are not great for version control (lots of files and diffs to track), we will recommend that you rebuild stable docs locally when you want to view them.
