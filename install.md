# Installing pyliqtr

This is a guide for installing `pyliqtr` on your machine.
It also lists all the various other applications you may need to install from scratch and get yourself running.

## Environments

Most people use [conda](https://docs.conda.io/en/latest/) for virutal environment management, which is the currently supported configuration for using and developing `pyliqtr`.

## `pyliqtr` installation steps

1. [Create a new environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands)
    for pyliqtr, and activate it:

    ```sh
    on Windows use:
    conda create -n <Environment Name> "python>=3.8,<=3.11.5" pip
    on Mac use:
    conda create -n <Environment Name> python'>=3.8,<=3.11.5' pip
    conda activate <Environment Name>
    ```

    It is important that you install `pip` within this environment, so that later steps that call `pip` will install `pyliqtr` to your new environment and not to the base environment.

    *N.B.* the quotes around the `python` version specification will override the
    pipe functionality of >.

2. Navigate one directory above where you want to clone the repository and clone:

    ```sh
    cd one_dir_above_repo/
    git clone https://github.com/isi-usc-edu/pyLIQTR.git
    ```

3. Navigate to the git repo, and install the package!
    This installs all requirements into the current environment, so it can take a while the first time you install.

    Note that the `-e` flag below installs in `pyliqtr` 'editable' mode.
    This means that as you switch branches, commit changes, or pull from remote, your local version of `pyliqtr` will update with those changes without you needing to reinstall it.

    If you would rather work with a **stable version and not change it**, you could drop the `-e .` and install directly from a stable release or git tag.
    Please don't hesitate to ask for assistance in that scenario!

    **Basic requirements installation:**

    ```sh
    cd repo
    pip install .
    ```

    **Development requirements installation (*only needed if you are going to be doing coding work*):**

    ```sh
    cd repo
    pip install -e .[dev]
    ```

    *N.B.* if using `zsh` instead of `bash` (e.g on a new Mac), the dependencies must be installed via a slightly modified command:

    ```sh
    pip install -e .'[dev]'
    ```

    Optional requirements can be combined with a comma-separated list like 

    ```sh
    pip install -e .[pygpen,dev]
    ```

4. Quick verification that installation was successful: the following command should execute silently.

    ```sh
    python -c "import pyliqtr"
    ```

## Disclaimer

DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Under Secretary of Defense for Research and Engineering.

© 2024 Massachusetts Institute of Technology

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.