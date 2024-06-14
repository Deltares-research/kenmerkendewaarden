# Contributing

## Checkout github repository

- this is just a suggestion, feel free to work with VScode or any other git-compatible workflow
- download git from [git-scm.com](https://git-scm.com/download/win), install with default settings
- open git bash window where you want to clone the github repository (e.g. ``C:\DATA\``)
- ``git clone https://github.com/deltares-research/kenmerkendewaarden`` (creates a local clone of the repository in a folder called kenmerkendewaarden)
- ``cd kenmerkendewaarden``
- optional: ``git config --global user.email [emailaddress]``
- optional: ``git config --global user.name [username]``

## Setup local developer environment

- download and install Anaconda 64 bit from [anaconda.com](https://www.anaconda.com/download/success)
- open anaconda prompt and navigate to the local checkout folder of the repository
- ``conda create --name kw_env python=3.11 git spyder -y`` (``git`` and ``spyder``)
- ``conda activate kw_env``
- ``python -m pip install -e .[dev,docs,examples]`` (pip developer mode, any updates to the local folder are immediately available in your python. It also installs all requirements via pip, square brackets are to install optional dependency groups)
- ``conda deactivate``
- to remove the environment when necessary: ``conda remove -n kw_env --all``

## Contributing

- open an existing issue or create a new issue at https://github.com/Deltares/kenmerkendewaarden/issues
- create a branch via ``Development`` on the right. This branch is now linked to the issue and the issue will be closed once the branch is merged with main again
- open git bash window in the local checkout folder of the repository
- ``git fetch origin`` followed by ``git checkout [branchname]``
- make your local changes to the code (including docstrings and unittests for functions), after each subtask do ``git commit -am 'description of what you did'`` (``-am`` adds all changed files to the commit)
- check if all edits were committed with ``git status``, if there are new files created also do ``git add [path-to-file]`` and commit again
- ``git push`` to push your committed changes your branch on github
- open a pull request at the branch on github, there you can see what you just pushed and the automated checks will show up (testbank and code quality analysis).
- optionally make additional local changes (+commit+push) untill you are done with the issue and the automated checks have passed
- request a review on the pull request
- after review, squash+merge the branch into main

## Running the testbank

- open anaconda prompt and navigate to the local checkout folder of the repository
- ``conda activate kw_env``
- ``pytest`` (runs all tests)
- the pytest testbank also runs automatically on Github for every PR (for different python versions and package versions)

## Generate html documentation
- open anaconda prompt and navigate to the local checkout folder of the repository
- ``conda activate kw_env``
- ``sphinx-build docs docs/_build``
- the documentation is also automatically updated upon every push/merge to the main branch

## Increase the version number

- commit all changes via git
- open anaconda prompt and navigate to the local checkout folder of the repository
- ``conda activate kw_env``
- ``bumpversion major`` or ``bumpversion minor`` or ``bumpversion patch``
- the version number of all relevant files will be updated, as stated in setup.cfg
