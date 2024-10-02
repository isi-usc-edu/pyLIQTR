# Contributing to `pyliqtr`

This package is maintained by all of us at EQuS!
As the team grows, it is important to keep things tidy and stave off entropy as best we can.
Many of the style choices we make are certainly arbitrary, but we have to enforce *something* so that we can maintain consistency across such a large user and contribution base.

We are excited to have more people contribute to this effort!


## Testing

We aim to have well-structured [unit tests](https://docs.pytest.org/en/latest/)
that provide high [coverage](https://en.wikipedia.org/wiki/Code_coverage) 
of  our code.
Any new contributions should also have tests written for them.

`pytest` should be run to ensure that no tests were broken by the new changes.

Tests live in `tests/` directories within each module/submodule:
```bash
├── pyliqtr
│   ├── __init__.py
│   ├── tests
│   │   ├── __init__.py
│   │   ├── test_acquisition.py
.   .   .
.   .   .
│   ├── future_submodule
│   │   ├── __init__.py
│   │   ├── cool_new_module.py
│   │   ├── tests
│   │   │   ├── __init__.py
│   │   │   ├── test_cool_new_module.py
.   .   .   .
.   .   .   .
```

We have a [Jenkins](https://www.jenkins.io/) server to do automatic testing, but integrating it with GitHub Enterprise is still a work in progress - stay tuned!

## Documentation

See the notes about [documentation](document.md) for more details


## Style

Automatic style-checking infrastructure will be built to ensure clean code.
It is still a work in progress, so we ask that submitted code mostly conforms to [PEP8](https://www.python.org/dev/peps/pep-0008/) (e.g. run [Black](https://github.com/ambv/black) on your code).


## Contribution instructions

1. Create an issue on the [Issues](https://github.com/isi-usc-edu/pyLIQTR/issues) page
    with appropriate labels. 
    Or pick an issue from the existing backlog.
2. Create a branch related to that issue in GitHub, and make your changes locally on that branch.
    The branch should have the form `type/issue#-short-description-of-issue`.
    Unfortunately, GitHub does not have the ability to automatically link branches with issues yet (+1 for GitLab and Bitbucket)
3. Check that your code conforms to our style guidelines, has tests, and is well-documented.
    Resolve any merge conflicts with `master`.
4. When the changes are ready for review, submit a Pull Request and assign
    it to one an admin or someone who would be familiar with the implentation.
    Be mindful of the fact that Code Review is hard, which is one motivation to keep PRs small in scope so that they are easy to review and evaluate!
    Note that Pull Requests can be opened early in your development process if
    you want to start a dialogue as you work through things.


## Code review
