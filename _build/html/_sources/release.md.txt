# Releases and development

Releases are a good way to have stable versions of code that are 'frozen in time' so that users can be assured their working version won't break due to development.
We want to have a lightweight notion of this - it won't be as rigorous or complete as a professional software engineering project, but we do want to track progress and make sure folks won't have a borked experiment due to a push to the main branch.


## Development cycle

We use the [Issues](https://github.com/isi-usc-edu/pyLIQTR/issues) page in GitHub to track issues
and assign them pertinent labels and priorities.
GitHub is not as fully featured as Jira when it comes to planning and tracking, but we can make use of things like Projects later to try and work towards concrete goals.


## Branches

There are myriad branching strategies out there, but we've settled on one that should minimally complicated
for users and developers of `pyliqtr`, while also allowing for a solid release process.

* `master`

The `master` branch can be considered the "dev" version of the next release. 
It will be under constant development with potentially breaking changes, so is definitely _not_ stable. 

* `stable/*` branches 

stable/* branches are our stable releases of `pyliqtr`, and are named down to the minor version. 
Note that "stable" is a loaded term, and does not necessarily imply bug-free.
For our purposes, they are meant to capture snapshots in time before any major new functionality
or breaking changes are introduced to the code.
The only changes to these branches should be bugfixes (instructions for that forthcoming),
where the patch digit is incremented and a new tag is added.

* issue branches

These should be made directly from an issue and have the form `type/<issue-#>-short-description`, where `type` can be one of `chore`, `feature`, `bugfix`, etc.
The process is outlined [here](contribute.md#contribution-instructions).
When the branch is merged, it should be deleted to keep the repo clean.

## Release cycle

1. Create a stable branch for the new minor version from the current HEAD on the master branch
2. Create a new tag with the version number on the HEAD of the new stable branch.
3. Change the master version to the next release version, with 'dev' appended.

## Release instructions

With that general philosophy in mind, let's turn to the steps for actually cutting a release.

1. Make two issues - one to do the release prep, another to bump the master version immediately following the release.
2. On the first issue, some tasks to be done:
    1. Make sure tests pass.
    2. Update `CHANGELOG` to reflect what's new/different in the release.
        1. Create a new heading for the changes: `## [X.Y.Z] - YYYY-MM-DD` and make the `## [UNRELEASED]` section empty.
        2. At the end of the `CHANGELOG` create a link to the released version that is the git compare between the new tag and the previous release tag.
        Similarly, update the link for `[UNRELEASED]` to compare `master` and the new tag.
    3. Remove `dev` from the version in `pyliqtr/_version.py`.
    4. Make sure tests pass locally.
    5. Make sure docs build locally.
3. Complete the Pull Request for the first issue.
  This is the last commit before releasing.
4. Make a new branch from master (Maintainers only) called `stable/X.Y`.
    1. The branch only goes to Minor number, but the tag will also have the Patch.
5. From the tags in GitHub, make a new tag `vX.Y.0` created from the new `stable/X.Y` branch.
    1. The tag has full `vX.Y.Z` semantic versioning, and will always start with `Z=0`.
    Bug fixes to this branch will bump the tag.
    2. The tag Comment should read `Release vX.Y.Z (\#<issue number>)`.
    3. Add the `CHANGELOG` contents for this release in the Release Notes of this tag.
6. You now have a real, actual-factual release!
7. Now, from `master`, create a Pull Request for the second issue, to bump the version number for the next release.
    1. In `pyliqtr/_version.py`, bump the version string to the next minor version with dev appended: `0.x+1.0.dev`.
8. Merge that in, and we're now ready to continue using `master` as our development branch.
