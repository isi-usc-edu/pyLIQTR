# This script publishes the build docs to gh-pages
# Run this from inside the ./docs directory
# Note: it takes a few minutes for https://github.com/isi-usc-edu/pyLIQTR/quantum_algorithms/
# to update after running this script

git init
git config --local user.email ''
git config --local user.name ''
git config --local --add safe.directory '*'
git add .
git commit -m "Deploy docs to GitHub Pages"
git remote add origin https://github.com/isi-usc-edu/pyLIQTR/quantum_algorithms.git
git push --force --set-upstream origin master:gh-pages
