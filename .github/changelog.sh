#!/usr/bin/env bash
# Print the CHANGELOG section for one version, without its heading.
#
# Used twice by the release workflow: once as a gate (empty output means the
# release was cut without writing down what changed) and once to fill in the
# GitHub release body.
#
#   ./.github/changelog.sh 2.0.0

set -euo pipefail

version="${1:?usage: changelog.sh VERSION}"
changelog="${2:-CHANGELOG.md}"

# Blank lines inside the section are kept, the ones padding either end are not,
# so an empty section prints nothing at all and the gate can test for that.
awk -v version="$version" '
  index($0, "## [" version "]") == 1 { inside = 1; next }
  inside && index($0, "## [") == 1 { exit }
  # The link definitions at the foot of the file belong to no section.
  inside && /^\[[^]]+\]:/ { exit }
  inside && NF == 0 { pending++; next }
  inside {
    for (; started && pending > 0; pending--) print ""
    pending = 0
    started = 1
    print
  }
' "$changelog"
