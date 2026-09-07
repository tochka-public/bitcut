# Changelog

Written by release-plz from the commit messages, so a commit subject is what a
reader of this file ends up with. A `feat!:` prefix or a `BREAKING CHANGE:`
trailer is what makes a release major.

Releases up to `v0.1.6` predate this file. They are on
[crates.io](https://crates.io/crates/bitcut/versions) and in the git history.

## [1.0.1](https://github.com/tochka-public/bitcut/compare/v1.0.0...v1.0.1) - 2026-09-07

### Documentation

- medians instead of best-of-five, and why escalation beats indexing first
- bring the README back in line with what shipped

## [1.0.0](https://github.com/tochka-public/bitcut/compare/v0.1.6...v1.0.0) - 2026-09-06

### Added

- [**breaking**] tiered resync differ and patch format v2

### Changed

- migrate unit tests to rstest + insta snapshots

### Documentation

- run the walkthrough on the crate itself, compiled to wasm
- update README.me

### Fixed

- *(test)* the corpus schema seed overflowed in debug builds

### Other

- *(deps)* bump rand from 0.9.2 to 0.9.4
