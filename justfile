default: lint fmt test

set dotenv-load
export RUST_BACKTRACE := "1"

lint:
  cargo clippy -q

fmt:
  cargo fmt -q

check: lint fmt

test:
  cargo test -q

alias t := test
alias l := lint
alias f := fmt

publish:
  test "$(git rev-parse --abbrev-ref HEAD)" = "master"
  git diff --quiet
  git diff --cached --quiet
  git fetch
  test "$(git rev-parse @{u})" = "$(git rev-parse HEAD)"

  old_version="$(git show HEAD^:Cargo.toml | grep '^version =' | head -n1 | cut -d '"' -f2)"
  new_version="$(grep '^version =' Cargo.toml | head -n1 | cut -d '"' -f2)"
  test "$old_version" != "$new_version"

  latest="$(curl -s https://crates.io/api/v1/crates/my_crate | jq -r '.crate.max_version')"
  test "$new_version" != "$latest"

  cargo publish --dry-run
