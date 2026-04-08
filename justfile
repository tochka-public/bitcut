default: check test

set dotenv-load
export RUST_BACKTRACE := "1"

# apply formatting
fmt:
  cargo fmt --all

# read-only format check
fmt-check:
  cargo fmt --all -- --check

clippy:
  cargo clippy --all-targets --all-features -- -D warnings
  cargo clippy --all-targets --no-default-features -- -D warnings

check: fmt-check clippy

test:
  cargo test --all-features
  cargo test --no-default-features

# reproduce CI locally
ci: check test

alias t := test
alias f := fmt
alias c := check
