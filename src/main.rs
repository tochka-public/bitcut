#![cfg(feature = "cli")]

use std::{
    fs,
    io::{stdout, Write},
    path::PathBuf,
};

use bitcut::{apply_patch, make_patch, Op};
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "bitcut")]
#[command(about = "Create and apply binary patches", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Create a binary patch from two files and write it to stdout
    Diff { old: PathBuf, new: PathBuf },
    /// Apply a binary patch to a file and write the result to stdout
    Patch { old: PathBuf, patch: PathBuf },
    /// Print patch opcodes
    Debug { patch: PathBuf },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Diff { old, new } => {
            let old = fs::read(old)?;
            let new = fs::read(new)?;
            let patch = make_patch(&old, &new)?;
            stdout().write_all(&patch)?;
        }
        Commands::Patch { old, patch } => {
            let old = fs::read(old)?;
            let patch = fs::read(patch)?;
            let new = apply_patch(&old, &patch)?;
            stdout().write_all(&new)?;
        }
        Commands::Debug { patch } => {
            let patch = fs::read(patch)?;
            let ops: Vec<_> = Op::iter(&patch).collect::<Result<_, _>>()?;
            println!("{ops:#?}");
        }
    }

    Ok(())
}
