extern crate clap;
extern crate gemla;
#[macro_use]
extern crate log;

mod test_state;
mod fighter_nn;

use easy_parallel::Parallel;
use file_linked::constants::data_format::DataFormat;
use gemla::{
    core::{Gemla, GemlaConfig},
    error::{log_error, Error},
};
use smol::{channel, channel::RecvError, future, Executor};
use std::{path::PathBuf, time::Instant};
use fighter_nn::FighterNN;
use clap::Parser;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    /// The file to read/write the dataset from/to.
    #[arg(short, long)]
    file: String,
}

/// Runs a simluation of a genetic algorithm against a dataset.
///
/// Use the -h, --h, or --help flag to see usage syntax.
/// TODO
fn main() -> anyhow::Result<()> {
    env_logger::init();
    info!("Starting");

    let now = Instant::now();

    // Obtainning number of threads to use
    let num_threads = num_cpus::get().max(1);
    let ex = Executor::new();
    let (signal, shutdown) = channel::unbounded::<()>();

    // Create an executor thread pool.
    let (_, result): (Vec<Result<(), RecvError>>, Result<(), Error>) = Parallel::new()
        .each(0..num_threads, |_| {
            future::block_on(ex.run(shutdown.recv()))
        })
        .finish(|| {
            smol::block_on(async {
                drop(signal);

                // Command line arguments are parsed with the clap crate.
                let args = Args::parse();

                // Checking that the first argument <FILE> is a valid file
                let mut gemla = log_error(Gemla::<FighterNN>::new(
                    &PathBuf::from(args.file),
                    GemlaConfig {
                        generations_per_node: 3,
                        overwrite: true,
                    },
                    DataFormat::Json,
                ))?;

                log_error(gemla.simulate(3).await)?;

                Ok(())
            })
        });

    result?;

    info!("Finished in {:?}", now.elapsed());

    Ok(())
}
