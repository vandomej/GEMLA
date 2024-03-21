extern crate clap;
extern crate gemla;
#[macro_use]
extern crate log;

mod test_state;
mod fighter_nn;

use file_linked::constants::data_format::DataFormat;
use gemla::{
    core::{Gemla, GemlaConfig},
    error::log_error,
};
use std::{path::PathBuf, time::Instant};
use fighter_nn::FighterNN;
use clap::Parser;
use anyhow::Result;

// const NUM_THREADS: usize = 12;

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
fn main() -> Result<()> {
    env_logger::init();
    info!("Starting");
    let now = Instant::now();

    // Manually configure the Tokio runtime
    let runtime: Result<()> = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(num_cpus::get()) 
        // .worker_threads(NUM_THREADS) 
        .build()?
        .block_on(async {
            let args = Args::parse(); // Assuming Args::parse() doesn't need to be async
            let mut gemla = log_error(Gemla::<FighterNN>::new(
                &PathBuf::from(args.file),
                GemlaConfig {
                    generations_per_height: 10,
                    overwrite: false,
                },
                DataFormat::Json,
            ))?;

            // let gemla_arc = Arc::new(gemla);

            // Setup your application logic here
            // If `gemla::simulate` needs to run sequentially, simply call it in sequence without spawning new tasks

            // Example placeholder loop to continuously run simulate
            loop { // Arbitrary loop count for demonstration
                gemla.simulate(5).await?;
            }
        });

    runtime?; // Handle errors from the block_on call

    info!("Finished in {:?}", now.elapsed());
    Ok(())
}