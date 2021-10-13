#[macro_use]
extern crate clap;
extern crate gemla;
#[macro_use]
extern crate log;

mod test_state;

use clap::App;
use gemla::core::{Gemla, GemlaConfig};
use gemla::error::log_error;
use std::path::PathBuf;
use std::time::Instant;
use test_state::TestState;
// use std::io::Write;

/// Runs a simluation of a genetic algorithm against a dataset.
///
/// Use the -h, --h, or --help flag to see usage syntax.
/// TODO
fn main() -> anyhow::Result<()> {
    env_logger::init();
    info!("Starting");

    let now = Instant::now();

    // Command line arguments are parsed with the clap crate. And this program uses
    // the yaml method with clap.
    let yaml = load_yaml!("../../cli.yml");
    let matches = App::from_yaml(yaml).get_matches();

    // Checking that the first argument <DIRECTORY> is a valid directory
    let file_path = matches.value_of(gemla::constants::args::FILE).unwrap();
    let mut gemla = log_error(Gemla::<TestState>::new(
        &PathBuf::from(file_path),
        GemlaConfig {
            generations_per_node: 10,
            overwrite: false,
        },
    ))?;

    log_error(gemla.simulate(10))?;

    // let mut f = std::fs::File::create("./test")?;
    // write!(f, "{}", serde_json::to_string(&gemla.data.readonly().0)?)?;

    info!("Finished in {:?}", now.elapsed());

    Ok(())
}
