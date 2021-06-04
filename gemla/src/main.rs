#[macro_use]
extern crate clap;
extern crate regex;

#[macro_use]
pub mod tree;
pub mod bracket;
pub mod constants;
pub mod file_linked;

#[cfg(test)]
mod tests;

use clap::App;
use std::fs::metadata;

/// Runs a simluation of a genetic algorithm against a dataset.
///
/// Use the -h, --h, or --help flag to see usage syntax.
/// TODO
fn main() {
    // Command line arguments are parsed with the clap crate. And this program uses
    // the yaml method with clap.
    let yaml = load_yaml!("../cli.yml");
    let matches = App::from_yaml(yaml).get_matches();

    // Checking that the first argument <DIRECTORY> is a valid directory
    let directory = matches.value_of(constants::args::DIRECTORY).unwrap();
    let metadata = metadata(directory);
    match &metadata {
        Ok(m) if m.is_dir() => {
            println!("{} is a valid directory!", directory);
            println!("Building tree for {}.", directory);
        }
        Ok(_) => println!("{} is not a valid directory!", directory),
        _ => println!("{} does not exist!", directory),
    }
}
