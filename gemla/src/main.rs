mod bracket;
mod constants;

#[macro_use]
extern crate clap;
use clap::App;
use std::fs::metadata;

/// Runs a simluation of a genetic algorithm against a dataset.
/// 
/// Use the -h, --h, or --help flag to see usage syntax.
fn main() {
	// Command line arguments are parsed with the clap crate. And this program uses
	// the yaml method with clap.
	let yaml = load_yaml!("../cli.yml");
	let matches = App::from_yaml(yaml).get_matches();

	// Checking that the first argument <DIRECTORY> is a valid directory
	let directory = matches.value_of(constants::args::DIRECTORY).unwrap();
	let metadata = metadata(directory);
	match &metadata {
		Ok(m) if m.is_dir() == true => {
			println!("{} is a valid directory!", directory);
			println!("Building tree for {}.", directory);
			bracket::run_bracket();
		},
		Ok(_) => println!("{} is not a valid directory!", directory),
		_ => println!("{} does not exist!", directory)
	}
}