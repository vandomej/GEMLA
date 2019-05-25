mod bracket;
mod constants;

#[macro_use]
extern crate clap;
use clap::App;
use std::fs::metadata;

fn main() {
	let yaml = load_yaml!("../cli.yml");
	let matches = App::from_yaml(yaml).get_matches();

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