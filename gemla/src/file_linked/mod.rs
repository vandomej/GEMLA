use std::fs;
use std::str::FromStr;
use std::fmt::Display;
use std::string::String;
use std::io::Read;
use std::io::Write;

pub struct FileLinked<T> {
	val: T,
	path: String
}

impl<T: FromStr + Display> FileLinked<T> {
	pub fn from_file(path: &str) -> Result<FileLinked<T>, String> {
		let meta = fs::metadata(path)
			.or(Err(format!("Path {} does not exist.", path)))?;
		
		if meta.is_file() {
			let mut file = fs::OpenOptions::new().read(true).open(path)
				.or(Err(format!("Unable to open file {}", path)))?;
			let mut s = String::new();
			file.read_to_string(&mut s)
				.or(Err(String::from("Unable to read from file.")))?;

			let val = T::from_str(&s).or(Err(String::from("Unable to parse value from file.")))?;

			Ok(FileLinked {
				val,
				path: String::from(path)
			})
		} else {
			Err(format!("{} is not a file.", path))
		}
	}

	pub fn new(val: T, path: &str) -> FileLinked<T> {
		let result = FileLinked {
			val,
			path: String::from(path)
		};

		result.write_data();

		result
	}

	pub fn write_data(&self) -> Result<(), String> {
		let mut file = fs::OpenOptions::new()
					.write(true)
					.create_new(true)
					.open(&self.path)
					.or(Err(format!("Unable to open path {}", self.path)))?;

		write!(file, "{}", self.val)
			.or(Err(String::from("Unable to write to file.")))?;

		Ok(())
	}

	pub fn readonly(&self) -> &T {
		&self.val
	}

	pub fn mutate<U, F: FnOnce(&mut T) -> U>(&mut self, op: F) -> U {
		let result = op(&mut self.val);

		self.write_data();

		result
	}
}