use std::fs;
use std::str::FromStr;
use std::fmt;
use std::string::String;
use std::string::ToString;
use std::io::Read;
use std::io::Write;

pub struct FileLinked<T> {
    val: T,
    path: String,
}

impl<T> FileLinked<T>
where
    T: FromStr + ToString + Default,
{
    pub fn from_file(path: &str) -> Result<FileLinked<T>, String> {
        let meta = fs::metadata(path);

        match &meta {
            Ok(m) if m.is_file() => {
                let mut file = fs::OpenOptions::new().read(true).open(path).or_else(|_| {
                    Err(format!("Unable to open file {}", path))
                })?;
                let mut s = String::new();
                file.read_to_string(&mut s).or_else(|_| {
                    Err(String::from("Unable to read from file."))
                })?;

                let val = T::from_str(&s).or_else(|_| {
                    Err(String::from("Unable to parse value from file."))
                })?;

                Ok(FileLinked {
                    val,
                    path: String::from(path),
                })
            }
            Ok(_) => Err(format!("{} is not a file.", path)),
            _ => {
                let result = FileLinked {
                    val: T::default(),
                    path: String::from(path),
                };

                result.write_data()?;

                Ok(result)
            }
        }
    }

    pub fn new(val: T, path: &str) -> Result<FileLinked<T>, String> {
        let result = FileLinked {
            val,
            path: String::from(path),
        };

        result.write_data()?;

        Ok(result)
    }

    pub fn write_data(&self) -> Result<(), String> {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&self.path)
            .or_else(|_| Err(format!("Unable to open path {}", self.path)))?;

        write!(file, "{}", self.val.to_string()).or_else(|_| {
            Err(String::from("Unable to write to file."))
        })?;

        Ok(())
    }

    pub fn readonly(&self) -> &T {
        &self.val
    }

    pub fn mutate<U, F: FnOnce(&mut T) -> U>(&mut self, op: F) -> Result<U, String> {
        let result = op(&mut self.val);

        self.write_data()?;

        Ok(result)
    }

    pub fn replace(&mut self, val: T) -> Result<(), String> {
        self.val = val;

        self.write_data()
    }
}

impl<T: fmt::Display> fmt::Display for FileLinked<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.val)
    }
}
