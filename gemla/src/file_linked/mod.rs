use std::fmt;
use std::fs;
use std::io::Read;
use std::io::Write;
use std::str::FromStr;
use std::string::String;
use std::string::ToString;

pub struct FileLinked<T> {
    val: T,
    path: String,
}

impl<T> FileLinked<T> {
    pub fn readonly(&self) -> &T {
        &self.val
    }
}

impl<T> FileLinked<T>
where
    T: ToString,
{
    pub fn new(val: T, path: String) -> Result<FileLinked<T>, String> {
        let result = FileLinked { val, path };

        result.write_data()?;

        Ok(result)
    }

    fn write_data(&self) -> Result<(), String> {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&self.path)
            .map_err(|_| format!("Unable to open path {}", self.path))?;

        write!(file, "{}", self.val.to_string())
            .or_else(|_| Err(String::from("Unable to write to file.")))?;

        Ok(())
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

impl<T> FileLinked<T>
where
    T: ToString + FromStr + Default,
{
    pub fn from_file(path: &str) -> Result<FileLinked<T>, String> {
        let meta = fs::metadata(path);

        match &meta {
            Ok(m) if m.is_file() => {
                let mut file = fs::OpenOptions::new()
                    .read(true)
                    .open(path)
                    .map_err(|_| format!("Unable to open file {}", path))?;
                let mut s = String::new();
                file.read_to_string(&mut s)
                    .map_err(|_| String::from("Unable to read from file."))?;

                let val = T::from_str(&s)
                    .map_err(|_| String::from("Unable to parse value from file."))?;

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
}

impl<T: fmt::Display> fmt::Display for FileLinked<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_mutate() -> Result<(), String> {
        let tree = btree!(1, btree!(2), btree!(3, btree!(4),));
        let mut linked_tree = FileLinked::new(tree, String::from("test.txt"))?;

        assert_eq!(
            format!("{}", linked_tree.readonly()),
            "{\"val\":1,\"left\":{\"val\":2,\"left\":null,\"right\":null},\"right\":{\"val\":3,\"left\":{\"val\":4,\"left\":null,\"right\":null},\"right\":null}}"
        );

        linked_tree.mutate(|v1| v1.val = 10)?;

        assert_eq!(
            format!("{}", linked_tree.readonly()),
            "{\"val\":10,\"left\":{\"val\":2,\"left\":null,\"right\":null},\"right\":{\"val\":3,\"left\":{\"val\":4,\"left\":null,\"right\":null},\"right\":null}}"
        );

        linked_tree.mutate(|v1| {
            let mut left = v1.left.clone().unwrap();
            left.val = 13;
            v1.left = Some(left);
        })?;

        assert_eq!(
            format!("{}", linked_tree.readonly()),
            "{\"val\":10,\"left\":{\"val\":13,\"left\":null,\"right\":null},\"right\":{\"val\":3,\"left\":{\"val\":4,\"left\":null,\"right\":null},\"right\":null}}"
        );

        fs::remove_file("test.txt").expect("Unable to remove file");

        Ok(())
    }
}
