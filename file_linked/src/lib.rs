//! A wrapper around an object that ties it to a physical file

pub mod error;

use anyhow::{anyhow, Context};
use error::Error;
use log::info;
use serde::{de::DeserializeOwned, Serialize};
use std::{
    fs::{copy, remove_file, File},
    io::{ErrorKind, Write},
    path::{Path, PathBuf},
    thread,
    thread::JoinHandle,
};

/// A wrapper around an object `T` that ties the object to a physical file
#[derive(Debug)]
pub struct FileLinked<T>
where
    T: Serialize,
{
    val: T,
    path: PathBuf,
    temp_file_path: PathBuf,
    file_thread: Option<JoinHandle<()>>,
}

impl<T> Drop for FileLinked<T>
where
    T: Serialize,
{
    fn drop(&mut self) {
        if let Some(file_thread) = self.file_thread.take() {
            file_thread
                .join()
                .expect("Error cleaning up file thread for file_linked object");
        }
    }
}

impl<T> FileLinked<T>
where
    T: Serialize,
{
    /// Returns a readonly reference of `T`
    ///
    /// # Examples
    /// ```
    /// # use file_linked::*;
    /// # use serde::{Deserialize, Serialize};
    /// # use std::fmt;
    /// # use std::string::ToString;
    /// # use std::path::PathBuf;
    /// #
    /// # #[derive(Deserialize, Serialize)]
    /// # struct Test {
    /// #     pub a: u32,
    /// #     pub b: String,
    /// #     pub c: f64
    /// # }
    /// #
    /// # fn main() {
    /// let test = Test {
    ///     a: 1,
    ///     b: String::from("two"),
    ///     c: 3.0
    /// };
    ///
    /// let linked_test = FileLinked::new(test, &PathBuf::from("./temp"))
    ///     .expect("Unable to create file linked object");
    ///
    /// assert_eq!(linked_test.readonly().a, 1);
    /// assert_eq!(linked_test.readonly().b, String::from("two"));
    /// assert_eq!(linked_test.readonly().c, 3.0);
    /// #
    /// # drop(linked_test);
    /// #
    /// # std::fs::remove_file("./temp").expect("Unable to remove file");
    /// # }
    /// ```
    pub fn readonly(&self) -> &T {
        &self.val
    }

    /// Creates a new [`FileLinked`] object of type `T` stored to the file given by `path`.
    ///
    /// # Examples
    /// ```
    /// # use file_linked::*;
    /// # use serde::{Deserialize, Serialize};
    /// # use std::fmt;
    /// # use std::string::ToString;
    /// # use std::path::PathBuf;
    /// #
    /// #[derive(Deserialize, Serialize)]
    /// struct Test {
    ///     pub a: u32,
    ///     pub b: String,
    ///     pub c: f64
    /// }
    ///
    /// # fn main() {
    /// let test = Test {
    ///     a: 1,
    ///     b: String::from("two"),
    ///     c: 3.0
    /// };
    ///
    /// let linked_test = FileLinked::new(test, &PathBuf::from("./temp"))
    ///     .expect("Unable to create file linked object");
    ///
    /// assert_eq!(linked_test.readonly().a, 1);
    /// assert_eq!(linked_test.readonly().b, String::from("two"));
    /// assert_eq!(linked_test.readonly().c, 3.0);
    /// #
    /// # drop(linked_test);
    /// #
    /// # std::fs::remove_file("./temp").expect("Unable to remove file");
    /// # }
    /// ```
    pub fn new(val: T, path: &Path) -> Result<FileLinked<T>, Error> {
        let mut temp_file_path = path.to_path_buf();
        temp_file_path.set_file_name(format!(
            ".temp{}",
            path.file_name()
                .ok_or_else(|| anyhow!("Unable to get filename for tempfile {}", path.display()))?
                .to_str()
                .ok_or_else(|| anyhow!("Unable to get filename for tempfile {}", path.display()))?
        ));

        let mut result = FileLinked {
            val,
            path: path.to_path_buf(),
            temp_file_path,
            file_thread: None,
        };

        result.write_data()?;
        Ok(result)
    }

    fn write_data(&mut self) -> Result<(), Error> {
        let thread_path = self.path.clone();
        let thread_temp_path = self.temp_file_path.clone();
        let thread_val = bincode::serialize(&self.val)
            .with_context(|| "Unable to serialize object into bincode".to_string())?;

        if let Some(file_thread) = self.file_thread.take() {
            file_thread
                .join()
                .expect("Error cleaning up file thread for file_linked object");
        }

        match File::open(&self.path) {
            Ok(_) => {
                let handle = thread::spawn(move || {
                    copy(&thread_path, &thread_temp_path).expect("Unable to copy temp file");

                    let mut file = File::create(&thread_path).expect("Error creating file handle");

                    file.write_all(thread_val.as_slice())
                        .expect("Failed to write data to file");

                    remove_file(&thread_temp_path).expect("Error removing temp file");
                });

                self.file_thread = Some(handle);
            }
            Err(error) if error.kind() == ErrorKind::NotFound => {
                let handle = thread::spawn(move || {
                    let mut file = File::create(&thread_path).expect("Error creating file handle");

                    file.write_all(thread_val.as_slice())
                        .expect("Failed to write data to file");
                });

                self.file_thread = Some(handle);
            }
            Err(error) => return Err(Error::IO(error)),
        }

        Ok(())
    }

    /// Modifies the data contained in a `FileLinked` object using a callback `op` that has a mutable reference to the
    /// underlying data. After the mutable operation is performed the data is written to a file to synchronize the state.
    ///
    /// # Examples
    /// ```
    /// # use file_linked::*;
    /// # use file_linked::error::Error;
    /// # use serde::{Deserialize, Serialize};
    /// # use std::fmt;
    /// # use std::string::ToString;
    /// # use std::path::PathBuf;
    /// #
    /// # #[derive(Deserialize, Serialize)]
    /// # struct Test {
    /// #     pub a: u32,
    /// #     pub b: String,
    /// #     pub c: f64
    /// # }
    /// #
    /// # fn main() -> Result<(), Error> {
    /// let test = Test {
    ///     a: 1,
    ///     b: String::from(""),
    ///     c: 0.0
    /// };
    ///
    /// let mut linked_test = FileLinked::new(test, &PathBuf::from("./temp"))
    ///     .expect("Unable to create file linked object");
    ///
    /// assert_eq!(linked_test.readonly().a, 1);
    ///
    /// linked_test.mutate(|t| t.a = 2)?;
    ///
    /// assert_eq!(linked_test.readonly().a, 2);
    /// #
    /// # drop(linked_test);
    /// #
    /// # std::fs::remove_file("./temp").expect("Unable to remove file");
    /// #
    /// # Ok(())
    /// # }
    /// ```
    pub fn mutate<U, F: FnOnce(&mut T) -> U>(&mut self, op: F) -> Result<U, Error> {
        let result = op(&mut self.val);

        self.write_data()?;

        Ok(result)
    }

    /// Replaces the value held by the `FileLinked` object with `val`. After replacing the object will be written to a file.
    ///
    /// # Examples
    /// ```
    /// # use file_linked::*;
    /// # use file_linked::error::Error;
    /// # use serde::{Deserialize, Serialize};
    /// # use std::fmt;
    /// # use std::string::ToString;
    /// # use std::path::PathBuf;
    /// #
    /// # #[derive(Deserialize, Serialize)]
    /// # struct Test {
    /// #     pub a: u32,
    /// #     pub b: String,
    /// #     pub c: f64
    /// # }
    /// #
    /// # fn main() -> Result<(), Error> {
    /// let test = Test {
    ///     a: 1,
    ///     b: String::from(""),
    ///     c: 0.0
    /// };
    ///
    /// let mut linked_test = FileLinked::new(test, &PathBuf::from("./temp"))
    ///     .expect("Unable to create file linked object");
    ///
    /// assert_eq!(linked_test.readonly().a, 1);
    ///
    /// linked_test.replace(Test {
    ///     a: 2,
    ///     b: String::from(""),
    ///     c: 0.0
    /// })?;
    ///
    /// assert_eq!(linked_test.readonly().a, 2);
    /// #
    /// # drop(linked_test);
    /// #
    /// # std::fs::remove_file("./temp").expect("Unable to remove file");
    /// #
    /// # Ok(())
    /// # }
    /// ```
    pub fn replace(&mut self, val: T) -> Result<(), Error> {
        self.val = val;

        self.write_data()
    }
}

impl<T> FileLinked<T>
where
    T: Serialize + DeserializeOwned,
{
    /// Deserializes an object `T` from the file given by `path`
    ///
    /// # Examples
    /// ```
    /// # use file_linked::*;
    /// # use file_linked::error::Error;
    /// # use serde::{Deserialize, Serialize};
    /// # use std::fmt;
    /// # use std::string::ToString;
    /// # use std::fs;
    /// # use std::fs::OpenOptions;
    /// # use std::io::Write;
    /// # use std::path::PathBuf;
    /// #
    /// # #[derive(Deserialize, Serialize)]
    /// # struct Test {
    /// #     pub a: u32,
    /// #     pub b: String,
    /// #     pub c: f64
    /// # }
    /// #
    /// # fn main() -> Result<(), Error> {
    /// let test = Test {
    ///     a: 1,
    ///     b: String::from("2"),
    ///     c: 3.0
    /// };
    ///
    /// let path = PathBuf::from("./temp");
    ///
    /// let mut file = OpenOptions::new()
    ///        .write(true)
    ///        .create(true)
    ///        .open(&path)
    ///        .expect("Unable to create file");
    ///
    /// bincode::serialize_into(file, &test).expect("Unable to serialize object");
    ///
    /// let mut linked_test = FileLinked::<Test>::from_file(&path)
    ///     .expect("Unable to create file linked object");
    ///
    /// assert_eq!(linked_test.readonly().a, test.a);
    /// assert_eq!(linked_test.readonly().b, test.b);
    /// assert_eq!(linked_test.readonly().c, test.c);
    /// #
    /// # drop(linked_test);
    /// #
    /// # std::fs::remove_file("./temp").expect("Unable to remove file");
    /// #
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_file(path: &Path) -> Result<FileLinked<T>, Error> {
        let mut temp_file_path = path.to_path_buf();
        temp_file_path.set_file_name(format!(
            ".temp{}",
            path.file_name()
                .ok_or_else(|| anyhow!("Unable to get filename for tempfile {}", path.display()))?
                .to_str()
                .ok_or_else(|| anyhow!("Unable to get filename for tempfile {}", path.display()))?
        ));

        match File::open(path).map_err(Error::from).and_then(|file| {
            bincode::deserialize_from::<File, T>(file)
                .with_context(|| format!("Unable to deserialize file {}", path.display()))
                .map_err(Error::from)
        }) {
            Ok(val) => Ok(FileLinked {
                val,
                path: path.to_path_buf(),
                temp_file_path,
                file_thread: None,
            }),
            Err(err) => {
                info!(
                    "Unable to read/deserialize file {} attempting to open temp file {}",
                    path.display(),
                    temp_file_path.display()
                );

                // Try to use temp file instead and see if that file exists and is serializable
                let val = FileLinked::from_temp_file(&temp_file_path, path)
                    .map_err(|_| err)
                    .with_context(|| format!("Failed to read/deserialize the object from the file {} and temp file {}", path.display(), temp_file_path.display()))?;

                Ok(FileLinked {
                    val,
                    path: path.to_path_buf(),
                    temp_file_path,
                    file_thread: None,
                })
            }
        }
    }

    fn from_temp_file(temp_file_path: &Path, path: &Path) -> Result<T, Error> {
        let file = File::open(temp_file_path)
            .with_context(|| format!("Unable to open file {}", temp_file_path.display()))?;

        let val = bincode::deserialize_from(file).with_context(|| {
            format!(
                "Could not deserialize from temp file {}",
                temp_file_path.display()
            )
        })?;

        info!("Successfully deserialized value from temp file");

        copy(temp_file_path, path)?;
        remove_file(temp_file_path)
            .with_context(|| format!("Unable to remove temp file {}", temp_file_path.display()))?;

        Ok(val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, fs::File};

    struct CleanUp {
        path: PathBuf,
    }

    impl CleanUp {
        fn new(path: &Path) -> CleanUp {
            CleanUp {
                path: path.to_path_buf(),
            }
        }

        pub fn run<F: FnOnce(&Path) -> Result<(), Error>>(&self, op: F) -> Result<(), Error> {
            op(&self.path)
        }
    }

    impl Drop for CleanUp {
        fn drop(&mut self) {
            if self.path.exists() {
                fs::remove_file(&self.path).expect("Unable to remove file");
            }
        }
    }

    #[test]
    fn test_readonly() -> Result<(), Error> {
        let path = PathBuf::from("test_readonly");
        let cleanup = CleanUp::new(&path);
        cleanup.run(|p| {
            let val = vec!["one", "two", ""];

            let linked_object = FileLinked::new(val.clone(), &p)?;
            assert_eq!(*linked_object.readonly(), val);

            Ok(())
        })
    }

    #[test]
    fn test_new() -> Result<(), Error> {
        let path = PathBuf::from("test_new");
        let cleanup = CleanUp::new(&path);
        cleanup.run(|p| {
            let val = "test";

            FileLinked::new(val, &p)?;

            let file = File::open(&p)?;
            let result: String =
                bincode::deserialize_from(file).expect("Unable to deserialize from file");
            assert_eq!(result, val);

            Ok(())
        })
    }

    #[test]
    fn test_mutate() -> Result<(), Error> {
        let path = PathBuf::from("test_mutate");
        let cleanup = CleanUp::new(&path);
        cleanup.run(|p| {
            let list = vec![1, 2, 3, 4];
            let mut file_linked_list = FileLinked::new(list, &p)?;
            assert_eq!(*file_linked_list.readonly(), vec![1, 2, 3, 4]);

            file_linked_list.mutate(|v1| v1.push(5))?;
            assert_eq!(*file_linked_list.readonly(), vec![1, 2, 3, 4, 5]);

            file_linked_list.mutate(|v1| v1[1] = 1)?;
            assert_eq!(*file_linked_list.readonly(), vec![1, 1, 3, 4, 5]);

            drop(file_linked_list);
            Ok(())
        })
    }

    #[test]
    fn test_replace() -> Result<(), Error> {
        let path = PathBuf::from("test_replace");
        let cleanup = CleanUp::new(&path);
        cleanup.run(|p| {
            let val1 = String::from("val1");
            let val2 = String::from("val2");
            let mut file_linked_list = FileLinked::new(val1.clone(), &p)?;
            assert_eq!(*file_linked_list.readonly(), val1);

            file_linked_list.replace(val2.clone())?;
            assert_eq!(*file_linked_list.readonly(), val2);

            drop(file_linked_list);
            Ok(())
        })
    }

    #[test]
    fn test_from_file() -> Result<(), Error> {
        let path = PathBuf::from("test_from_file");
        let cleanup = CleanUp::new(&path);
        cleanup.run(|p| {
            let value: Vec<f64> = vec![2.0, 3.0, 5.0];
            let file = File::create(&p)?;

            bincode::serialize_into(&file, &value).expect("Unable to serialize into file");
            drop(file);

            let linked_object: FileLinked<Vec<f64>> = FileLinked::from_file(&p)?;
            assert_eq!(*linked_object.readonly(), value);

            drop(linked_object);
            Ok(())
        })
    }
}
