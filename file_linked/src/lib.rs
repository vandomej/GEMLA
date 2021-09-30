//! A wrapper around an object that ties it to a physical file

extern crate serde;

use std::fmt;
use std::fs;
use std::io::prelude::*;

use serde::de::DeserializeOwned;
use serde::Serialize;

/// A wrapper around an object `T` that ties the object to a physical file
pub struct FileLinked<T>
where
    T: Serialize,
{
    val: T,
    path: String,
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
    /// let linked_test = FileLinked::new(test, String::from("./temp"))
    ///     .expect("Unable to create file linked object");
    ///
    /// assert_eq!(linked_test.readonly().a, 1);
    /// assert_eq!(linked_test.readonly().b, String::from("two"));
    /// assert_eq!(linked_test.readonly().c, 3.0);
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
    /// let linked_test = FileLinked::new(test, String::from("./temp"))
    ///     .expect("Unable to create file linked object");
    ///
    /// assert_eq!(linked_test.readonly().a, 1);
    /// assert_eq!(linked_test.readonly().b, String::from("two"));
    /// assert_eq!(linked_test.readonly().c, 3.0);
    /// #
    /// # std::fs::remove_file("./temp").expect("Unable to remove file");
    /// # }
    /// ```
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

        write!(file, "{}", serde_json::to_string(&self.val).map_err(|e| e.to_string())?)
            .or_else(|_| Err(String::from("Unable to write to file.")))?;

        Ok(())
    }

    /// Modifies the data contained in a `FileLinked` object using a callback `op` that has a mutable reference to the 
    /// underlying data. After the mutable operation is performed the data is written to a file to synchronize the state.
    /// 
    /// # Examples
    /// ```
    /// # use file_linked::*;
    /// # use serde::{Deserialize, Serialize};
    /// # use std::fmt;
    /// # use std::string::ToString;
    /// #
    /// # #[derive(Deserialize, Serialize)]
    /// # struct Test {
    /// #     pub a: u32,
    /// #     pub b: String,
    /// #     pub c: f64
    /// # }
    /// #
    /// # fn main() -> Result<(), String> {
    /// let test = Test {
    ///     a: 1,
    ///     b: String::from(""),
    ///     c: 0.0
    /// };
    ///
    /// let mut linked_test = FileLinked::new(test, String::from("./temp"))
    ///     .expect("Unable to create file linked object");
    ///
    /// assert_eq!(linked_test.readonly().a, 1);
    /// 
    /// linked_test.mutate(|t| t.a = 2)?;
    /// 
    /// assert_eq!(linked_test.readonly().a, 2);
    /// #
    /// # std::fs::remove_file("./temp").expect("Unable to remove file");
    /// #
    /// # Ok(())
    /// # }
    /// ```
    pub fn mutate<U, F: FnOnce(&mut T) -> U>(&mut self, op: F) -> Result<U, String> {
        let result = op(&mut self.val);

        self.write_data()?;

        Ok(result)
    }

    /// Replaces the value held by the `FileLinked` object with `val`. After replacing the object will be written to a file.
    /// 
    /// # Examples
    /// ```
    /// # use file_linked::*;
    /// # use serde::{Deserialize, Serialize};
    /// # use std::fmt;
    /// # use std::string::ToString;
    /// #
    /// # #[derive(Deserialize, Serialize)]
    /// # struct Test {
    /// #     pub a: u32,
    /// #     pub b: String,
    /// #     pub c: f64
    /// # }
    /// #
    /// # fn main() -> Result<(), String> {
    /// let test = Test {
    ///     a: 1,
    ///     b: String::from(""),
    ///     c: 0.0
    /// };
    ///
    /// let mut linked_test = FileLinked::new(test, String::from("./temp"))
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
    /// # std::fs::remove_file("./temp").expect("Unable to remove file");
    /// #
    /// # Ok(())
    /// # }
    /// ```
    pub fn replace(&mut self, val: T) -> Result<(), String> {
        self.val = val;

        self.write_data()
    }
}

impl<T> FileLinked<T>
where
    T: Serialize + DeserializeOwned + Default,
{
    /// Deserializes an object `T` from the file given by `path`
    /// 
    /// # Examples
    /// ```
    /// # use file_linked::*;
    /// # use serde::{Deserialize, Serialize};
    /// # use std::fmt;
    /// # use std::string::ToString;
    /// # use std::fs;
    /// # use std::fs::OpenOptions;
    /// # use std::io::Write;
    /// #
    /// # #[derive(Deserialize, Serialize, Default)]
    /// # struct Test {
    /// #     pub a: u32,
    /// #     pub b: String,
    /// #     pub c: f64
    /// # }
    /// #
    /// # fn main() -> Result<(), String> {
    /// let test = Test {
    ///     a: 1,
    ///     b: String::from("2"),
    ///     c: 3.0
    /// };
    /// 
    /// let path = String::from("./temp");
    /// 
    /// let mut file = OpenOptions::new()
    ///        .write(true)
    ///        .create(true)
    ///        .open(path.clone())
    ///        .expect("Unable to create file");
    /// 
    /// write!(file, "{}", serde_json::to_string(&test)
    ///     .expect("Unable to serialize object"))
    ///     .expect("Unable to write file");
    /// 
    /// drop(file);
    ///
    /// let mut linked_test = FileLinked::<Test>::from_file(&path)
    ///     .expect("Unable to create file linked object");
    ///
    /// assert_eq!(linked_test.readonly().a, test.a);
    /// assert_eq!(linked_test.readonly().b, test.b);
    /// assert_eq!(linked_test.readonly().c, test.c);
    /// #
    /// # std::fs::remove_file("./temp").expect("Unable to remove file");
    /// #
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_file(path: &str) -> Result<FileLinked<T>, String> {
        let meta = fs::metadata(path);

        match &meta {
            Ok(m) if m.is_file() => {
                let file = fs::OpenOptions::new()
                    .read(true)
                    .open(path)
                    .map_err(|_| format!("Unable to open file {}", path))?;

                let val = serde_json::from_reader(file)
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

impl<T> fmt::Debug for FileLinked<T>
where
    T: fmt::Debug 
        + Serialize,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_mutate() -> Result<(), String> {
        let list = vec![1, 2, 3, 4];
        let mut file_linked_list = FileLinked::new(list, String::from("test.txt"))?;

        assert_eq!(
            format!("{:?}", file_linked_list.readonly()),
            "[1, 2, 3, 4]"
        );

        file_linked_list.mutate(|v1| v1.push(5))?;

        assert_eq!(
            format!("{:?}", file_linked_list.readonly()),
            "[1, 2, 3, 4, 5]"
        );

        file_linked_list.mutate(|v1| {
            v1[1] = 1
        })?;

        assert_eq!(
            format!("{:?}", file_linked_list.readonly()),
            "[1, 1, 3, 4, 5]"
        );

        fs::remove_file("test.txt").expect("Unable to remove file");

        Ok(())
    }
}
