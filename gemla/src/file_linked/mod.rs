//! A wrapper around an object that ties it to a physical file

use std::fmt;
use std::fs;
use std::io::Read;
use std::io::Write;
use std::str::FromStr;
use std::string::String;
use std::string::ToString;

/// A wrapper around an object `T` that ties the object to a physical file
pub struct FileLinked<T>
where
    T: ToString,
{
    val: T,
    path: String,
}

impl<T> FileLinked<T>
where
    T: ToString,
{
    /// Returns a readonly reference of `T`
    ///
    /// # Examples
    /// ```
    /// # use gemla::file_linked::*;
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
    /// # impl ToString for Test {
    /// #     fn to_string(&self) -> String {
    /// #         serde_json::to_string(self)
    /// #             .expect("unable to deserialize")
    /// #     }
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
    /// # use gemla::file_linked::*;
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
    /// impl ToString for Test {
    ///     fn to_string(&self) -> String {
    ///         serde_json::to_string(self)
    ///             .expect("unable to deserialize")
    ///     }
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

        write!(file, "{}", self.val.to_string())
            .or_else(|_| Err(String::from("Unable to write to file.")))?;

        Ok(())
    }

    /// Modifies the data contained in a `FileLinked` object using a callback `op` that has a mutable reference to the 
    /// underlying data. After the mutable operation is performed the data is written to a file to synchronize the state.
    /// 
    /// # Examples
    /// ```
    /// # use gemla::file_linked::*;
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
    /// # impl ToString for Test {
    /// #     fn to_string(&self) -> String {
    /// #         serde_json::to_string(self)
    /// #             .expect("unable to deserialize")
    /// #     }
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
    /// # use gemla::file_linked::*;
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
    /// # impl ToString for Test {
    /// #     fn to_string(&self) -> String {
    /// #         serde_json::to_string(self)
    /// #             .expect("unable to deserialize")
    /// #     }
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

impl<T: fmt::Display> fmt::Display for FileLinked<T>
where
    T: ToString,
{
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
