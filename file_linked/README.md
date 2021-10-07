# File Linked - controlling objects linked directly to a file

This library provides a wrapper around objects and ties the data to a file. It uses serde and bincode currently for serializing and deserializing the files.

## Examples
```rust
use file_linked::*;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::string::ToString;
use std::path::PathBuf;

#[derive(Deserialize, Serialize)]
struct Test {
    pub a: u32,
    pub b: String,
    pub c: f64
}

let test = Test {
    a: 1,
    b: String::from("two"),
    c: 3.0
};

let file_path = PathBuf::from("./file");

// Object is consumed and can only be interacted with through the FileLinked object
let mut linked_test = FileLinked::new(test, &file_path)?;

// You can obtain a readonly reference of the underlying data
assert_eq!(linked_test.readonly().b, String::from("two"));

// Whenever a mutable operation is performed, the changed data is rewritten to the file
linked_test.mutate(|x| x.a += 1)?;
assert_eq!(linked_test.readonly().a, 2);

drop(linked_test);

// You can also initialize an object from a file
let from_file = FileLinked::<Test>::from_file(&file_path)?;

assert_eq!(from_file.readonly().a, 2);
assert_eq!(from_file.readonly().b, String::from("two"));
assert_eq!(from_file.readonly().c, 3.0);
```

This library is still in development and missing some features and so may not be stable:
- Currently after any mutable operations the FileLinked object will rewrite the entire file
- Custom selection of serializers is not implemented, the serializer used is just bincode as of now