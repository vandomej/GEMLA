//! An unbalanced binary tree type where each node has an optional left and right child.
//!
//! # Examples
//!
//! ```
//! let mut t = Tree::new(1, None, None);
//! let t2 = Tree::new(2, Some(Box::new(t)), Some(Box::new(Tree::new(3, None, None))));
//! let s = format!("{}", t2);
//!
//! assert_eq!(s, "(2: (1: _|_)|(3: _|_))");
//! t.left = Some(Box::new(Tree::new(4, None, None)));
//! assert_eq!(Tree::fmt_node(t.left), 4);
//! assert_eq!(Tree::from_str(s), t2);
//! ```
//!
//! Additionally the `btree!` macro can be used to conveniently initialize trees:
//!
//! ```
//! # #[macro_use] extern crate tree;
//! # fn main() {
//! let t1 = btree!(1,btree!(2),btree!(3))
//! assert_eq!(format!("{}", t1), "(1: (2: _|_)|(3: _|_)")
//! # }
//! ```

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// An unbalanced binary tree type where each node has an optional left and right child.
///
/// # Examples
///
/// ```
/// let mut t = Tree::new(1, None, None);
/// let t2 = Tree::new(2, Some(Box::new(t)), Some(Box::new(Tree::new(3, None, None))));
/// let s = format!("{}", t2);
///
/// assert_eq!(s, "(2: (1: _|_)|(3: _|_))");
/// t.left = Some(Box::new(Tree::new(4, None, None)));
/// assert_eq!(Tree::fmt_node(t.left), 4);
/// assert_eq!(Tree::from_str(s), t2);
/// ```
///
/// Additionally the `btree!` macro can be used to conveniently initialize trees:
///
/// ```
/// let t1 = btree!(1,btree!(2),btree!(3))
/// assert_eq!(format!("{}", t1), "(1: (2: _|_)|(3: _|_)")
/// ```
#[derive(Default, Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct Tree<T> {
    pub val: T,
    pub left: Option<Box<Tree<T>>>,
    pub right: Option<Box<Tree<T>>>,
}

/// Used to construct trees in a cleaner manner. `btree!` takes 3 arguments, the first being the
/// value of the root node, and the other two being child nodes. The last two arguments are
/// optional.
///
/// ```
/// // A tree with two child nodes.
/// let t = btree!(1, Some(btree!(2)), Some(btree!(3)));
///
/// // A tree with only a left node.
/// let t_left = btree!(1, Some(btree!(2)),);
///
/// // A tree with only a right node.
/// let t_right = btree!(1, ,Some(btree!(3)));
///
/// // A tree with no children nodes.
/// let t_single = btree!(1)
/// ```
#[macro_export]
macro_rules! btree {
    ($val:expr, $l:expr, $r:expr) => {
        $crate::tree::Tree::new($val, Some(Box::new($l)), Some(Box::new($r)))
    };
    ($val:expr, , $r:expr) => {
        $crate::tree::Tree::new($val, None, Some(Box::new($r)))
    };
    ($val:expr, $l:expr,) => {
        $crate::tree::Tree::new($val, Some(Box::new($l)), None)
    };
    ($val:expr) => {
        $crate::tree::Tree::new($val, None, None)
    };
}

impl<T> Tree<T> {
    /// Constructs a new tree object.
    pub fn new(val: T, left: Option<Box<Tree<T>>>, right: Option<Box<Tree<T>>>) -> Tree<T> {
        Tree { val, left, right }
    }

    pub fn fmt_node(t: &Option<Box<Tree<T>>>) -> String
    where
        T: fmt::Display,
    {
        match t {
            Some(n) => format!("{}", (*n).val),
            _ => String::from("_"),
        }
    }
}

impl<T: fmt::Display + Serialize> fmt::Display for Tree<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let result = serde_json::to_string(self);

        match result {
            Ok(string) => write!(f, "{}", string),
            Err(_) => Err(std::fmt::Error),
        }
    }
}

impl<T> FromStr for Tree<T>
where
    T: FromStr + DeserializeOwned,
{
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        serde_json::from_str(s).map_err(|_| format!("Unable to parse string {}", s))
    }
}
