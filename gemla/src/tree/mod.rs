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

use std::fmt;
use std::str::FromStr;
use regex::Regex;

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
#[derive(Default, Clone, PartialEq, Debug)]
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
/// let t = btree!(1, btree!(2), btree!(3));
/// 
/// // A tree with only a left node.
/// let t_left = btree!(1, btree!(2),);
/// 
/// // A tree with only a right node.
/// let t_right = btree!(1, ,btree!(3));
/// 
/// // A tree with no children nodes.
/// let t_single = btree!(1)
/// ```
#[macro_export]
macro_rules! btree {
    ($val:expr, $l:expr, $r:expr) => { 
    $crate::tree::Tree::new(
        $val, 
        Some(Box::new($l)), 
        Some(Box::new($r))
    ) };
    ($val:expr, , $r:expr) => { $crate::tree::Tree::new($val, None, Some(Box::new($r))) };
    ($val:expr, $l:expr,) => { $crate::tree::Tree::new($val, Some(Box::new($l)), None) };
    ($val:expr) => { Tree::new($val, None, None) };
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

impl<T: fmt::Display> fmt::Display for Tree<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let node_str = |t: &Option<Box<Tree<T>>>| -> String {
            match t {
                Some(n) => format!("{}", *n),
                _ => String::from("_"),
            }
        };

        write!(
            f,
            "({}: {}|{})",
            self.val,
            node_str(&self.left),
            node_str(&self.right)
        )
    }
}

fn seperate_nodes(s: &str) -> Result<(&str, &str), ParseTreeError> {
    let mut result = Err(ParseTreeError::new(
        format!("Unable to seperate string: {}", s),
    ));
    let mut stack: Vec<char> = Vec::new();

    for (i, c) in s.char_indices() {
        if c == '(' {
            stack.push(c);
        } else if c == ')' {
            if stack.is_empty() {
                result = Err(ParseTreeError::new(
                    format!("Unbalanced parenthesis found in string: {}", s),
                ));
                break;
            }

            stack.pop();
        } else if c == '|' && stack.is_empty() {
            result = Ok((&s[..i], &s[i + 1..]));
            break;
        }
    }

    result
}

fn from_str_helper<T: FromStr>(s: &str) -> Result<Option<Box<Tree<T>>>, ParseTreeError> {
    let mut result = Err(ParseTreeError::new(String::from(
        "Unable to parse tree, string format unrecognized.",
    )));
    let emptyre = Regex::new(r"\s*_\s*").unwrap();
    let re = Regex::new(r"\(([0-9a-fA-F-]+)\s*:\s*(.*)\)$").unwrap();
    let caps = re.captures(s);

    if let Some(c) = caps {
        let val = T::from_str(c.get(1).unwrap().as_str()).or_else(|_| {
            Err(ParseTreeError::new(format!(
                "Unable to parse node value: {}",
                c.get(1).unwrap().as_str()
            )))
        })?;
        let (left, right) = seperate_nodes(c.get(2).unwrap().as_str())?;
        let left = from_str_helper(left)?;
        let right = from_str_helper(right)?;

        result = Ok(Some(Box::new(Tree::new(val, left, right))));
    } else if emptyre.is_match(s) {
        result = Ok(None);
    }

    result
}

impl<T> FromStr for Tree<T>
where
    T: FromStr,
{
    type Err = ParseTreeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let result = from_str_helper(s)?;

        result
            .ok_or_else(|| {
                ParseTreeError::new(format!("Unable to parse string {}", s))
            })
            .and_then(|t| Ok(*t))
    }
}

#[derive(Debug)]
pub struct ParseTreeError {
    pub msg: String,
}

impl ParseTreeError {
    fn new(msg: String) -> ParseTreeError {
        ParseTreeError { msg }
    }
}
