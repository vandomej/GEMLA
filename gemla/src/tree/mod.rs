//! An unbalanced binary tree type where each node has an optional left and right child.
//!
//! # Examples
//!
//! ```
//! use gemla::btree;
//!
//! // Tree with 2 nodes, one root node and one on the left side
//! let mut t = btree!(1, btree!(2),);
//!
//! assert_eq!(t.height(), 2);
//! assert_eq!(t.left.unwrap().val, 2);
//! assert_eq!(t.right, None);
//!
//! t.right = Some(Box::new(btree!(3)));
//! assert_eq!(t.right.unwrap().val, 3);
//! ```

use serde::{Deserialize, Serialize};
use std::cmp::max;

/// An unbalanced binary tree type where each node has an optional left and right child.
///
/// # Examples
///
/// ```
/// use gemla::btree;
///
/// // Tree with 2 nodes, one root node and one on the left side
/// let mut t = btree!(1, btree!(2),);
///
/// assert_eq!(t.height(), 2);
/// assert_eq!(t.left.unwrap().val, 2);
/// assert_eq!(t.right, None);
///
/// t.right = Some(Box::new(btree!(3)));
/// assert_eq!(t.right.unwrap().val, 3);
/// ```
#[derive(Default, Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct Tree<T> {
    pub val: T,
    pub left: Option<Box<Tree<T>>>,
    pub right: Option<Box<Tree<T>>>,
}

/// Short-hand for constructing Trees. `btree!` takes 3 arguments, the first being the
/// value of the root node, and the other two being child nodes. The last two arguments are
/// optional.
///
/// ```
/// use gemla::tree::*;
/// use gemla::btree;
///
/// # fn main() {
/// // A tree with two child nodes.
/// let t = btree!(1, btree!(2), btree!(3));
/// assert_eq!(t,
///     Tree::new(1,
///         Some(Box::new(Tree::new(2, None, None))),
///         Some(Box::new(Tree::new(3, None, None)))));
///
/// // A tree with only a left node.
/// let t_left = btree!(1, btree!(2),);
/// assert_eq!(t_left,
///     Tree::new(1,
///         Some(Box::new(Tree::new(2, None, None))),
///         None));
///
/// // A tree with only a right node.
/// let t_right = btree!(1, , btree!(3));
/// assert_eq!(t_right,
///     Tree::new(1,
///         None,
///         Some(Box::new(Tree::new(3, None, None)))));
///
/// // A tree with no child nodes.
/// let t_single = btree!(1);
/// assert_eq!(t_single,
///     Tree::new(1,
///         None,
///         None));
/// # }
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
    /// Constructs a new [`Tree`] object
    ///
    /// # Examples
    ///
    /// ```
    /// use gemla::tree::*;
    ///
    /// let t = Tree::new(1, None, None);
    /// assert_eq!(t, Tree {
    ///     val: 1,
    ///     left: None,
    ///     right: None
    /// });
    /// ```
    pub fn new(val: T, left: Option<Box<Tree<T>>>, right: Option<Box<Tree<T>>>) -> Tree<T> {
        Tree { val, left, right }
    }

    /// Obtains the height of the longest branch in a [`Tree`]
    ///
    /// # Examples
    ///
    /// ```
    /// use gemla::tree::*;
    /// use gemla::btree;
    ///
    /// let t =
    ///     btree!("a",
    ///         btree!("aa",
    ///             btree!("aaa"),),
    ///         btree!("ab"));
    /// assert_eq!(t.height(), 3);
    /// ```
    pub fn height(&self) -> u64 {
        match (self.left.as_ref(), self.right.as_ref()) {
            (Some(l), Some(r)) => max(l.height(), r.height()) + 1,
            (Some(l), None) => l.height() + 1,
            (None, Some(r)) => r.height() + 1,
            _ => 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        assert_eq!(
            Tree::new(30, None, Some(Box::new(Tree::new(20, None, None)))),
            Tree {
                val: 30,
                left: None,
                right: Some(Box::new(Tree {
                    val: 20,
                    left: None,
                    right: None,
                })),
            }
        );
    }

    #[test]
    fn test_height() {
        assert_eq!(1, btree!(1).height());

        assert_eq!(3, btree!(1, btree!(2), btree!(2, btree!(3),)).height());
    }
}
