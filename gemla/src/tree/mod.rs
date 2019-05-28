use std::fmt;

pub struct Tree<T> {
	pub val: T,
	pub left: Option<Box<Tree<T>>>,
	pub right: Option<Box<Tree<T>>>
}

pub fn combine_trees<T>(v: T, l: Option<Box<Tree<T>>>, r: Option<Box<Tree<T>>>) -> Tree<T> {
	Tree {
		val: v,
		left: l,
		right: r
	}
}

impl<T: fmt::Display> fmt::Display for Tree<T> {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		let node_str = |t: &Option<Box<Tree<T>>>| -> String {
			match t {
				Some(n) => format!("{}", *n),
				_ => String::from("_")
			}
		};

		write!(f, "({} :{}|{})", self.val, node_str(&self.left), node_str(&self.right))
	}
}

pub fn fmt_node<T: fmt::Display>(t: &Option<Box<Tree<T>>>) -> String {
	match t {
		Some(n) => format!("{}", (*n).val),
		_ => String::from("_")
	}
}

struct ParseTreeError {
	msg: String
}