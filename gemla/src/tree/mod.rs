use std::fmt;

pub struct Tree<T> {
	val: T,
	left: Option<Box<Tree<T>>>,
	right: Option<Box<Tree<T>>>
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

fn fmt_node<T: fmt::Display>(t: &Option<Box<Tree<T>>>) -> String {
	match t {
		Some(n) => format!("{}", (*n).val),
		_ => String::from("_")
	}
}

impl<T: fmt::Display> Tree<T> {
	pub fn run_simulation(&self) {
		println!("================================");
		println!("Running simulation for node: {}", self.val);
		println!("With children {} and {}", fmt_node(&self.left), fmt_node(&self.right));
	}
}