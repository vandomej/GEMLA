use std::fmt;
use uuid::Uuid;

pub struct Tree {
	id: Uuid,
	left: Option<Box<Tree>>,
	right: Option<Box<Tree>>
}

pub fn combine_trees(id: Uuid, l: Option<Box<Tree>>, r: Option<Box<Tree>>) -> Tree {
	Tree {
		id: id,
		left: l,
		right: r
	}
}

impl fmt::Display for Tree {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		let node_str = |t: &Option<Box<Tree>>| -> String {
			match t {
				Some(n) => format!("{}", *n),
				_ => String::from("_")
			}
		};

		write!(f, "({} :{}|{})", self.id, node_str(&self.left), node_str(&self.right))
	}
}

fn fmt_node(t: &Option<Box<Tree>>) -> String {
	match t {
		Some(n) => format!("{}", (*n).id),
		_ => String::from("_")
	}
}

impl Tree {
	pub fn run_simulation(&self) {
		println!("================================");
		println!("Running simulation for node: {}", self.id);
		println!("With children {} and {}", fmt_node(&self.left), fmt_node(&self.right));
	}
}