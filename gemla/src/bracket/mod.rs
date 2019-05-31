mod state;

use super::tree;

use uuid::Uuid;

impl tree::Tree<Uuid> {
	pub fn run_simulation(&self) {
		println!("================================");
		println!("Running simulation for node: {}", self.val);
		println!("With children {} and {}", tree::fmt_node(&self.left), tree::fmt_node(&self.right));
	}
}

// pub struct Bracket {
//     tree: tree::Tree,
//     directory: String
// }

/// Constructs a tree with a given height while simultaneously running a simulation on each node.
fn build_tree(h: u32) -> Option<Box<tree::Tree<Uuid>>> {
	let mut result: Option<Box<tree::Tree<Uuid>>> = None;

	// Recursively building a tree and running the simulation after wards to ensure a bottom-up
	// execution order.
	if h != 0 {
		result = Some(Box::new(tree::Tree::new(Uuid::new_v4(), build_tree(h - 1), build_tree(h - 1))));
		match &result {
			Some(r) => (*r).run_simulation(),
			_ => ()
		}
	}

	result
}

/// Generates a bracket tree and runs simulation against each node.
/// 
/// TODO: Explain reasoning for bracket system against genetic algorithm.
pub fn run_bracket() {
	let mut height = 1;
	let mut tree: tree::Tree<Uuid> = *build_tree(height).expect("Error getting result from build_tree.");

	// Building tree one node at a time, appending to the top.
	loop {
		println!("=========================================");
		println!("Running bracket...");
		height += 1;
		tree = tree::Tree::new(Uuid::new_v4(), Some(Box::new(tree)), build_tree(height));
		tree.run_simulation();

		if height == 3 {
			println!("{}\n\n", tree);
			let s = format!("{}", tree);
			println!("{}\n\n", s);
			// let tree2: tree::Tree<Uuid> = FromStr::from_str(&s).expect("");
			// println!("{}\n\n", tree2);
			break;
		}
	}
}