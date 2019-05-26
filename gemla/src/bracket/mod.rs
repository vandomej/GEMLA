mod tree;
mod state;

use uuid::Uuid;

// pub struct Bracket {
//     tree: tree::Tree,
//     directory: String
// }

/// Constructs a tree with a given height while simultaneously running a simulation on each node.
fn build_tree(h: u32) -> Option<Box<tree::Tree>> {
	let mut result: Option<Box<tree::Tree>> = None;

	// Recursively building a tree and running the simulation after wards to ensure a bottom-up
	// execution order.
	if h != 0 {
		result = Some(Box::new(tree::combine_trees(Uuid::new_v4(), build_tree(h - 1), build_tree(h - 1))));
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
	let mut tree: tree::Tree = *build_tree(height).expect("Error getting result from build_tree.");

	// Building tree one node at a time, appending to the top.
	loop {
		println!("=========================================");
		println!("Running bracket...");
		height += 1;
		tree = tree::combine_trees(Uuid::new_v4(), Some(Box::new(tree)), build_tree(height));
		tree.run_simulation();

		if height == 3 {
			break;
		}
	}
}