mod tree;
mod state;

use uuid::Uuid;

// pub struct Bracket {
//     tree: tree::Tree,
//     directory: String
// }

fn build_tree(h: u32) -> Option<Box<tree::Tree>> {
	let mut result: Option<Box<tree::Tree>> = None;

	if h != 0 {
		result = Some(Box::new(tree::combine_trees(Uuid::new_v4(), build_tree(h - 1), build_tree(h - 1))));
		match &result {
			Some(r) => (*r).run_simulation(),
			_ => ()
		}
	}

	result
}

pub fn run_bracket() {
	let mut height = 1;
	let mut tree: tree::Tree = *build_tree(height).expect("Error getting result from build_tree.");

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