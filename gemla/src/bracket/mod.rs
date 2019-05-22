mod tree;

use uuid::Uuid;

fn build_tree(h: u32) -> Option<Box<tree::Tree>> {
    let mut result: Option<Box<tree::Tree>> = None;

    if h != 0 {
        result = Some(tree::concat_trees(Uuid::new_v4(), build_tree(h - 1), build_tree(h - 1)));
    }

    result
}

pub fn run_bracket() {
    let mut height = 1;
    let mut tree: tree::Tree = *build_tree(height).expect("Error getting result from build_tree.");

    loop {
        println!("=========================================");
        println!("Running bracket...");
        println!("{}", tree);
        height += 1;
        tree = *tree::concat_trees(Uuid::new_v4(), Some(Box::new(tree)), build_tree(height));
    }
}