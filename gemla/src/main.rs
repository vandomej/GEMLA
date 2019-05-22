mod tree;

fn build_tree(h: u32) -> Option<Box<tree::Tree>> {
    let mut result: Option<Box<tree::Tree>> = None;

    if h != 0 {
        result = Some(tree::concat_trees(h, build_tree(h - 1), build_tree(h - 1)));
    }

    result
}

fn main() {
    let tree: tree::Tree = *build_tree(25).expect("Error getting result from build tree.");

    println!("Resulting tree from build_tree.");
    println!("{}", tree);
}