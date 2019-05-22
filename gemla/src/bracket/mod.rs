mod tree;
mod state;

fn build_tree(h: u32) -> Option<Box<tree::Tree>> {
    let mut result: Option<Box<tree::Tree>> = None;

    if h != 0 {
        result = Some(tree::concat_trees(state::create(), build_tree(h - 1), build_tree(h - 1)));
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
        tree = *tree::concat_trees(state::create(), Some(Box::new(tree)), build_tree(height));
        tree.run_simulation();
    }
}