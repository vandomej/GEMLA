use std::fmt;
use super::state;

pub struct Tree {
    state: state::State,
    left: Option<Box<Tree>>,
    right: Option<Box<Tree>>
}

pub fn concat_trees(s: state::State, l: Option<Box<Tree>>, r: Option<Box<Tree>>) -> Box<Tree> {
    Box::new(Tree {
        state: s,
        left: l,
        right: r
    })
}

impl fmt::Display for Tree {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let node_str = |t: &Option<Box<Tree>>| -> String {
            match t {
                Some(n) => format!("{}", *n),
                _ => String::from("_")
            }
        };

        write!(f, "({} :{}|{})", self.state, node_str(&self.left), node_str(&self.right))
    }
}

fn fmt_node(t: &Option<Box<Tree>>) -> String {
    match t {
        Some(n) => format!("{}", (*n).state),
        _ => String::from("_")
    }
}

impl Tree {
    pub fn run_simulation(&self) {
        println!("================================");
        println!("Running simulation for node: {}", self.state);
        println!("With children {} and {}", fmt_node(&self.left), fmt_node(&self.right));
    }
}