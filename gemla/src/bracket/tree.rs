use std::fmt;
use uuid::Uuid;

pub struct Tree {
    id: Uuid,
    left: Option<Box<Tree>>,
    right: Option<Box<Tree>>
}

pub fn concat_trees(v: Uuid, l: Option<Box<Tree>>, r: Option<Box<Tree>>) -> Box<Tree> {
    Box::new(Tree {
        id: v,
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

        write!(f, "({} :{}|{})", self.id, node_str(&self.left), node_str(&self.right))
    }
}