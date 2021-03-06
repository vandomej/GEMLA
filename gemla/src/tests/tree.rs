use super::super::tree::Tree;

#[test]
fn test_new() {
    assert_eq!(
        Tree::new(30, None, Some(Box::new(Tree::new(20, None, None)))),
        Tree {
            val: 30,
            left: None,
            right: Some(Box::new(Tree {
                val: 20,
                left: None,
                right: None,
            })),
        }
    );
}

#[test]
fn test_fmt() {
    assert_eq!(
        format!(
			"{}",
			btree!("foo", btree!("bar"),),
		),
        "(foo: (bar: _|_)|_)"
    );
}

#[test]
fn test_fmt_node() {
    let t = btree!(17, btree!(16), btree!(12));
    assert_eq!(
        Tree::fmt_node(&t.left),
        "16"
    );
    assert_eq!(
        Tree::fmt_node(&Some(Box::new(btree!(btree!("foo"))))),
        "(foo: _|_)"
    );
    assert_eq!(Tree::<i32>::fmt_node(&None), "_");
}
