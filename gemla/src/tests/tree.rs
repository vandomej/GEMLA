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
            Tree::new("foo", Some(Box::new(Tree::new("bar", None, None))), None)
        ),
        "(foo: (bar: _|_)|_)"
    );
}

#[test]
fn test_fmt_node() {
    assert_eq!(
        Tree::fmt_node(&Some(Box::new(Tree::new(
            17,
            Some(Box::new(Tree::new(16, None, None))),
            Some(Box::new(Tree::new(12, None, None))),
        )))),
        "17"
    );
    assert_eq!(
        Tree::fmt_node(&Some(Box::new(
            Tree::new(Tree::new("foo", None, None), None, None),
        ))),
        "(foo: _|_)"
    );
    assert_eq!(Tree::<i32>::fmt_node(&None), "_");
}
