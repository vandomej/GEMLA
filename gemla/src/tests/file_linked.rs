use super::super::file_linked::FileLinked;
use super::super::tree::Tree;

#[test]
fn test_mutate() -> Result<(), String> {
    let tree = btree!(1, Some(btree!(2)), Some(btree!(3, Some(btree!(4)),)));
    let mut linked_tree = FileLinked::new(tree, "blah.txt")?;

    assert_eq!(
        format!("{}", linked_tree.readonly()),
        "val = 1\n\n[left]\nval = 2\n\n[right]\nval = 3\n\n[right.left]\nval = 4\n"
    );

    linked_tree.mutate(|v1| v1.val = 10)?;

    assert_eq!(
        format!("{}", linked_tree.readonly()),
        "val = 10\n\n[left]\nval = 2\n\n[right]\nval = 3\n\n[right.left]\nval = 4\n"
    );

    linked_tree.mutate(|v1| {
        let mut left = v1.left.clone().unwrap();
        left.val = 13;
        v1.left = Some(left);
    })?;

    assert_eq!(
        format!("{}", linked_tree.readonly()),
        "val = 10\n\n[left]\nval = 13\n\n[right]\nval = 3\n\n[right.left]\nval = 4\n"
    );

    Ok(())
}
