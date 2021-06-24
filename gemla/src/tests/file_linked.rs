use super::super::file_linked::FileLinked;

use std::fs;

#[test]
fn test_mutate() -> Result<(), String> {
    let tree = btree!(1, Some(btree!(2)), Some(btree!(3, Some(btree!(4)),)));
    let mut linked_tree = FileLinked::new(tree, String::from("test.txt"))?;

    assert_eq!(
        format!("{}", linked_tree.readonly()),
        "{\"val\":1,\"left\":{\"val\":2,\"left\":null,\"right\":null},\"right\":{\"val\":3,\"left\":{\"val\":4,\"left\":null,\"right\":null},\"right\":null}}"
    );

    linked_tree.mutate(|v1| v1.val = 10)?;

    assert_eq!(
        format!("{}", linked_tree.readonly()),
        "{\"val\":10,\"left\":{\"val\":2,\"left\":null,\"right\":null},\"right\":{\"val\":3,\"left\":{\"val\":4,\"left\":null,\"right\":null},\"right\":null}}"
    );

    linked_tree.mutate(|v1| {
        let mut left = v1.left.clone().unwrap();
        left.val = 13;
        v1.left = Some(left);
    })?;

    assert_eq!(
        format!("{}", linked_tree.readonly()),
        "{\"val\":10,\"left\":{\"val\":13,\"left\":null,\"right\":null},\"right\":{\"val\":3,\"left\":{\"val\":4,\"left\":null,\"right\":null},\"right\":null}}"
    );

    fs::remove_file("test.txt").expect("Unable to remove file");

    Ok(())
}
