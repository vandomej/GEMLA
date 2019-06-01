use std::fmt;
use std::str::FromStr;
use regex::Regex;

#[derive(Default, Clone)]
pub struct Tree<T> {
	pub val: T,
	pub left: Option<Box<Tree<T>>>,
	pub right: Option<Box<Tree<T>>>
}

impl<T> Tree<T> {
	pub fn new(val: T, left: Option<Box<Tree<T>>>, right: Option<Box<Tree<T>>>) -> Tree<T> {
		Tree {
			val,
			left,
			right
		}
	}
}

impl<T: fmt::Display> fmt::Display for Tree<T> {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		let node_str = |t: &Option<Box<Tree<T>>>| -> String {
			match t {
				Some(n) => format!("{}", *n),
				_ => String::from("_")
			}
		};

		write!(f, "({} :{}|{})", self.val, node_str(&self.left), node_str(&self.right))
	}
}

pub fn fmt_node<T: fmt::Display>(t: &Option<Box<Tree<T>>>) -> String {
	match t {
		Some(n) => format!("{}", (*n).val),
		_ => String::from("_")
	}
}

fn seperate_nodes(s: &str) -> Result<(&str, &str), ParseTreeError> {
	let mut result = Err(ParseTreeError::new(format!("Unable to seperate string: {}", s)));
	let mut stack: Vec<char> = Vec::new();

	for (i, c) in s.char_indices() {
		if c == '(' {
			stack.push(c);
		} else if c == ')' {
			if stack.is_empty() {
				result = Err(ParseTreeError::new(format!("Unbalanced parenthesis found in string: {}", s)));
				break;
			}

			stack.pop();
		} else if c == '|' && stack.is_empty() {
			result = Ok((&s[..i], &s[i+1..]));
			break;
		}
	}

	result
}

fn from_str_helper<T: FromStr>(s: &str) -> Result<Option<Box<Tree<T>>>, ParseTreeError> {
	let mut result = Err(ParseTreeError::new(String::from("Unable to parse tree, string format unrecognized.")));
	let emptyre = Regex::new(r"\s*_\s*").unwrap();
	let re = Regex::new(r"\(([0-9a-fA-F-]+)\s*:\s*(.*)\)$").unwrap();
	let caps = re.captures(s);

	if let Some(c) = caps {
		let val = T::from_str(c.get(1).unwrap().as_str())
		.or(Err(ParseTreeError::new(format!("Unable to parse node value: {}", c.get(1).unwrap().as_str()))))?;
		let (left, right) = seperate_nodes(c.get(2).unwrap().as_str())?;
		let left = from_str_helper(left)?;
		let right = from_str_helper(right)?;

		result = Ok(Some(Box::new(Tree::new(val, left, right))));
	} else if emptyre.is_match(s) {
		result = Ok(None);
	}

	result
}

impl<T> FromStr for Tree<T> where T: FromStr {
	type Err = ParseTreeError;

	fn from_str(s: &str) -> Result<Self, Self::Err> {
		let mut result = Err(ParseTreeError::new(String::from("Unable to parse tree, string format unreognized.")));
		let re = Regex::new(r"\(([0-9a-fA-F-]+)\s*:\s*(.*)\)$").unwrap();
		let caps = re.captures(s);

		if let Some(c) = caps {
			let val = T::from_str(c.get(1).unwrap().as_str())
			.or(Err(ParseTreeError::new(format!("Unable to parse node value: {}", c.get(1).unwrap().as_str()))))?;
			let (left, right) = seperate_nodes(c.get(2).unwrap().as_str())?;
			let left = from_str_helper(left)?;
			let right = from_str_helper(right)?;
			
			result = Ok(Tree::new(val, left, right));
		}

		// if let Some(c) = caps {
		// 	result = T::from_str(c.get(1).unwrap().as_str())
		// 	.or(Err(ParseTreeError::new(format!("Unable to parse node value: {}", c.get(1).unwrap().as_str()))))
		// 	.and_then(|v| {
		// 		seperate_nodes(c.get(2).unwrap().as_str())
		// 		.and_then(|(left, right)| {
		// 			from_str_helper(left)
		// 			.and_then(|l| {
		// 				from_str_helper(right)
		// 				.and_then(|r| {
		// 					Ok(Tree::new(v, l, r))
		// 				})
		// 			})
		// 		})
		// 	})
		// }

		// if let Some(c) = caps {
		// 	let val = T::from_str(c.get(1).unwrap().as_str());
		// 	if let Ok(v) = val {
		// 		match seperate_nodes(c.get(2).unwrap().as_str()) {
		// 			Ok((l, r)) => {
		// 				match (from_str_helper(l), from_str_helper(r)) {
		// 					(Ok(left), Ok(right)) => {
		// 						result = Ok(Tree::new(v, left, right));
		// 					},
		// 					(Err(e), _) => {
		// 						result = Err(e);
		// 					},
		// 					(_, Err(e)) => {
		// 						result = Err(e);
		// 					}
		// 				}
		// 			},
		// 			Err(e) => {
		// 				result = Err(e);
		// 			}
		// 		}
		// 	} else {
		// 		result = Err(ParseTreeError::new(format!("Unable to parse node value: {}", c.get(1).unwrap().as_str())));
		// 	}
		// }

		result
	}
}

#[derive(Debug)]
pub struct ParseTreeError {
	pub msg: String
}

impl ParseTreeError {
	fn new(msg: String) -> ParseTreeError {
		ParseTreeError {
			msg
		}
	}
}