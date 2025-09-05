extern crate fann;

use fann::Fann;
use std::os::raw::c_uint;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <network_file>", args[0]);
        std::process::exit(1);
    }

    let network_file = &args[1];
    match Fann::from_file(network_file) {
        Ok(ann) => {
            // Output layer sizes
            let layer_sizes = ann.get_layer_sizes();
            let bias_counts = ann.get_bias_counts();

            println!("Layers:");
            for (layer_size, bias_count) in layer_sizes.iter().zip(bias_counts.iter()) {
                println!("{} {}", layer_size, bias_count);
            }

            // Output connections
            println!("Connections:");
            let connections = ann.get_connections();

            for connection in connections {
                println!("{} {} {}", connection.from_neuron, connection.to_neuron, connection.weight);
            }
        },
        Err(err) => {
            eprintln!("Error loading network from file {}: {}", network_file, err);
            std::process::exit(1);
        }
    }
}
