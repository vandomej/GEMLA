use std::{cmp::min, collections::HashMap, ops::Range};

use anyhow::Context;
use fann::{ActivationFunc, Fann};
use gemla::error::Error;
use rand::{distributions::{Distribution, Uniform}, seq::IteratorRandom, thread_rng, Rng};

use super::{FighterNN, NEURAL_NETWORK_HIDDEN_LAYER_SIZE_MIN};


/// Crossbreeds two neural networks of different shapes by finding cut points, and swapping neurons between the two networks.
/// Algorithm tries to ensure similar functionality is maintained between the two networks.
/// It does this by preserving connections between the same neurons from the original to the new network, and if a connection cannot be found
/// it will create a new connection with a random weight.
pub fn crossbreed(fighter_nn: &FighterNN, primary: &Fann, secondary: &Fann, crossbreed_segments: usize) -> Result<Fann, Error> {
    // First we need to get the shape of the networks and transform this into a format that is easier to work with
    // We want a list of every neuron id, and the layer it is in
    let primary_shape = primary.get_layer_sizes();
    let secondary_shape = secondary.get_layer_sizes();
    let primary_neurons = generate_neuron_datastructure(&primary_shape);
    let secondary_neurons = generate_neuron_datastructure(&secondary_shape);

    let segments = generate_segments(primary_shape, secondary_shape, crossbreed_segments);
        
    let new_neurons = crossbreed_neuron_arrays(segments, primary_neurons, secondary_neurons);
    
    // Now we need to create the new network with the shape we've determined
    let mut new_shape = vec![];
    for (_, _, layer, _) in new_neurons.iter() {
        // Check if new_shape has an entry for layer in it
        if new_shape.len() <= *layer as usize {
            new_shape.push(1);
        }
        else {
            new_shape[*layer as usize] += 1;
        }
    }

    let mut new_fann = Fann::new(new_shape.as_slice())
        .with_context(|| "Failed to create new fann")?;
    // We need to randomize the weights to a small value
    new_fann.randomize_weights(fighter_nn.weight_initialization_range.start, fighter_nn.weight_initialization_range.end);
    new_fann.set_activation_func_hidden(ActivationFunc::SigmoidSymmetric);
    new_fann.set_activation_func_output(ActivationFunc::SigmoidSymmetric);

    consolidate_old_connections(primary, secondary, new_shape, new_neurons, &mut new_fann);

    Ok(new_fann)
}

pub fn generate_segments(primary_shape: Vec<u32>, secondary_shape: Vec<u32>, crossbreed_segments: usize) -> Vec<(u32, u32)> {
    // Now we need to find the cut points for the crossbreed
    let start = primary_shape[0] + 1;
    // Start at the first hidden layer
    let end = min(primary_shape.iter().sum::<u32>() - primary_shape.last().unwrap(), secondary_shape.iter().sum::<u32>() - secondary_shape.last().unwrap());
    // End at the last hidden layer
    let segment_distribution = Uniform::from(start..end);
    // Ensure segments are not too small

    let mut cut_points = Vec::new();
    for _ in 0..crossbreed_segments {
        let cut_point = segment_distribution.sample(&mut thread_rng());
        if !cut_points.contains(&cut_point) {
            cut_points.push(cut_point);
        }
    }
    // Sort the cut points to make it easier to iterate over them
    cut_points.sort_unstable();

    // We need to transform the cut_points vector to a vector of tuples that contain the start and end of each segment
    let mut segments = Vec::new();
    let mut previous = 0;
    for &cut_point in cut_points.iter() {
        segments.push((previous, cut_point - 1));
        previous = cut_point;
    }
    segments
}

pub fn consolidate_old_connections(primary: &Fann, secondary: &Fann, new_shape: Vec<u32>, new_neurons: Vec<(u32, bool, usize, u32)>, new_fann: &mut Fann) {
    // Now we need to copy the connections from the original networks to the new network
    // We can do this by referencing our connections array, it will contain the original id's of the neurons 
    // and their new id as well as their layer. We can iterate one layer at a time and copy the connections

    let primary_shape = primary.get_layer_sizes();
    let secondary_shape = secondary.get_layer_sizes();
    debug!("Primary shape: {:?}", primary_shape);
    debug!("Secondary shape: {:?}", secondary_shape);
    
    // Start by iterating layer by later
    let primary_connections = primary.get_connections();
    let secondary_connections = secondary.get_connections();
    for layer in 1..new_shape.len() {
        // filter out the connections that are in the current layer and previous layer
        let current_layer_connections = new_neurons.iter().filter(|(_, _, l, _)| l == &layer).collect::<Vec<_>>();
        let previous_layer_connections = new_neurons.iter().filter(|(_, _, l, _)| l == &(layer - 1)).collect::<Vec<_>>();

        // Now we need to iterate over the connections in the current layer
        for (neuron_id, is_primary, _, new_id) in current_layer_connections.iter() {
            // We need to find the connections from the previous layer to this neuron
            for (previous_neuron_id, _, _, previous_new_id) in previous_layer_connections.iter() {
                // First we use primary to and check the correct connections array to see if the connection exists
                // If it does, we add it to the new network
                let mut connection;
                let mut found_in_primary = false;
                if *is_primary {
                    connection = primary_connections.iter()
                        .find(|connection| {
                            let from_neuron = to_non_bias_network_id(connection.from_neuron, &primary_shape);
                            let to_neuron = to_non_bias_network_id(connection.to_neuron, &primary_shape);

                            // If both neurons have a Some value
                            if let (Some(from_neuron), Some(to_neuron)) = (from_neuron, to_neuron) {
                                from_neuron == *previous_neuron_id && to_neuron == *neuron_id
                            } else {
                                false
                            }
                        });

                    if let None = connection {
                        connection = secondary_connections.iter()
                            .find(|connection| {
                                let from_neuron = to_non_bias_network_id(connection.from_neuron, &secondary_shape);
                                let to_neuron = to_non_bias_network_id(connection.to_neuron, &secondary_shape);

                                // If both neurons have a Some value
                                if let (Some(from_neuron), Some(to_neuron)) = (from_neuron, to_neuron) {
                                    from_neuron == *previous_neuron_id && to_neuron == *neuron_id
                                } else {
                                    false
                                }
                            });
                    } else {
                        found_in_primary = true;
                    }
                }
                else {
                    connection = secondary_connections.iter()
                        .find(|connection| {
                            let from_neuron = to_non_bias_network_id(connection.from_neuron, &secondary_shape);
                            let to_neuron = to_non_bias_network_id(connection.to_neuron, &secondary_shape);

                            // If both neurons have a Some value
                            if let (Some(from_neuron), Some(to_neuron)) = (from_neuron, to_neuron) {
                                from_neuron == *previous_neuron_id && to_neuron == *neuron_id
                            } else {
                                false
                            }
                        });

                    if let None = connection {
                        connection = primary_connections.iter()
                            .find(|connection| {
                                let from_neuron = to_non_bias_network_id(connection.from_neuron, &primary_shape);
                                let to_neuron = to_non_bias_network_id(connection.to_neuron, &primary_shape);
    
                                // If both neurons have a Some value
                                if let (Some(from_neuron), Some(to_neuron)) = (from_neuron, to_neuron) {
                                    from_neuron == *previous_neuron_id && to_neuron == *neuron_id
                                } else {
                                    false
                                }
                            });
                    } else {
                        found_in_primary = true;
                    }
                };

                // If the connection exists, we need to add it to the new network
                if let Some(connection) = connection {
                    if *is_primary {
                        let original_from_neuron = to_non_bias_network_id(connection.from_neuron, &primary_shape);
                        let original_to_neuron = to_non_bias_network_id(connection.to_neuron, &primary_shape);
                        debug!("Primary: Adding connection from ({} -> {}) translated to ({:?} -> {:?}) with weight {} for primary:{} [{} -> {}] [{} -> {}]", previous_new_id, new_id, original_from_neuron, original_to_neuron, connection.weight, found_in_primary, connection.from_neuron, connection.to_neuron, previous_neuron_id, neuron_id);
                    } else {
                        let original_from_neuron = to_non_bias_network_id(connection.from_neuron, &secondary_shape);
                        let original_to_neuron = to_non_bias_network_id(connection.to_neuron, &secondary_shape);
                        debug!("Secondary: Adding connection from ({} -> {}) translated to ({:?} -> {:?}) with weight {} for primary:{} [{} -> {}] [{} -> {}]", previous_new_id, new_id, original_from_neuron, original_to_neuron, connection.weight, found_in_primary, connection.from_neuron, connection.to_neuron, previous_neuron_id, neuron_id);
                    }
                    let translated_from = to_bias_network_id(previous_new_id, &new_shape);
                    let translated_to = to_bias_network_id(new_id, &new_shape);
                    new_fann.set_weight(translated_from, translated_to, connection.weight);
                } else {
                    debug!("Connection not found for ({}, {}) -> ({}, {})",  previous_new_id, new_id, previous_neuron_id, neuron_id);
                }
            }
        }

        // Add bias neuron connections
        let bias_neuron = get_bias_neuron_for_layer(layer, &new_shape);
        if let Some(bias_neuron) = bias_neuron {
            // Loop through neurons in current layer
            for (neuron_id, is_primary, _, new_id) in current_layer_connections.iter() {
                let translated_neuron_id = to_bias_network_id(new_id, &new_shape);

                let mut connection;
                let mut found_in_primary = false;
                if *is_primary {
                    let primary_bias_neuron = get_bias_neuron_for_layer(layer, &primary_shape).unwrap();
                    connection = primary_connections.iter()
                        .find(|connection| {
                            let to_neuron = to_non_bias_network_id(connection.to_neuron, &primary_shape);

                            if let Some(to_neuron) = to_neuron {
                                connection.from_neuron == primary_bias_neuron && to_neuron == *neuron_id
                            } else {
                                false
                            }
                        });

                    if let None = connection {
                        let secondary_bias_neuron = get_bias_neuron_for_layer(layer, &secondary_shape).unwrap();
                        connection = secondary_connections.iter()
                            .find(|connection| {
                                let to_neuron = to_non_bias_network_id(connection.to_neuron, &secondary_shape);

                                if let Some(to_neuron) = to_neuron {
                                    connection.from_neuron == secondary_bias_neuron && to_neuron == *neuron_id
                                } else {
                                    false
                                }
                            });
                    } else {
                        found_in_primary = true;
                    }
                } else {
                    let secondary_bias_neuron = get_bias_neuron_for_layer(layer, &secondary_shape).unwrap();
                    connection = secondary_connections.iter()
                        .find(|connection| {
                            let to_neuron = to_non_bias_network_id(connection.to_neuron, &secondary_shape);

                            if let Some(to_neuron) = to_neuron {
                                connection.from_neuron == secondary_bias_neuron && to_neuron == *neuron_id
                            } else {
                                false
                            }
                        });

                    if let None = connection {
                        let primary_bias_neuron = get_bias_neuron_for_layer(layer, &primary_shape).unwrap();
                        connection = primary_connections.iter()
                            .find(|connection| {
                                let to_neuron = to_non_bias_network_id(connection.to_neuron, &primary_shape);

                                if let Some(to_neuron) = to_neuron {
                                    connection.from_neuron == primary_bias_neuron && to_neuron == *neuron_id
                                } else {
                                    false
                                }
                            });
                    } else {
                        found_in_primary = true;
                    }
                }

                if let Some(connection) = connection {
                    if *is_primary {
                        let original_from_neuron = to_non_bias_network_id(connection.from_neuron, &primary_shape);
                        let original_to_neuron = to_non_bias_network_id(connection.to_neuron, &primary_shape);
                        debug!("Primary: Adding connection from ({} -> {}) translated to ({:?} -> {:?}) with weight {} for primary:{} [{} -> {}] [{} -> {}]", bias_neuron, translated_neuron_id, original_from_neuron, original_to_neuron, connection.weight, found_in_primary, connection.from_neuron, connection.to_neuron, bias_neuron, neuron_id);
                    } else {
                        let original_from_neuron = to_non_bias_network_id(connection.from_neuron, &secondary_shape);
                        let original_to_neuron = to_non_bias_network_id(connection.to_neuron, &secondary_shape);
                        debug!("Secondary: Adding connection from ({} -> {}) translated to ({:?} -> {:?}) with weight {} for primary:{} [{} -> {}] [{} -> {}]", bias_neuron, translated_neuron_id, original_from_neuron, original_to_neuron, connection.weight, found_in_primary, connection.from_neuron, connection.to_neuron, bias_neuron, neuron_id);
                    }
                    new_fann.set_weight(bias_neuron, translated_neuron_id, connection.weight);
                } else {
                    debug!("Connection not found for bias ({}, {}) -> ({}, {}) primary: {}",  bias_neuron, neuron_id, bias_neuron, translated_neuron_id, is_primary);
                }
            }
        }
    }
}

pub fn crossbreed_neuron_arrays(segments: Vec<(u32, u32)>, primary_neurons: Vec<(u32, usize)>, secondary_neurons: Vec<(u32, usize)>) -> Vec<(u32, bool, usize, u32)> {
    // We now need to determine the resulting location of the neurons in the new network.
    // To do this we need a new structure that keeps track of the following information:
    // - The neuron id from the original network
    // - Which network it originated from (primary or secondary)
    // - The layer the neuron is in
    // - The resulting neuron id in the new network which will be calculated after the fact
    let mut new_neurons = Vec::new();
    let mut current_layer = 0;
    // keep track of the last layer that we inserted a neuron into for each network
    let mut primary_last_layer = 0;
    let mut secondary_last_layer = 0;
    let mut is_primary = true;
    for (i, &segment) in segments.iter().enumerate() {
        // If it's the first slice, copy neurons from the primary network up to the cut_point
        if i == 0 {
            for (neuron_id, layer) in primary_neurons.iter() {
                if neuron_id <= &segment.1 {
                    if layer > &current_layer {
                        current_layer += 1;
                    }
                    new_neurons.push((*neuron_id, is_primary, current_layer, 0));
                    if is_primary {
                        primary_last_layer = current_layer;
                    }
                    else {
                        secondary_last_layer = current_layer;
                    }
                }
                else {
                    break;
                }
            }
        }
        else {
            let target_neurons = if is_primary { &primary_neurons } else { &secondary_neurons };

            for (neuron_id, layer) in target_neurons.iter() {
                // Iterate until neuron_id equals the cut_point
                if neuron_id >= &segment.0 && neuron_id <= &segment.1 {
                    // We need to do something different depending on whether the neuron layer is, lower, higher or equal to the target layer
                
                    // Equal
                    if layer == &current_layer {
                        new_neurons.push((*neuron_id, is_primary, current_layer, 0));

                        if is_primary {
                            primary_last_layer = current_layer;
                        }
                        else {
                            secondary_last_layer = current_layer;
                        }
                    }
                    // Earlier
                    else if layer < &current_layer {
                        // If it's in an earlier layer, add it to the earlier layer
                        // Check if there's a lower id from the same individual in that earlier layer
                        // As long as there isn't a neuron from the other individual in between the lower id and current id, add the id values from the same individual
                        let earlier_layer_neurons = new_neurons.iter().filter(|(_, _, l, _)| l == layer).collect::<Vec<_>>();
                        // get max id from that layer
                        let highest_id = earlier_layer_neurons.iter().max_by(|a, b| a.2.cmp(&b.2).then(a.0.cmp(&b.0)));
                        if let Some(highest_id) = highest_id {
                            if highest_id.1 == is_primary {
                                let neurons_to_add = target_neurons.iter().filter(|(id, l)| id > &highest_id.0 && id < neuron_id  && l == layer).collect::<Vec<_>>();
                                for (neuron_id, layer) in neurons_to_add {
                                    new_neurons.push((*neuron_id, is_primary, *layer, 0));

                                    if is_primary {
                                        primary_last_layer = *layer;
                                    }
                                    else {
                                        secondary_last_layer = *layer;
                                    }
                                }
                            }
                        }

                        new_neurons.push((*neuron_id, is_primary, *layer, 0));

                        if is_primary {
                            primary_last_layer = *layer;
                        }
                        else {
                            secondary_last_layer = *layer;
                        }
                    }
                    // Later
                    else if layer > &current_layer {
                        // If the highest id in the current layer is from the same individual, add anything with a higher id to the current layer before moving to the next layer
                        // First filter new_neurons to look at neurons from the current layer
                        let current_layer_neurons = new_neurons.iter().filter(|(_, _, l, _)| l == &current_layer).collect::<Vec<_>>();
                        let highest_id = current_layer_neurons.iter().max_by_key(|(id, _, _, _)| id);
                        if let Some(highest_id) = highest_id {
                            if highest_id.1 == is_primary {
                                let neurons_to_add = target_neurons.iter().filter(|(id, l)| id > &highest_id.0 && *l == layer - 1).collect::<Vec<_>>();
                                for (neuron_id, _) in neurons_to_add {
                                    new_neurons.push((*neuron_id, is_primary, current_layer, 0));

                                    if is_primary {
                                        primary_last_layer = current_layer;
                                    }
                                    else {
                                        secondary_last_layer = current_layer;
                                    }
                                }
                            }
                        }

                        // If it's in a future layer, move to the next layer
                        current_layer += 1;

                        // Add the neuron to the new network
                        // Along with any neurons that have a lower id in the future layer
                        let neurons_to_add = target_neurons.iter().filter(|(id, l)| id <= &neuron_id && l == layer).collect::<Vec<_>>();
                        for (neuron_id, _) in neurons_to_add {
                            new_neurons.push((*neuron_id, is_primary, current_layer, 0));

                            if is_primary {
                                primary_last_layer = current_layer;
                            }
                            else {
                                secondary_last_layer = current_layer;
                            }
                        }
                    }
        
                }
                else if neuron_id >= &segment.1 {
                    break;
                }
            }
        }

        // Switch to the other network
        is_primary = !is_primary;
    }

    // For the last segment, copy the remaining neurons
    let target_neurons = if is_primary { &primary_neurons } else { &secondary_neurons };
    // Get output layer number
    let output_layer = target_neurons.iter().max_by_key(|(_, l)| l).unwrap().1;

    // For the last segment, copy the remaining neurons from the target network
    // But when we reach the output layer, we need to add a new layer to the end of new_neurons regardless of it's length
    // and copy the output neurons to that layer
    for (neuron_id, layer) in target_neurons.iter() {
        if neuron_id > &segments.last().unwrap().1 {
            if layer == &output_layer {
                // Calculate which layer the neurons should be in
                current_layer = new_neurons.iter().max_by_key(|(_, _, l, _)| l).unwrap().2 + 1;
                for (neuron_id, _) in target_neurons.iter().filter(|(_, l)| l == &output_layer) {
                    new_neurons.push((*neuron_id, is_primary, current_layer, 0));
                }
                break;
            }
            else if *neuron_id == &segments.last().unwrap().1 + 1 {
                let target_layer = if is_primary { primary_last_layer } else { secondary_last_layer };
                let earlier_layer_neurons = new_neurons.iter().filter(|(_, _, l, _)| *l >= target_layer && l <= layer).collect::<Vec<_>>();
                // get max neuron from with both
                // The highest layer
                // get max id from that layer
                let highest_id = earlier_layer_neurons.iter().max_by(|a, b| a.2.cmp(&b.2).then(a.0.cmp(&b.0)));
                if let Some(highest_id) = highest_id {
                    if highest_id.1 == is_primary {
                        let neurons_to_add = target_neurons.iter().filter(|(id, _)| id > &highest_id.0 && id < neuron_id).collect::<Vec<_>>();
                        for (neuron_id, l) in neurons_to_add {
                            new_neurons.push((*neuron_id, is_primary, *l, 0));
                        }
                    }
                }

                new_neurons.push((*neuron_id, is_primary, *layer, 0));
            }
            else {
                new_neurons.push((*neuron_id, is_primary, *layer, 0));
            }
        }
    }

    // Filtering layers with too few neurons, if necessary
    let layer_counts = new_neurons.iter().fold(vec![0; current_layer + 1], |mut counts, &(_, _, layer, _)| {
        counts[layer] += 1;
        counts
    });

    // Filter out layers based on the minimum number of neurons per layer
    new_neurons = new_neurons.into_iter()
    .filter(|&(_, _, layer, _)| layer_counts[layer] >= NEURAL_NETWORK_HIDDEN_LAYER_SIZE_MIN)
    .collect::<Vec<_>>();

    // Collect and sort unique layer numbers
    let mut unique_layers = new_neurons.iter()
        .map(|(_, _, layer, _)| *layer)
        .collect::<Vec<_>>();
    unique_layers.sort();
    unique_layers.dedup(); // Removes duplicates, keeping only unique layer numbers

    // Create a mapping from old layer numbers to new (gap-less) layer numbers
    let layer_mapping = unique_layers.iter().enumerate()
        .map(|(new_layer, &old_layer)| (old_layer, new_layer))
        .collect::<HashMap<usize, usize>>();

    // Apply the mapping to renumber layers in new_neurons
    new_neurons.iter_mut().for_each(|(_, _, layer, _)| {
        *layer = *layer_mapping.get(layer).unwrap_or(layer); // Fallback to original layer if not found, though it should always find a match
    });

    // Assign new IDs
    // new_neurons must be sorted by layer, then by neuron ID within the layer
    new_neurons.sort_unstable_by(|a, b| a.2.cmp(&b.2).then(a.0.cmp(&b.0)));
    new_neurons.iter_mut().enumerate().for_each(|(new_id, neuron)| {
        neuron.3 = new_id as u32;
    });

    new_neurons
}

pub fn major_mutation(fann: &Fann, weight_initialization_range: Range<f32>) -> Result<Fann, Error> {
    // add or remove a random neuron from a hidden layer
    let mut mutated_shape = fann.get_layer_sizes().to_vec();
    let mut mutated_neurons = generate_neuron_datastructure(&mutated_shape).iter().map(|(id, layer)| (*id, true, *layer, *id)).collect::<Vec<_>>();

    // Determine first whether to add or remove a neuron
    if thread_rng().gen_range(0..2) == 0 {
        // To add a neuron we need to create a new fann object with the new layer sizes, then copy the information and connections over
        let max_id = mutated_neurons.iter().max_by_key(|(id, _, _, _)| id).unwrap().0;

        // Now we inject the new neuron into mutated_neurons
        let layer = thread_rng().gen_range(1..fann.get_num_layers() - 1) as usize;
        let new_id = max_id + 1;
        mutated_neurons.push((new_id, true, layer, new_id));
        mutated_shape[layer] += 1;
    }
    else {
        // Remove a neuron
        let layer = thread_rng().gen_range(1..fann.get_num_layers() - 1) as usize;
        // Do not remove from layer if it would result in less than NEURALNETWORK_HIDDEN_LAYER_SIZE_MIN neurons
        if mutated_shape[layer] > NEURAL_NETWORK_HIDDEN_LAYER_SIZE_MIN as u32 {
            let remove_id = mutated_neurons.iter().filter(|(_, _, l, _)| l == &layer).choose(&mut thread_rng()).unwrap().0;
            mutated_neurons.retain(|(id, _, _, _)| id != &remove_id);
            mutated_shape[layer] -= 1;
        }
    }

    let mut mutated_fann = Fann::new(mutated_shape.as_slice())
        .with_context(|| "Failed to create new fann")?;
    mutated_fann.randomize_weights(weight_initialization_range.start, weight_initialization_range.end);
    mutated_fann.set_activation_func_hidden(ActivationFunc::SigmoidSymmetric);
    mutated_fann.set_activation_func_output(ActivationFunc::SigmoidSymmetric);

    // We need to regenerate the new_id's in mutated_neurons (the 4th item in the tuple) we can do this by iterating over the mutated_neurons all over again starting from ZERO
    mutated_neurons.sort_by(|a, b| a.2.cmp(&b.2).then(a.0.cmp(&b.0)));
    let mut i = 0;
    for (_, _, _, new_id) in mutated_neurons.iter_mut() {
        *new_id = i;
        i += 1;
    }

    // We need to copy the connections from the old fann to the new fann
    consolidate_old_connections(&fann, &fann, mutated_shape, mutated_neurons, &mut mutated_fann);

    Ok(mutated_fann)
}

pub fn generate_neuron_datastructure(shape: &[u32]) -> Vec<(u32, usize)> {
    let mut result = Vec::new();
    let mut global_index = 0; // Keep a global index that does not reset

    for (layer_index, &neurons) in shape.iter().enumerate() {
        for _ in 0..neurons {
            result.push((global_index, layer_index));
            global_index += 1; // Increment global index for each neuron
        }
        // global_index += 1; // Skip index for bias neuron at the end of each layer
    }

    result
}


fn to_bias_network_id(id: &u32, shape: &Vec<u32>) -> u32 {
    // The given id comes from a network without a bias neuron at the end of every layer
    // We need to translate this id to the id in the network with bias neurons
    let mut translated_id = 0;
    for (layer_index, &neurons) in shape.iter().enumerate() {
        for _ in 0..neurons {
            if &translated_id == id {
                return translated_id + layer_index as u32;
            }
            translated_id += 1;
        }
    }

    // If the id is not found, return the id
    translated_id
}

fn to_non_bias_network_id(id: u32, shape: &[u32]) -> Option<u32> {
    let mut bias_count = 0; // Count of bias neurons encountered up to the current ID
    let mut total_neurons = 0; // Total count of neurons (excluding bias neurons) processed

    for &neurons in shape {
        let layer_end = total_neurons + neurons; // End of the current layer, excluding the bias neuron
        if id < layer_end {
            // ID is within the current layer (excluding the bias neuron)
            return Some(id - bias_count);
        }
        if id == layer_end {
            // ID matches the position where a bias neuron would be
            return None;
        }

        // Update counts after considering the current layer
        total_neurons += neurons + 1; // Move to the next layer, accounting for the bias neuron
        bias_count += 1; // Increment bias count as we've moved past where a bias neuron would be
    }

    // If the ID is beyond the range of all neurons (including bias), it's treated as invalid
    // Adjust this behavior based on your application's needs
    None
}

fn get_bias_neuron_for_layer(layer: usize, shape: &[u32]) -> Option<u32> {
    if layer == 0 || layer >= shape.len() {
        // No bias neuron for the first and last layers
        None
    } else {
        // Compute the bias neuron for intermediate layers
        let mut bias = 0;
        for i in 0..layer {
            bias += shape[i];
        }
        Some(bias + layer as u32 - 1)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn major_mutation_test() -> Result<(), Box<dyn std::error::Error>> {
        // Assign
        let primary_shape = vec![2, 8, 5, 3, 1];
        // [2, 11, 17, 21]
        // [0, 1,  2,  3]

        let mut primary_fann = Fann::new(&primary_shape)?;

        let mut primary_connections = primary_fann.get_connections();
        for connection in primary_connections.iter_mut() {
            connection.weight = ((connection.from_neuron * 100) + connection.to_neuron) as f32;
        }
        primary_fann.set_connections(&primary_connections);

        let weight_initialization_range = -1.0..-0.5;

        for _ in 0..100 {
            let result = major_mutation(&primary_fann, weight_initialization_range.clone())?;

            let connections = result.get_connections();
            for connection in connections.iter() {
                println!("Connection: {:?}", connection);
            }

            let new_shape = result.get_layer_sizes();
            println!("New Shape: {:?}", new_shape);

            // Assert that input and output layers have the same size
            assert_eq!(primary_shape[0], new_shape[0]);
            assert_eq!(primary_shape[primary_shape.len() - 1], new_shape[new_shape.len() - 1]);

            // Determine if a neuron was removed or added
            if new_shape.iter().sum::<u32>() == primary_shape.iter().sum::<u32>() + 1 {
                //Neuron was added
                // Find id of neuron that was added
                let mut added_neuron_id = 0;
                let matching_layers = new_shape.iter().zip(primary_shape.iter());
                for (i, (new_layer, primary_layer)) in matching_layers.enumerate() {
                    if new_layer > primary_layer {
                        added_neuron_id += primary_layer + i as u32;
                        break;
                    }
                    added_neuron_id += primary_layer;
                }

                for connection in connections.iter() {
                    if connection.from_neuron == added_neuron_id || connection.to_neuron == added_neuron_id {
                        assert!(connection.weight < 0.0, "Connection: {:?}, Added Neuron: {}", connection, added_neuron_id);
                    } else {
                        assert!(connection.weight > 0.0, "Connection: {:?}, Added Neuron: {}", connection, added_neuron_id);
                    }
                }
            }
            else if new_shape.iter().sum::<u32>() == primary_shape.iter().sum::<u32>() - 1 {
                //Neuron was removed
                for connection in connections.iter() {
                    assert!(connection.weight > 0.0, "Connection: {:?}", connection);
                }

                for (i, layer) in new_shape.iter().enumerate() {
                    // if layer isn't input or output
                    if i != 0 && i as u32 != new_shape.len() as u32 - 1 {
                        assert!(*layer >= NEURAL_NETWORK_HIDDEN_LAYER_SIZE_MIN as u32, "Layer: {}", layer);
                    }
                }
            }
            else {
                //Neuron was neither added nor removed
                for connection in connections.iter() {
                    assert!(connection.weight > 0.0, "Connection: {:?}", connection);
                }
            }
        }

        Ok(())
    }

    #[test]
    fn generate_segments_test() {
        // Assign
        let primary_shape = vec![4, 8, 6, 4];
        let secondary_shape = vec![4, 3, 3, 3, 3, 3, 4];
        let crossbreed_segments = 5;

        // Act
        let result = generate_segments(primary_shape.clone(), secondary_shape.clone(), crossbreed_segments);

        println!("{:?}", result);

        // Assert
        assert!(result.len() <= crossbreed_segments, "Segments: {:?}", result);
        //Assert that segments are within the bounds of the layers
        for (start, end) in result.iter() {
            // Bounds are the end of the first layer to the end of the second to last layer
            let bounds = 3..17;

            assert!(bounds.contains(end));
            assert!(start < &bounds.end);
        }

        //Assert that segments start and end are in ascending order
        for (start, end) in result.iter()
        {
            assert!(*start <= *end, "Start: {}, End: {}", start, end);
        }

        // Test that segments are contiguous
        for i in 0..result.len() - 1 {
            assert_eq!(result[i].1 + 1, result[i + 1].0);
        }

        // Testing with more segments than possible
        let crossbreed_segments = 15;

        // Act
        let result = generate_segments(primary_shape.clone(), secondary_shape.clone(), crossbreed_segments);

        println!("{:?}", result);

        //Assert that segments are within the bounds of the layers
        for (start, end) in result.iter() {
            // Bounds are the end of the first layer to the end of the second to last layer
            let bounds = 3..17;

            assert!(bounds.contains(end));
            assert!(start < &bounds.end);
        }

        //Assert that segments start and end are in ascending order
        for (start, end) in result.iter()
        {
            assert!(*start <= *end, "Start: {}, End: {}", start, end);
        }

        // Test that segments are contiguous
        for i in 0..result.len() - 1 {
            assert_eq!(result[i].1 + 1, result[i + 1].0);
        }
    }

    #[test]
    fn get_bias_neuron_for_layer_test() {
        // Assign
        let shape = vec![4, 8, 6, 4];

        // Act
        let result = get_bias_neuron_for_layer(0, &shape);

        // Assert
        assert_eq!(result, None);

        // Act
        let result = get_bias_neuron_for_layer(1, &shape);

        // Assert
        assert_eq!(result, Some(4));

        // Act
        let result = get_bias_neuron_for_layer(2, &shape);

        // Assert
        assert_eq!(result, Some(13));

        // Act
        let result = get_bias_neuron_for_layer(3, &shape);

        // Assert
        assert_eq!(result, Some(20));

        // Act
        let result = get_bias_neuron_for_layer(4, &shape);

        // Assert
        assert_eq!(result, None);
    }

    #[test]
    fn crossbreed_neuron_arrays_test() {
        // Assign
        let segments = vec![(0, 3), (4, 6), (7, 8), (9, 10)];
        
        let primary_network = generate_neuron_datastructure(&vec![4, 8, 6, 4]);

        let secondary_network = generate_neuron_datastructure(&vec![4, 3, 3, 3, 3, 3, 4]);

        // Act
        let result = crossbreed_neuron_arrays(segments.clone(), primary_network.clone(), secondary_network.clone());

        // Expected Result Set
        let expected: HashSet<(u32, bool, usize, u32)> = vec![
            // Input layer: Expect 4
            (0, true, 0, 0), (1, true, 0, 1), (2, true, 0, 2), (3, true, 0, 3),
            // Hidden Layer 1: Expect 8
            (4, false, 1, 4), (5, false, 1, 5), (6, false, 1, 6), (7, true, 1, 7), (8, true, 1, 8), (9, true, 1, 9), (10, true, 1, 10), (11, true, 1, 11),
            // Hidden Layer 2: Expect 9
            (7, false, 2, 12), (8, false, 2, 13), (9, false, 2, 14), (12, true, 2, 15), (13, true, 2, 16), (14, true, 2, 17), (15, true, 2, 18), (16, true, 2, 19), (17, true, 2, 20),
            // Output Layer: Expect 4
            (18, true, 3, 21), (19, true, 3, 22), (20, true, 3, 23), (21, true, 3, 24),
        ].into_iter().collect();

        // Convert Result to HashSet for Comparison
        let result_set: HashSet<(u32, bool, usize, u32)> = result.into_iter().collect();

        // Assert
        assert_eq!(result_set, expected);

        // Now we test the ooposite case
        // Act
        let result = crossbreed_neuron_arrays(segments.clone(), secondary_network.clone(), primary_network.clone());

        // Expected Result Set
        let expected: HashSet<(u32, bool, usize, u32)> = vec![
            // Input layer: Expect 4
            (0, true, 0, 0), (1, true, 0, 1), (2, true, 0, 2), (3, true, 0, 3),
            // Hidden Layer 1: Expect 7
            (4, false, 1, 4), (5, false, 1, 5), (6, false, 1, 6), (7, false, 1, 7), (8, false, 1, 8), (9, false, 1, 9), (10, false, 1, 10),
            // Hidden Layer 2: Expect 3
            (7, true, 2, 11), (8, true, 2, 12), (9, true, 2, 13),
            // Hidden Layer 3: Expect 3
            (10, true, 3, 14), (11, true, 3, 15), (12, true, 3, 16),
            // Hidden Layer 4: Expect 3
            (13, true, 4, 17), (14, true, 4, 18), (15, true, 4, 19),
            // Hidden Layer 5: Expect 3
            (16, true, 5, 20), (17, true, 5, 21), (18, true, 5, 22),
            // Output Layer: Expect 4
            (19, true, 6, 23), (20, true, 6, 24), (21, true, 6, 25), (22, true, 6, 26),
        ].into_iter().collect();

        // Convert Result to HashSet for Comparison
        let result_set: HashSet<(u32, bool, usize, u32)> = result.into_iter().collect();

        // Assert
        assert_eq!(result_set, expected);

        // Testing with a different segment
        // Assign
        let segments = vec![(0, 4), (5, 14), (15, 15), (16, 16)];

        // Act
        let result = crossbreed_neuron_arrays(segments.clone(), primary_network.clone(), secondary_network.clone());

        // Expected Result Set
        let expected: HashSet<(u32, bool, usize, u32)> = vec![
            // Input layer: Expect 4
            (0, true, 0, 0), (1, true, 0, 1), (2, true, 0, 2), (3, true, 0, 3),
            // Hidden Layer 1: Expect 3
            (4, true, 1, 4), (5, false, 1, 5), (6, false, 1, 6),
            // Hidden Layer 2: Expect 6
            (7, false, 2, 7), (8, false, 2, 8), (9, false, 2, 9), (15, true, 2, 10), (16, true, 2, 11), (17, true, 2, 12),
            // Hidden Layer 3: Expect 3
            (10, false, 3, 13), (11, false, 3, 14), (12, false, 3, 15),
            // Hidden Layer 4: Expect 3
            (13, false, 4, 16), (14, false, 4, 17), (15, false, 4, 18),
            // Output Layer: Expect 4
            (18, true, 5, 19), (19, true, 5, 20), (20, true, 5, 21), (21, true, 5, 22),
        ].into_iter().collect();

        // print result before comparison
        for r in result.iter() {
            println!("{:?}", r);
        }

        // Convert Result to HashSet for Comparison
        let result_set: HashSet<(u32, bool, usize, u32)> = result.into_iter().collect();

        // Assert
        assert_eq!(result_set, expected);

        // Swapping order
        let result = crossbreed_neuron_arrays(segments.clone(), secondary_network.clone(), primary_network.clone());

        // Expected Result Set
        let expected: HashSet<(u32, bool, usize, u32)> = vec![
            // Input layer: Expect 4
            (0, true, 0, 0), (1, true, 0, 1), (2, true, 0, 2), (3, true, 0, 3),
            // Hidden Layer 1: Expect 8
            (4, true, 1, 4), (5, false, 1, 5), (6, false, 1, 6), (7, false, 1, 7), (8, false, 1, 8), (9, false, 1, 9), (10, false, 1, 10), (11, false, 1, 11),
            // Hidden Layer 2: Expect 5
            (12, false, 2, 12), (13, false, 2, 13), (14, false, 2, 14), (15, false, 2, 15), (16, false, 2, 16),
            // Hidden Layer 3: Expect 3
            (13, true, 3, 17), (14, true, 3, 18), (15, true, 3, 19),
            // Hidden Layer 4: Expect 3
            (16, true, 4, 20), (17, true, 4, 21), (18, true, 4, 22),
            // Output Layer: Expect 4
            (19, true, 5, 23), (20, true, 5, 24), (21, true, 5, 25), (22, true, 5, 26),
        ].into_iter().collect();

        // print result before comparison
        for r in result.iter() {
            println!("{:?}", r);
        }

        // Convert Result to HashSet for Comparison
        let result_set: HashSet<(u32, bool, usize, u32)> = result.into_iter().collect();

        // Assert
        assert_eq!(result_set, expected);

        // Testing with a different segment
        // Assign
        let segments = vec![(0, 7), (8, 9), (10, 10), (11, 12)];

        // Act
        let result = crossbreed_neuron_arrays(segments.clone(), primary_network.clone(), secondary_network.clone());

        // Expected Result Set
        let expected: HashSet<(u32, bool, usize, u32)> = vec![
            // Input layer: Expect 4
            (0, true, 0, 0), (1, true, 0, 1), (2, true, 0, 2), (3, true, 0, 3),
            // Hidden Layer 1: Expect 7
            (4, true, 1, 4), (5, true, 1, 5), (6, true, 1, 6), (7, true, 1, 7), (8, true, 1, 8), (9, true, 1, 9), (10, true, 1, 10),
            // Hidden Layer 2: Expect 8
            (7, false, 2, 11), (8, false, 2, 12), (9, false, 2, 13), (13, true, 2, 14), (14, true, 2, 15), (15, true, 2, 16), (16, true, 2, 17), (17, true, 2, 18),
            // Hidden Layer 3: Expect 3
            (10, false, 3, 19), (11, false, 3, 20), (12, false, 3, 21),
            // Output Layer: Expect 4
            (18, true, 4, 22), (19, true, 4, 23), (20, true, 4, 24), (21, true, 4, 25),
        ].into_iter().collect();

        // print result before comparison
        for r in result.iter() {
            println!("{:?}", r);
        }

        // Convert Result to HashSet for Comparison
        let result_set: HashSet<(u32, bool, usize, u32)> = result.into_iter().collect();

        // Assert
        assert_eq!(result_set, expected);

        // Swapping order
        let result = crossbreed_neuron_arrays(segments.clone(), secondary_network.clone(), primary_network.clone());

        // Expected Result Set
        let expected: HashSet<(u32, bool, usize, u32)> = vec![
            // Input layer: Expect 4
            (0, true, 0, 0), (1, true, 0, 1), (2, true, 0, 2), (3, true, 0, 3),
            // Hidden Layer 1: Expect 7
            (4, true, 1, 4), (5, true, 1, 5), (6, true, 1, 6), (8, false, 1, 7), (9, false, 1, 8), (10, false, 1, 9), (11, false, 1, 10),
            // Hidden Layer 2: Expect 4
            (7, true, 2, 11), (8, true, 2, 12), (9, true, 2, 13), (12, false, 2, 14),
            // Hidden Layer 3: Expect 3
            (10, true, 3, 15), (11, true, 3, 16), (12, true, 3, 17),
            // Hidden Layer 4: Expect 3
            (13, true, 4, 18), (14, true, 4, 19), (15, true, 4, 20),
            // Hidden Layer 5: Expect 3
            (16, true, 5, 21), (17, true, 5, 22), (18, true, 5, 23),
            // Output Layer: Expect 4
            (19, true, 6, 24), (20, true, 6, 25), (21, true, 6, 26), (22, true, 6, 27),
        ].into_iter().collect();

        // print result before comparison
        for r in result.iter() {
            println!("{:?}", r);
        }

        // Convert Result to HashSet for Comparison
        let result_set: HashSet<(u32, bool, usize, u32)> = result.into_iter().collect();

        // Assert
        assert_eq!(result_set, expected);

        // Testing networks with the same size
        // Assign
        let segments = vec![(0, 3), (4, 6), (7, 8), (9, 11)];
        
        let primary_network = generate_neuron_datastructure(&vec![4, 3, 4, 5, 4]);
        
        vec![
            // Input layer
            (0, 0), (1, 0), (2, 0), (3, 0),
            // Hidden layer 1: 3 neurons
            (4, 1), (5, 1), (6, 1),
            // Hidden Layer 2: 4 neurons
            (7, 2), (8, 2), (9, 2), (10, 2),
            // Hidden Layer 3: 5 neurons
            (11, 3), (12, 3), (13, 3), (14, 3), (15, 3),
            // Output layer
            (16, 4), (17, 4), (18, 4), (19, 4),
        ];

        let secondary_network = primary_network.clone();

        // Act
        let result = crossbreed_neuron_arrays(segments.clone(), primary_network.clone(), secondary_network.clone());

        // Expected Result Set
        let expected: HashSet<(u32, bool, usize, u32)> = vec![
            // Input layer: Expect 4
            (0, true, 0, 0), (1, true, 0, 1), (2, true, 0, 2), (3, true, 0, 3),
            // Hidden Layer 1: Expect 3
            (4, false, 1, 4), (5, false, 1, 5), (6, false, 1, 6),
            // Hidden Layer 2: Expect 4
            (7, true, 2, 7), (8, true, 2, 8), (9, false, 2, 9), (10, false, 2, 10),
            // Hidden Layer 3: Expect 5
            (11, false, 3, 11), (12, true, 3, 12), (13, true, 3, 13), (14, true, 3, 14), (15, true, 3, 15),
            // Output Layer: Expect 4
            (16, true, 4, 16), (17, true, 4, 17), (18, true, 4, 18), (19, true, 4, 19),
        ].into_iter().collect();

        // print result before comparison
        for r in result.iter() {
            println!("{:?}", r);
        }

        // Convert Result to HashSet for Comparison
        let result_set: HashSet<(u32, bool, usize, u32)> = result.into_iter().collect();

        // Assert
        assert_eq!(result_set, expected);

        // Testing with different segment
        let segments = vec![(0, 5), (6, 6), (7, 11), (12, 13)];

        // Act
        let result = crossbreed_neuron_arrays(segments.clone(), primary_network.clone(), secondary_network.clone());

        // Expected Result Set
        let expected: HashSet<(u32, bool, usize, u32)> = vec![
            // Input layer: Expect 4
            (0, true, 0, 0), (1, true, 0, 1), (2, true, 0, 2), (3, true, 0, 3),
            // Hidden Layer 1: Expect 3
            (4, true, 1, 4), (5, true, 1, 5), (6, false, 1, 6),
            // Hidden Layer 2: Expect 4
            (7, true, 2, 7), (8, true, 2, 8), (9, true, 2, 9), (10, true, 2, 10),
            // Hidden Layer 3: Expect 5
            (11, true, 3, 11), (12, false, 3, 12), (13, false, 3, 13), (14, true, 3, 14), (15, true, 3, 15),
            // Output Layer: Expect 4
            (16, true, 4, 16), (17, true, 4, 17), (18, true, 4, 18), (19, true, 4, 19),
        ].into_iter().collect();

        // print result before comparison
        for r in result.iter() {
            println!("{:?}", r);
        }

        // Convert Result to HashSet for Comparison
        let result_set: HashSet<(u32, bool, usize, u32)> = result.into_iter().collect();

        // Assert
        assert_eq!(result_set, expected);
    }

    #[test]
    fn generate_neuron_datastructure_test() {
        // Assign
        let shape = vec![4, 3, 5, 4];

        // Act
        let result = generate_neuron_datastructure(shape.as_slice());

        // Expected Result
        let expected: Vec<(u32, usize)> = vec![
            (0, 0), (1, 0), (2, 0), (3, 0),
            (4, 1), (5, 1), (6, 1),
            (7, 2), (8, 2), (9, 2), (10, 2), (11, 2),
            (12, 3), (13, 3), (14, 3), (15, 3),
        ];

        // Assert
        assert_eq!(result, expected);
    }

    #[test]
    fn translate_neuron_id_test() {
        // Assign
        let shape = vec![4, 3, 5, 4];

        let expected = vec![
            // (input, expected output)
            (0, 0), (1, 1), (2, 2), (3, 3),
            (4, 5), (5, 6), (6, 7),
            (7, 9), (8, 10), (9, 11), (10, 12), (11, 13),
            (12, 15), (13, 16), (14, 17), (15, 18),
        ];

        // Act
        for (input, expected_output) in expected {
            let result = to_bias_network_id(&input, &shape);
            // Assert
            assert_eq!(result, expected_output);

            // Go the other direction too
            let result = to_non_bias_network_id(expected_output, &shape);

            // Assert
            if let Some(result) = result {
                assert_eq!(result, input);
            }
            else {
                assert!(false, "Expected Some, got None");
            }
        }

        // Validate bias neuron values
        let bias_neurons = vec![4, 8, 14, 19];

        for &bias_neuron in bias_neurons.iter() {
            let result = to_non_bias_network_id(bias_neuron, &shape);

            // Assert
            assert!(result.is_none());
        }
    }

    #[test]
    fn consolidate_old_connections_test() -> Result<(), Box<dyn std::error::Error>>{
        // Assign
        let primary_shape = vec![4, 8, 6, 4];
        let secondary_shape = vec![4, 3, 3, 3, 3, 3, 4];

        let mut primary_fann = Fann::new(&primary_shape)?;
        let mut secondary_fann = Fann::new(&secondary_shape)?;

        let mut primary_connections = primary_fann.get_connections();
        for connection in primary_connections.iter_mut() {
            connection.weight = ((connection.from_neuron * 100) + connection.to_neuron) as f32;
        }
        primary_fann.set_connections(&primary_connections);

        let mut secondary_connections = secondary_fann.get_connections();
        for connection in secondary_connections.iter_mut() {
            connection.weight = ((connection.from_neuron * 100) + connection.to_neuron) as f32;
            connection.weight = connection.weight * -1.0;
        }
        secondary_fann.set_connections(&secondary_connections);

        let new_neurons = vec![
            // Input layer: Expect 4
            (0, true, 0, 0), (1, true, 0, 1), (2, true, 0, 2), (3, true, 0, 3),
            // Hidden Layer 1: Expect 8
            (4, false, 1, 4), (5, false, 1, 5), (6, false, 1, 6), (7, true, 1, 7), (8, true, 1, 8), (9, true, 1, 9), (10, true, 1, 10), (11, true, 1, 11),
            // Hidden Layer 2: Expect 9
            (7, false, 2, 12), (8, false, 2, 13), (9, false, 2, 14), (12, true, 2, 15), (13, true, 2, 16), (14, true, 2, 17), (15, true, 2, 18), (16, true, 2, 19), (17, true, 2, 20),
            // Output Layer: Expect 4
            (18, true, 3, 21), (19, true, 3, 22), (20, true, 3, 23), (21, true, 3, 24),
        ];
        let new_shape = vec![4, 8, 9, 4];
        let mut new_fann = Fann::new(&[4, 8, 9, 4])?;
        // Initialize weights to 0
        let mut new_connections = new_fann.get_connections();
        for connection in new_connections.iter_mut() {
            connection.weight = 0.0;
        }
        new_fann.set_connections(&new_connections);

        // Act
        consolidate_old_connections(&primary_fann, &secondary_fann, new_shape, new_neurons, &mut new_fann);

        // Bias neurons
        // Layer 1: 4
        // Layer 2: 13
        // Layer 3: 23
        let expected_connections = vec![
            // (from_neuron, to_neuron, weight)
            // Hidden Layer 1 (5-12)
            (0, 5, -5.0), (1, 5, -105.0), (2, 5, -205.0), (3, 5, -305.0),
            (0, 6, -6.0), (1, 6, -106.0), (2, 6, -206.0), (3, 6, -306.0),
            (0, 7, -7.0), (1, 7, -107.0), (2, 7, -207.0), (3, 7, -307.0),
            (0, 8, 8.0), (1, 8, 108.0), (2, 8, 208.0), (3, 8, 308.0),
            (0, 9, 9.0), (1, 9, 109.0), (2, 9, 209.0), (3, 9, 309.0),
            (0, 10, 10.0), (1, 10, 110.0), (2, 10, 210.0), (3, 10, 310.0),
            (0, 11, 11.0), (1, 11, 111.0), (2, 11, 211.0), (3, 11, 311.0),
            (0, 12, 12.0), (1, 12, 112.0), (2, 12, 212.0), (3, 12, 312.0),
            // Hidden Layer 2 (14-22)
            (5, 14, -509.0), (6, 14, -609.0), (7, 14, -709.0), (8, 14, 0.0), (9, 14, 0.0), (10, 14, 0.0), (11, 14, 0.0), (12, 14, 0.0),
            (5, 15, -510.0), (6, 15, -610.0), (7, 15, -710.0), (8, 15, 0.0), (9, 15, 0.0), (10, 15, 0.0), (11, 15, 0.0), (12, 15, 0.0),
            (5, 16, -511.0), (6, 16, -611.0), (7, 16, -711.0), (8, 16, 0.0), (9, 16, 0.0), (10, 16, 0.0), (11, 16, 0.0), (12, 16, 0.0),
            (5, 17, 514.0), (6, 17, 614.0), (7, 17, 714.0), (8, 17, 814.0), (9, 17, 914.0), (10, 17, 1014.0), (11, 17, 1114.0), (12, 17, 1214.0),
            (5, 18, 515.0), (6, 18, 615.0), (7, 18, 715.0), (8, 18, 815.0), (9, 18, 915.0), (10, 18, 1015.0), (11, 18, 1115.0), (12, 18, 1215.0),
            (5, 19, 516.0), (6, 19, 616.0), (7, 19, 716.0), (8, 19, 816.0), (9, 19, 916.0), (10, 19, 1016.0), (11, 19, 1116.0), (12, 19, 1216.0),
            (5, 20, 517.0), (6, 20, 617.0), (7, 20, 717.0), (8, 20, 817.0), (9, 20, 917.0), (10, 20, 1017.0), (11, 20, 1117.0), (12, 20, 1217.0),
            (5, 21, 518.0), (6, 21, 618.0), (7, 21, 718.0), (8, 21, 818.0), (9, 21, 918.0), (10, 21, 1018.0), (11, 21, 1118.0), (12, 21, 1218.0),
            (5, 22, 519.0), (6, 22, 619.0), (7, 22, 719.0), (8, 22, 819.0), (9, 22, 919.0), (10, 22, 1019.0), (11, 22, 1119.0), (12, 22, 1219.0),
            // Output layer (24-27)
            (14, 24, 0.0), (15, 24, 0.0), (16, 24, 0.0), (17, 24, 1421.0), (18, 24, 1521.0), (19, 24, 1621.0), (20, 24, 1721.0), (21, 24, 1821.0), (22, 24, 1921.0),
            (14, 25, 0.0), (15, 25, 0.0), (16, 25, 0.0), (17, 25, 1422.0), (18, 25, 1522.0), (19, 25, 1622.0), (20, 25, 1722.0), (21, 25, 1822.0), (22, 25, 1922.0),
            (14, 26, 0.0), (15, 26, 0.0), (16, 26, 0.0), (17, 26, 1423.0), (18, 26, 1523.0), (19, 26, 1623.0), (20, 26, 1723.0), (21, 26, 1823.0), (22, 26, 1923.0),
            (14, 27, 0.0), (15, 27, 0.0), (16, 27, 0.0), (17, 27, 1424.0), (18, 27, 1524.0), (19, 27, 1624.0), (20, 27, 1724.0), (21, 27, 1824.0), (22, 27, 1924.0),
        ];

        for connection in new_fann.get_connections().iter() {
            println!("{:?}", connection);
        }

        // Assert
        // Compare each connection to the expected connection
        let new_connections = new_fann.get_connections();
        for connection in expected_connections.iter() {
            let matching_connection = new_connections.iter().find(|&c| c.from_neuron == connection.0 && c.to_neuron == connection.1);
            if let Some(matching_connection) = matching_connection {
                assert_eq!(matching_connection.weight, connection.2, "Connection: {:?}", matching_connection);
            } else {
                assert!(false, "Connection not found: {:?}", connection);
            }
        }

        let expected_bias_neuron_connections = vec![
            // (from_neuron, to_neuron, weight)
            // Bias Neurons
            // Layer 2: bias neuron_id 4
            (4, 5, -405.0), (4, 6, -406.0), (4, 7, -407.0), (4, 8, 408.0), (4, 9, 409.0), (4, 10, 410.0), (4, 11, 411.0), (4, 12, 412.0),
            // Layer 3: bias neuron_id 13
            (13, 14, -809.0), (13, 15, -810.0), (13, 16, -811.0), (13, 17, 1314.0), (13, 18, 1315.0), (13, 19, 1316.0), (13, 20, 1317.0), (13, 21, 1318.0), (13, 22, 1319.0),
            // Layer 4: bias neuron_id 23
            (23, 24, 2021.0), (23, 25, 2022.0), (23, 26, 2023.0), (23, 27, 2024.0),
        ];

        for connection in expected_bias_neuron_connections.iter() {
            let matching_connection = new_connections.iter().find(|&c| c.from_neuron == connection.0 && c.to_neuron == connection.1);
            if let Some(matching_connection) = matching_connection {
                assert_eq!(matching_connection.weight, connection.2, "Connection: {:?}", matching_connection);
            } else {
                assert!(false, "Connection not found: {:?}", connection);
            }
        }

        Ok(())
    }
}