extern crate fann;

use std::{cmp::min, fs::{self, File}, io::{self, BufRead, BufReader}, path::{Path, PathBuf}, sync::Arc};
use fann::{ActivationFunc, Fann};
use futures::future::join_all;
use gemla::{core::genetic_node::{GeneticNode, GeneticNodeContext}, error::Error};
use rand::prelude::*;
use rand::distributions::{Distribution, Uniform};
use serde::{Deserialize, Serialize};
use anyhow::Context;
use tokio::process::Command;
use uuid::Uuid;
use std::collections::HashMap;
use async_trait::async_trait;

const BASE_DIR: &str = "F:\\\\vandomej\\Projects\\dootcamp-AI-Simulation\\Simulations";
const POPULATION: usize = 50;

const NEURAL_NETWORK_INPUTS: usize = 14;
const NEURAL_NETWORK_OUTPUTS: usize = 8;
const NEURAL_NETWORK_HIDDEN_LAYERS_MIN: usize = 1;
const NEURAL_NETWORK_HIDDEN_LAYERS_MAX: usize = 10;
const NEURAL_NETWORK_HIDDEN_LAYER_SIZE_MIN: usize = 3;
const NEURAL_NETWORK_HIDDEN_LAYER_SIZE_MAX: usize = 35;
const NEURAL_NETWORK_INITIAL_WEIGHT_MIN: f32 = -2.0;
const NEURAL_NETWORK_INITIAL_WEIGHT_MAX: f32 = 2.0;
const NEURAL_NETWORK_CROSSBREED_SEGMENTS_MIN: usize = 2;
const NEURAL_NETWORK_CROSSBREED_SEGMENTS_MAX: usize = 20;

const SIMULATION_ROUNDS: usize = 5;
const SURVIVAL_RATE: f32 = 0.5;
const GAME_EXECUTABLE_PATH: &str = "F:\\\\vandomej\\Projects\\dootcamp-AI-Simulation\\Package\\Windows\\AI_Fight_Sim.exe";

// Here is the folder structure for the FighterNN:
// base_dir/fighter_nn_{fighter_id}/{generation}/{fighter_id}_fighter_nn_{nn_id}.net

// A neural network that utilizes the fann library to save and read nn's from files
// FighterNN contains a list of file locations for the nn's stored, all of which are stored under the same folder which is also contained. 
// there is no training happening to the neural networks
// the neural networks are only used to simulate the nn's and to save and read the nn's from files
// Filenames are stored in the format of "{fighter_id}_fighter_nn_{generation}.net".
// The main folder contains a subfolder for each generation, containing a population of 10 nn's

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FighterNN {
    pub id: Uuid,
    pub folder: PathBuf,
    pub population_size: usize,
    pub generation: u64,
    // A map of each nn identifier in a generation and their physics score
    pub scores: Vec<HashMap<u64, f32>>,
    // A map of the id of the nn in the current generation and their neural network shape
    pub nn_shapes: HashMap<u64, Vec<u32>>,
    pub crossbreed_segments: usize
}

#[async_trait]
impl GeneticNode for FighterNN {
    // Check for the highest number of the folder name and increment it by 1
    fn initialize(context: GeneticNodeContext) -> Result<Box<Self>, Error> {
        let base_path = PathBuf::from(BASE_DIR);
    
        let folder = base_path.join(format!("fighter_nn_{:06}", context.id));
        // Ensures directory is created if it doesn't exist and does nothing if it exists
        fs::create_dir_all(&folder)
            .with_context(|| format!("Failed to create or access the folder: {:?}", folder))?;
    
        //Create a new directory for the first generation, using create_dir_all to avoid errors if it already exists
        let gen_folder = folder.join("0");
        fs::create_dir_all(&gen_folder)
            .with_context(|| format!("Failed to create or access the generation folder: {:?}", gen_folder))?;

        let mut nn_shapes = HashMap::new();
    
        // Create the first generation in this folder
        for i in 0..POPULATION {
            // Filenames are stored in the format of "xxxxxx_fighter_nn_0.net", "xxxxxx_fighter_nn_1.net", etc. Where xxxxxx is the folder name
            let nn = gen_folder.join(format!("{:06}_fighter_nn_{}.net", context.id, i));

            // Randomly generate a neural network shape based on constants
            let hidden_layers = thread_rng().gen_range(NEURAL_NETWORK_HIDDEN_LAYERS_MIN..NEURAL_NETWORK_HIDDEN_LAYERS_MAX);
            let mut nn_shape = vec![NEURAL_NETWORK_INPUTS as u32];
            for _ in 0..hidden_layers {
                nn_shape.push(thread_rng().gen_range(NEURAL_NETWORK_HIDDEN_LAYER_SIZE_MIN..NEURAL_NETWORK_HIDDEN_LAYER_SIZE_MAX) as u32);
            }
            nn_shape.push(NEURAL_NETWORK_OUTPUTS as u32);
            nn_shapes.insert(i as u64, nn_shape.clone());

            let mut fann = Fann::new(nn_shape.as_slice())
                .with_context(|| "Failed to create nn")?;
            fann.randomize_weights(thread_rng().gen_range(NEURAL_NETWORK_INITIAL_WEIGHT_MIN..0.0), thread_rng().gen_range(0.0..=NEURAL_NETWORK_INITIAL_WEIGHT_MAX));
            fann.set_activation_func_hidden(ActivationFunc::SigmoidSymmetric);
            fann.set_activation_func_output(ActivationFunc::SigmoidSymmetric);
            // This will overwrite any existing file with the same name
            fann.save(&nn)
                .with_context(|| format!("Failed to save nn at {:?}", nn))?;
        }

        let mut crossbreed_segments = thread_rng().gen_range(NEURAL_NETWORK_CROSSBREED_SEGMENTS_MIN..NEURAL_NETWORK_CROSSBREED_SEGMENTS_MAX);
        if crossbreed_segments % 2 != 0 {
            crossbreed_segments += 1;
        }
    
        Ok(Box::new(FighterNN {
            id: context.id,
            folder,
            population_size: POPULATION,
            generation: 0,
            scores: vec![HashMap::new()],
            nn_shapes,
            // we need crossbreed segments to be even
            crossbreed_segments,
        }))
    }

    async fn simulate(&mut self, context: GeneticNodeContext) -> Result<(), Error> {
        trace!("Context: {:?}", context);
        let mut tasks = Vec::new();

        // For each nn in the current generation:
        for i in 0..self.population_size {
            let self_clone = self.clone();
            let semaphore_clone = Arc::clone(context.semaphore.as_ref().unwrap());

            let task = async move {
                let nn = self_clone.folder.join(format!("{}", self_clone.generation)).join(format!("{:06}_fighter_nn_{}.net", self_clone.id, i));
                let mut simulations = Vec::new();
        
                // Using the same original nn, repeat the simulation with 5 random nn's from the current generation concurrently
                for _ in 0..SIMULATION_ROUNDS {
                    let random_nn_index = thread_rng().gen_range(0..self_clone.population_size);
                    let id = self_clone.id.clone();
                    let folder = self_clone.folder.clone();
                    let generation = self_clone.generation;
                    let semaphore_clone = Arc::clone(&semaphore_clone);

                    let random_nn = folder.join(format!("{}", generation)).join(format!("{:06}_fighter_nn_{}.net", id, random_nn_index));
                    let nn_clone = nn.clone(); // Clone the path to use in the async block
        
                    let config1_arg = format!("-NN1Config=\"{}\"", nn_clone.to_str().unwrap());
                    let config2_arg = format!("-NN2Config=\"{}\"", random_nn.to_str().unwrap());
                    let disable_unreal_rendering_arg = "-nullrhi".to_string();

                    
        
                    let future = async move {
                        let permit = semaphore_clone.acquire_owned().await.with_context(|| "Failed to acquire semaphore permit")?;

                        // Construct the score file path
                        let nn_id = format!("{:06}_fighter_nn_{}", id, i);
                        let random_nn_id = format!("{:06}_fighter_nn_{}", id, random_nn_index);
                        let score_file_name = format!("{}_vs_{}.txt", nn_id, random_nn_id);
                        let score_file = folder.join(format!("{}", generation)).join(&score_file_name);

                        // Check if score file already exists before running the simulation
                        if score_file.exists() {
                            let round_score = read_score_from_file(&score_file, &nn_id).await
                                .with_context(|| format!("Failed to read score from file: {:?}", score_file_name))?;

                            trace!("{} scored {}", nn_id, round_score);

                            return Ok::<f32, Error>(round_score);
                        }

                        // Check if the opposite round score has been determined
                        let opposite_score_file = folder.join(format!("{}", generation)).join(format!("{}_vs_{}.txt", random_nn_id, nn_id));
                        if opposite_score_file.exists() {
                            let round_score = read_score_from_file(&opposite_score_file, &nn_id).await
                                .with_context(|| format!("Failed to read score from file: {:?}", opposite_score_file))?;

                            trace!("{} scored {}", nn_id, round_score);

                            return Ok::<f32, Error>(1.0 - round_score);
                        }

                        // Run simulation until score file is generated
                        while !score_file.exists() {
                            let _output = if thread_rng().gen_range(0..100) < 1 {
                                Command::new(GAME_EXECUTABLE_PATH)
                                    .arg(&config1_arg)
                                    .arg(&config2_arg)
                                    .output()
                                    .await
                                    .expect("Failed to execute game")
                            } else {
                                Command::new(GAME_EXECUTABLE_PATH)
                                    .arg(&config1_arg)
                                    .arg(&config2_arg)
                                    .arg(&disable_unreal_rendering_arg)
                                    .output()
                                    .await
                                    .expect("Failed to execute game")
                            };
                        }

                        drop(permit);

                        // Read the score from the file
                        if score_file.exists() {
                            let round_score = read_score_from_file(&score_file, &nn_id).await
                                .with_context(|| format!("Failed to read score from file: {:?}", score_file_name))?;

                            trace!("{} scored {}", nn_id, round_score);

                            Ok(round_score)
                        } else {
                            trace!("Score file not found: {:?}", score_file_name);
                            Ok(0.0)
                        }
                    };
        
                    simulations.push(future);
                }
        
                // Wait for all simulation rounds to complete
                let results: Result<Vec<f32>, Error> = join_all(simulations).await.into_iter().collect();
        
                let score = match results {
                    Ok(scores) => scores.into_iter().sum::<f32>() / SIMULATION_ROUNDS as f32,
                    Err(e) => return Err(e), // Return the error if results collection failed
                };
                trace!("NN {:06}_fighter_nn_{} scored {}", self_clone.id, i, score);
                Ok((i, score))
            };

            tasks.push(task);
        }

        let results = join_all(tasks).await;

        for result in results {
            match result {
                Ok((index, score)) => {
                    // Update the original `self` object with the score.
                    self.scores[self.generation as usize].insert(index as u64, score);
                },
                Err(e) => {
                    // Handle task panic or execution error
                    return Err(Error::Other(anyhow::anyhow!(format!("Task failed: {:?}", e))));
                },
            }
        }
    
        Ok(())
    }


    fn mutate(&mut self, _context: GeneticNodeContext) -> Result<(), Error> {
        let survivor_count = (self.population_size as f32 * SURVIVAL_RATE) as usize;

        // Create the new generation folder
        let new_gen_folder = self.folder.join(format!("{}", self.generation + 1));
        fs::create_dir_all(&new_gen_folder).with_context(|| format!("Failed to create or access new generation folder: {:?}", new_gen_folder))?;

        // Remove the 5 nn's with the lowest scores
        let mut sorted_scores: Vec<_> = self.scores[self.generation as usize].iter().collect();
        sorted_scores.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());
        let to_keep = sorted_scores[survivor_count..].iter().map(|(k, _)| *k).collect::<Vec<_>>();

        // Save the remaining 5 nn's to the new generation folder
        for i in 0..survivor_count {
            let nn_id = to_keep[i];
            let nn = self.folder.join(format!("{}", self.generation)).join(format!("{:06}_fighter_nn_{}.net", self.id, nn_id));
            let new_nn = new_gen_folder.join(format!("{:06}_fighter_nn_{}.net", self.id, i));
            fs::copy(&nn, &new_nn)?;
        }

        // Take the remaining 5 nn's and create 5 new nn's by the following:
        for i in 0..survivor_count {
            let nn_id = to_keep[i];
            let nn = self.folder.join(format!("{}", self.generation)).join(format!("{:06}_fighter_nn_{}.net", self.id, nn_id));
            let fann = Fann::from_file(&nn)
                .with_context(|| format!("Failed to load nn"))?;

            // Load another nn from the current generation and cross breed it with the current nn
            let cross_nn = self.folder.join(format!("{}", self.generation)).join(format!("{:06}_fighter_nn_{}.net", self.id, to_keep[thread_rng().gen_range(0..survivor_count)]));
            let cross_fann = Fann::from_file(&cross_nn)
                .with_context(|| format!("Failed to load cross nn"))?;

            let mut new_fann = crossbreed(&fann, &cross_fann, self.crossbreed_segments)?;

            // For each weight in the 5 new nn's there is a 20% chance of a minor mutation (a random number between -0.1 and 0.1 is added to the weight)
            // And a 5% chance of a major mutation (a random number between -0.3 and 0.3 is added to the weight)
            let mut connections = new_fann.get_connections(); // Vector of connections
            for c in &mut connections {
                if thread_rng().gen_range(0..100) < 20 {
                    c.weight += thread_rng().gen_range(-0.1..0.1);
                }
                else if thread_rng().gen_range(0..100) < 5 {
                    c.weight += thread_rng().gen_range(-0.3..0.3);
                }
            }
            new_fann.set_connections(&connections);

            // Save the new nn's to the new generation folder
            let new_nn = new_gen_folder.join(format!("{:06}_fighter_nn_{}.net", self.id, i + survivor_count));
            fann.save(&new_nn)
                .with_context(|| format!("Failed to save nn"))?;
        }

        self.generation += 1;
        self.scores.push(HashMap::new());

        Ok(())
    }

    fn merge(left: &FighterNN, right: &FighterNN, id: &Uuid) -> Result<Box<FighterNN>, Error> {
        let base_path = PathBuf::from(BASE_DIR);
        let folder = base_path.join(format!("fighter_nn_{:06}", id));
    
        // Ensure the folder exists, including the generation subfolder.
        fs::create_dir_all(&folder.join("0"))
            .with_context(|| format!("Failed to create directory {:?}", folder.join("0")))?;
    
        // Function to copy NNs from a source FighterNN to the new folder.
        let copy_nns = |source: &FighterNN, folder: &PathBuf, id: &Uuid, start_idx: usize| -> Result<(), Error> {
            let mut sorted_scores: Vec<_> = source.scores[source.generation as usize].iter().collect();
            sorted_scores.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());
            let remaining = sorted_scores[(source.population_size / 2)..].iter().map(|(k, _)| *k).collect::<Vec<_>>();
    
            for (i, nn_id) in remaining.into_iter().enumerate() {
                let nn_path = source.folder.join(source.generation.to_string()).join(format!("{:06}_fighter_nn_{}.net", source.id, nn_id));
                let new_nn_path = folder.join("0").join(format!("{:06}_fighter_nn_{}.net", id, start_idx + i));
                fs::copy(&nn_path, &new_nn_path)
                    .with_context(|| format!("Failed to copy nn from {:?} to {:?}", nn_path, new_nn_path))?;
            }
            Ok(())
        };
    
        // Copy the top half of NNs from each parent to the new folder.
        copy_nns(left, &folder, id, 0)?;
        copy_nns(right, &folder, id, left.population_size as usize / 2)?;
    
        Ok(Box::new(FighterNN {
            id: *id,
            folder,
            generation: 0,
            population_size: left.population_size, // Assuming left and right have the same population size.
            scores: vec![HashMap::new()],
            crossbreed_segments: left.crossbreed_segments,
            nn_shapes: HashMap::new(),
        }))
    }
}

/// Crossbreeds two neural networks of different shapes by finding cut points, and swapping neurons between the two networks.
/// Algorithm tries to ensure similar functionality is maintained between the two networks.
/// It does this by preserving connections between the same neurons from the original to the new network, and if a connection cannot be found
/// it will create a new connection with a random weight.
fn crossbreed(primary: &Fann, secondary: &Fann, crossbreed_segments: usize) -> Result<Fann, Error> {
    // First we need to get the shape of the networks and transform this into a format that is easier to work with
    // We want a list of every neuron id, and the layer it is in
    let primary_shape = primary.get_layer_sizes();
    let secondary_shape = secondary.get_layer_sizes();
    let primary_neurons = generate_neuron_datastructure(&primary_shape);
    let secondary_neurons = generate_neuron_datastructure(&secondary_shape);

    // Now we need to find the cut points for the crossbreed
    let start = primary_shape[0] + 1; // Start at the first hidden layer
    let end = min(primary_shape.iter().sum::<u32>() - primary_shape.last().unwrap(), secondary_shape.iter().sum::<u32>() - secondary_shape.last().unwrap()); // End at the last hidden layer
    let segment_distribution = Uniform::from(start..end); // Ensure segments are not too small

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
        segments.push((previous, cut_point));
        previous = cut_point;
    }
    
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
    new_fann.randomize_weights(-0.1, 0.1);
    new_fann.set_activation_func_hidden(ActivationFunc::SigmoidSymmetric);
    new_fann.set_activation_func_output(ActivationFunc::SigmoidSymmetric);

    consolidate_old_connections(primary, secondary, new_shape, new_neurons, &mut new_fann);

    Ok(new_fann)
}

fn consolidate_old_connections(primary: &Fann, secondary: &Fann, new_shape: Vec<u32>, new_neurons: Vec<(u32, bool, usize, u32)>, new_fann: &mut Fann) {
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
                let mut connection ;
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
    }
}

fn crossbreed_neuron_arrays(segments: Vec<(u32, u32)>, primary_neurons: Vec<(u32, usize)>, secondary_neurons: Vec<(u32, usize)>) -> Vec<(u32, bool, usize, u32)> {
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

fn generate_neuron_datastructure(shape: &[u32]) -> Vec<(u32, usize)> {
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


async fn read_score_from_file(file_path: &Path, nn_id: &str) -> Result<f32, io::Error> {
    let mut attempts = 0;

    loop {
        match File::open(file_path) {
            Ok(file) => {
                let reader = BufReader::new(file);

                for line in reader.lines() {
                    let line = line?;
                    if line.starts_with(nn_id) {
                        let parts: Vec<&str> = line.split(':').collect();
                        if parts.len() == 2 {
                            return parts[1].trim().parse::<f32>().map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e));
                        }
                    }
                }

                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    "NN ID not found in scores file",
                ));
            },
            Err(e) if e.kind() == io::ErrorKind::WouldBlock || e.kind() == io::ErrorKind::PermissionDenied || e.kind() == io::ErrorKind::Other => {
                if attempts >= 5 { // Attempt 5 times before giving up.
                    return Err(e);
                }

                attempts += 1;
                // wait 1 second to ensure the file is written
                tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
            },
            Err(e) => return Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

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

        Ok(())
    }
}