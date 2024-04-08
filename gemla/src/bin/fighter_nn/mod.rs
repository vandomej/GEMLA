extern crate fann;

pub mod fighter_context;
pub mod neural_network_utility;

use anyhow::Context;
use async_trait::async_trait;
use fann::{ActivationFunc, Fann};
use futures::future::join_all;
use gemla::{
    core::genetic_node::{GeneticNode, GeneticNodeContext},
    error::Error,
};
use lerp::Lerp;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::{
    cmp::max,
    fs::{self, File},
    io::{self, BufRead, BufReader},
    ops::Range,
    path::{Path, PathBuf},
};
use tokio::process::Command;
use uuid::Uuid;

use self::neural_network_utility::{crossbreed, major_mutation};

const BASE_DIR: &str = "F:\\\\vandomej\\Projects\\dootcamp-AI-Simulation\\Simulations";
const POPULATION: usize = 50;

const NEURAL_NETWORK_INPUTS: usize = 18;
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
const GAME_EXECUTABLE_PATH: &str =
    "F:\\\\vandomej\\Projects\\dootcamp-AI-Simulation\\Package\\Windows\\AI_Fight_Sim.exe";

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
    pub crossbreed_segments: usize,
    pub weight_initialization_range: Range<f32>,
    pub minor_mutation_rate: f32,
    pub major_mutation_rate: f32,
    pub mutation_weight_range: Range<f32>,
}

#[async_trait]
impl GeneticNode for FighterNN {
    type Context = fighter_context::FighterContext;

    // Check for the highest number of the folder name and increment it by 1
    async fn initialize(context: GeneticNodeContext<Self::Context>) -> Result<Box<Self>, Error> {
        let base_path = PathBuf::from(BASE_DIR);

        let folder = base_path.join(format!("fighter_nn_{:06}", context.id));
        // Ensures directory is created if it doesn't exist and does nothing if it exists
        fs::create_dir_all(&folder)
            .with_context(|| format!("Failed to create or access the folder: {:?}", folder))?;

        //Create a new directory for the first generation, using create_dir_all to avoid errors if it already exists
        let gen_folder = folder.join("0");
        fs::create_dir_all(&gen_folder).with_context(|| {
            format!(
                "Failed to create or access the generation folder: {:?}",
                gen_folder
            )
        })?;

        let mut nn_shapes = HashMap::new();
        let weight_initialization_range = thread_rng()
            .gen_range(NEURAL_NETWORK_INITIAL_WEIGHT_MIN..0.0)
            ..thread_rng().gen_range(0.0..=NEURAL_NETWORK_INITIAL_WEIGHT_MAX);

        // Create the first generation in this folder
        for i in 0..POPULATION {
            // Filenames are stored in the format of "xxxxxx_fighter_nn_0.net", "xxxxxx_fighter_nn_1.net", etc. Where xxxxxx is the folder name
            let nn = gen_folder
                .join(format!("{:06}_fighter_nn_{}", context.id, i))
                .with_extension("net");

            // Randomly generate a neural network shape based on constants
            let hidden_layers = thread_rng()
                .gen_range(NEURAL_NETWORK_HIDDEN_LAYERS_MIN..NEURAL_NETWORK_HIDDEN_LAYERS_MAX);
            let mut nn_shape = vec![NEURAL_NETWORK_INPUTS as u32];
            for _ in 0..hidden_layers {
                nn_shape.push(thread_rng().gen_range(
                    NEURAL_NETWORK_HIDDEN_LAYER_SIZE_MIN..NEURAL_NETWORK_HIDDEN_LAYER_SIZE_MAX,
                ) as u32);
            }
            nn_shape.push(NEURAL_NETWORK_OUTPUTS as u32);
            nn_shapes.insert(i as u64, nn_shape.clone());

            let mut fann = Fann::new(nn_shape.as_slice()).with_context(|| "Failed to create nn")?;
            fann.randomize_weights(
                weight_initialization_range.start,
                weight_initialization_range.end,
            );
            fann.set_activation_func_hidden(ActivationFunc::SigmoidSymmetric);
            fann.set_activation_func_output(ActivationFunc::SigmoidSymmetric);
            // This will overwrite any existing file with the same name
            fann.save(&nn)
                .with_context(|| format!("Failed to save nn at {:?}", nn))?;
        }

        let mut crossbreed_segments = thread_rng().gen_range(
            NEURAL_NETWORK_CROSSBREED_SEGMENTS_MIN..NEURAL_NETWORK_CROSSBREED_SEGMENTS_MAX,
        );
        if crossbreed_segments % 2 == 0 {
            crossbreed_segments += 1;
        }

        let mutation_weight_amplitude = thread_rng().gen_range(0.0..1.0);

        Ok(Box::new(FighterNN {
            id: context.id,
            folder,
            population_size: POPULATION,
            generation: 0,
            scores: vec![HashMap::new()],
            nn_shapes,
            // we need crossbreed segments to be even
            crossbreed_segments,
            weight_initialization_range,
            minor_mutation_rate: thread_rng().gen_range(0.0..1.0),
            major_mutation_rate: thread_rng().gen_range(0.0..1.0),
            mutation_weight_range: -mutation_weight_amplitude..mutation_weight_amplitude,
        }))
    }

    async fn simulate(&mut self, context: GeneticNodeContext<Self::Context>) -> Result<(), Error> {
        debug!("Context: {:?}", context);
        let mut tasks = Vec::new();

        // For each nn in the current generation:
        for i in 0..self.population_size {
            let self_clone = self.clone();
            let semaphore_clone = context.gemla_context.shared_semaphore.clone();
            let display_simulation_semaphore = context.gemla_context.visible_simulations.clone();

            let task = async move {
                let nn = self_clone
                    .folder
                    .join(format!("{}", self_clone.generation))
                    .join(self_clone.get_individual_id(i as u64))
                    .with_extension("net");
                let mut simulations = Vec::new();

                // Using the same original nn, repeat the simulation with 5 random nn's from the current generation concurrently
                for _ in 0..SIMULATION_ROUNDS {
                    let random_nn_index = thread_rng().gen_range(0..self_clone.population_size);
                    let folder = self_clone.folder.clone();
                    let generation = self_clone.generation;
                    let semaphore_clone = semaphore_clone.clone();
                    let display_simulation_semaphore = display_simulation_semaphore.clone();

                    let random_nn = folder
                        .join(format!("{}", generation))
                        .join(self_clone.get_individual_id(random_nn_index as u64))
                        .with_extension("net");
                    let nn_clone = nn.clone(); // Clone the path to use in the async block

                    let future = async move {
                        let permit = semaphore_clone
                            .acquire_owned()
                            .await
                            .with_context(|| "Failed to acquire semaphore permit")?;

                        let display_simulation =
                            match display_simulation_semaphore.try_acquire_owned() {
                                Ok(s) => Some(s),
                                Err(_) => None,
                            };

                        let (score, _) = if let Some(display_simulation) = display_simulation {
                            let result = run_1v1_simulation(&nn_clone, &random_nn, true).await?;
                            drop(display_simulation);
                            result
                        } else {
                            run_1v1_simulation(&nn_clone, &random_nn, false).await?
                        };

                        drop(permit);

                        Ok(score)
                    };

                    simulations.push(future);
                }

                // Wait for all simulation rounds to complete
                let results: Result<Vec<f32>, Error> =
                    join_all(simulations).await.into_iter().collect();

                let score = match results {
                    Ok(scores) => scores.into_iter().sum::<f32>() / SIMULATION_ROUNDS as f32,
                    Err(e) => return Err(e), // Return the error if results collection failed
                };
                debug!("NN {:06}_fighter_nn_{} scored {}", self_clone.id, i, score);
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
                }
                Err(e) => {
                    // Handle task panic or execution error
                    return Err(Error::Other(anyhow::anyhow!(format!(
                        "Task failed: {:?}",
                        e
                    ))));
                }
            }
        }

        Ok(())
    }

    async fn mutate(&mut self, _context: GeneticNodeContext<Self::Context>) -> Result<(), Error> {
        let survivor_count = (self.population_size as f32 * SURVIVAL_RATE) as usize;
        let mut nn_sizes = Vec::new();

        // Create the new generation folder
        let new_gen_folder = self.folder.join(format!("{}", self.generation + 1));
        fs::create_dir_all(&new_gen_folder).with_context(|| {
            format!(
                "Failed to create or access new generation folder: {:?}",
                new_gen_folder
            )
        })?;

        // Remove the 5 nn's with the lowest scores
        let mut sorted_scores: Vec<_> = self.scores[self.generation as usize].iter().collect();
        sorted_scores.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        let scores_to_keep: Vec<&(&u64, &f32)> =
            sorted_scores.iter().take(survivor_count).collect();
        let to_keep = scores_to_keep.iter().map(|(k, _)| *k).collect::<Vec<_>>();

        // Save the remaining 5 nn's to the new generation folder
        for (i, nn_id) in to_keep.iter().enumerate().take(survivor_count) {
            let nn = self
                .folder
                .join(format!("{}", self.generation))
                .join(format!("{:06}_fighter_nn_{}.net", self.id, nn_id));
            let new_nn = new_gen_folder.join(format!("{:06}_fighter_nn_{}.net", self.id, i));
            debug!("Copying nn from {:?} to {:?}", nn_id, i);
            fs::copy(&nn, &new_nn)?;
            nn_sizes.push(self.nn_shapes.get(nn_id).unwrap().clone());
        }

        let weights: HashMap<u64, f32> = scores_to_keep.iter().map(|(k, v)| (**k, **v)).collect();

        debug!("scores: {:?}", scores_to_keep);

        let mut tasks = Vec::new();

        // Take the remaining nn's and create new nn's by the following:
        for i in 0..survivor_count {
            let self_clone = self.clone();

            // randomly select individual id's sorted scores proportional to their score
            let nn_id = weighted_random_selection(&weights);
            let nn = self_clone
                .folder
                .join(format!("{}", self_clone.generation))
                .join(self_clone.get_individual_id(nn_id))
                .with_extension("net");

            // Load another nn from the current generation and cross breed it with the current nn
            let cross_id = loop {
                let cross_id = weighted_random_selection(&weights);
                if cross_id != nn_id {
                    break cross_id;
                }
            };

            let cross_nn = self_clone
                .folder
                .join(format!("{}", self_clone.generation))
                .join(self_clone.get_individual_id(cross_id))
                .with_extension("net");

            let new_gen_folder = new_gen_folder.clone();

            let future = tokio::task::spawn_blocking(move || -> Result<Vec<u32>, Error> {
                let fann = Fann::from_file(&nn).with_context(|| "Failed to load nn")?;
                let cross_fann =
                    Fann::from_file(&cross_nn).with_context(|| "Failed to load cross nn")?;

                let mut new_fann = crossbreed(
                    &self_clone,
                    &fann,
                    &cross_fann,
                    self_clone.crossbreed_segments,
                )?;

                // For each weight in the 5 new nn's there is a 20% chance of a minor mutation (a random number between -0.1 and 0.1 is added to the weight)
                // And a 5% chance of a major mutation a new neuron is randomly added to a hidden layer
                let mut connections = new_fann.get_connections(); // Vector of connections
                for c in &mut connections {
                    if thread_rng().gen_range(0.0..1.0) < self_clone.minor_mutation_rate {
                        trace!("Minor mutation on connection {:?}", c);
                        c.weight +=
                            thread_rng().gen_range(self_clone.weight_initialization_range.clone());
                        trace!("New weight: {}", c.weight);
                    }
                }

                new_fann.set_connections(&connections);

                if thread_rng().gen_range(0.0..1.0) < self_clone.major_mutation_rate {
                    new_fann =
                        major_mutation(&new_fann, self_clone.weight_initialization_range.clone())?;
                }

                let new_nn = new_gen_folder
                    .join(self_clone.get_individual_id((i + survivor_count) as u64))
                    .with_extension("net");
                new_fann
                    .save(&new_nn)
                    .with_context(|| "Failed to save nn")?;

                Ok::<Vec<u32>, Error>(new_fann.get_layer_sizes())
            });

            tasks.push(future);
        }

        let results = join_all(tasks).await;

        for result in results.into_iter() {
            let new_size = result.with_context(|| "Failed to create new nn")??;
            nn_sizes.push(new_size);
        }

        self.generation += 1;
        self.scores.push(HashMap::new());

        Ok(())
    }

    async fn merge(
        left: &FighterNN,
        right: &FighterNN,
        id: &Uuid,
        gemla_context: Self::Context,
    ) -> Result<Box<FighterNN>, Error> {
        let base_path = PathBuf::from(BASE_DIR);
        let folder = base_path.join(format!("fighter_nn_{:06}", id));

        // Ensure the folder exists, including the generation subfolder.
        fs::create_dir_all(folder.join("0"))
            .with_context(|| format!("Failed to create directory {:?}", folder.join("0")))?;

        let get_highest_scores = |fighter: &FighterNN| -> Vec<(u64, f32)> {
            let mut sorted_scores: Vec<_> =
                fighter.scores[fighter.generation as usize].iter().collect();
            sorted_scores.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
            sorted_scores
                .iter()
                .take(fighter.population_size / 2)
                .map(|(k, v)| (**k, **v))
                .collect()
        };

        let left_scores = get_highest_scores(left);
        let right_scores = get_highest_scores(right);

        debug!("Left scores: {:?}", left_scores);
        debug!("Right scores: {:?}", right_scores);

        let mut simulations = Vec::new();

        let left_weights: HashMap<u64, f32> = left_scores.iter().map(|(k, v)| (*k, *v)).collect();
        let right_weights: HashMap<u64, f32> = right_scores.iter().map(|(k, v)| (*k, *v)).collect();

        let num_simulations = max(left.population_size, right.population_size) * SIMULATION_ROUNDS;

        for _ in 0..num_simulations {
            let left_nn_id = weighted_random_selection(&left_weights);
            let right_nn_id = weighted_random_selection(&right_weights);

            let left_nn_path = left
                .folder
                .join(left.generation.to_string())
                .join(left.get_individual_id(left_nn_id))
                .with_extension("net");
            let right_nn_path = right
                .folder
                .join(right.generation.to_string())
                .join(right.get_individual_id(right_nn_id))
                .with_extension("net");
            let semaphore_clone = gemla_context.shared_semaphore.clone();
            let display_simulation_semaphore = gemla_context.visible_simulations.clone();

            let future = async move {
                let permit = semaphore_clone
                    .acquire_owned()
                    .await
                    .with_context(|| "Failed to acquire semaphore permit")?;

                let display_simulation = match display_simulation_semaphore.try_acquire_owned() {
                    Ok(s) => Some(s),
                    Err(_) => None,
                };

                let (left_score, right_score) = if let Some(display_simulation) = display_simulation
                {
                    let result = run_1v1_simulation(&left_nn_path, &right_nn_path, true).await?;
                    drop(display_simulation);
                    result
                } else {
                    run_1v1_simulation(&left_nn_path, &right_nn_path, false).await?
                };

                drop(permit);

                Ok::<(f32, f32), Error>((left_score, right_score))
            };

            simulations.push(future);
        }

        let results: Result<Vec<(f32, f32)>, Error> =
            join_all(simulations).await.into_iter().collect();
        let scores = results?;

        let total_left_score = scores.iter().map(|(l, _)| l).sum::<f32>() / num_simulations as f32;
        let total_right_score = scores.iter().map(|(_, r)| r).sum::<f32>() / num_simulations as f32;

        debug!("Total left score: {}", total_left_score);
        debug!("Total right score: {}", total_right_score);

        let score_difference = total_right_score - total_left_score;
        // Use the sigmoid function to determine lerp amount
        let lerp_amount = 1.0 / (1.0 + (-score_difference).exp());

        debug!("Lerp amount: {}", lerp_amount);

        let mut nn_shapes = HashMap::new();

        // Function to copy NNs from a source FighterNN to the new folder.
        let mut copy_nns = |source: &FighterNN,
                            folder: &PathBuf,
                            id: &Uuid,
                            start_idx: usize|
         -> Result<(), Error> {
            let mut sorted_scores: Vec<_> =
                source.scores[source.generation as usize].iter().collect();
            sorted_scores.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());
            let remaining = sorted_scores[(source.population_size / 2)..]
                .iter()
                .map(|(k, _)| *k)
                .collect::<Vec<_>>();

            for (i, nn_id) in remaining.into_iter().enumerate() {
                let nn_path = source
                    .folder
                    .join(source.generation.to_string())
                    .join(format!("{:06}_fighter_nn_{}.net", source.id, nn_id));
                let new_nn_path =
                    folder
                        .join("0")
                        .join(format!("{:06}_fighter_nn_{}.net", id, start_idx + i));
                fs::copy(&nn_path, &new_nn_path).with_context(|| {
                    format!("Failed to copy nn from {:?} to {:?}", nn_path, new_nn_path)
                })?;

                nn_shapes.insert(
                    (start_idx + i) as u64,
                    source.nn_shapes.get(nn_id).unwrap().clone(),
                );
            }

            Ok(())
        };

        // Copy the top half of NNs from each parent to the new folder.
        copy_nns(left, &folder, id, 0)?;
        copy_nns(right, &folder, id, left.population_size / 2)?;

        debug!("nn_shapes: {:?}", nn_shapes);

        // Lerp the mutation rates and weight ranges
        let crossbreed_segments = (left.crossbreed_segments as f32)
            .lerp(right.crossbreed_segments as f32, lerp_amount)
            as usize;

        let weight_initialization_range_start = left
            .weight_initialization_range
            .start
            .lerp(right.weight_initialization_range.start, lerp_amount);
        let weight_initialization_range_end = left
            .weight_initialization_range
            .end
            .lerp(right.weight_initialization_range.end, lerp_amount);
        // Have to ensure the range is valid
        let weight_initialization_range =
            if weight_initialization_range_start < weight_initialization_range_end {
                weight_initialization_range_start..weight_initialization_range_end
            } else {
                weight_initialization_range_end..weight_initialization_range_start
            };

        debug!(
            "weight_initialization_range: {:?}",
            weight_initialization_range
        );

        let minor_mutation_rate = left
            .minor_mutation_rate
            .lerp(right.minor_mutation_rate, lerp_amount);
        let major_mutation_rate = left
            .major_mutation_rate
            .lerp(right.major_mutation_rate, lerp_amount);

        debug!("minor_mutation_rate: {}", minor_mutation_rate);
        debug!("major_mutation_rate: {}", major_mutation_rate);

        let mutation_weight_range_start = left
            .mutation_weight_range
            .start
            .lerp(right.mutation_weight_range.start, lerp_amount);
        let mutation_weight_range_end = left
            .mutation_weight_range
            .end
            .lerp(right.mutation_weight_range.end, lerp_amount);
        // Have to ensure the range is valid
        let mutation_weight_range = if mutation_weight_range_start < mutation_weight_range_end {
            mutation_weight_range_start..mutation_weight_range_end
        } else {
            mutation_weight_range_end..mutation_weight_range_start
        };

        debug!("mutation_weight_range: {:?}", mutation_weight_range);

        Ok(Box::new(FighterNN {
            id: *id,
            folder,
            generation: 0,
            population_size: nn_shapes.len(),
            scores: vec![HashMap::new()],
            crossbreed_segments,
            nn_shapes,
            weight_initialization_range,
            minor_mutation_rate,
            major_mutation_rate,
            mutation_weight_range,
        }))
    }
}

impl FighterNN {
    pub fn get_individual_id(&self, nn_id: u64) -> String {
        format!("{:06}_fighter_nn_{}", self.id, nn_id)
    }
}

fn weighted_random_selection<T: Clone + std::hash::Hash + Eq>(weights: &HashMap<T, f32>) -> T {
    let mut rng = thread_rng();

    // Identify the minimum weight
    let min_weight = weights.values().fold(f32::INFINITY, |a, &b| a.min(b));

    // Adjust all weights to be non-negative
    let offset = if min_weight < 0.0 {
        (-min_weight) + 0.5
    } else {
        0.0
    };
    let total_weight: f32 = weights.values().map(|w| w + offset).sum();

    let mut cumulative_weight = 0.0;
    let random_weight = rng.gen::<f32>() * total_weight;

    for (item, weight) in weights.iter() {
        cumulative_weight += *weight + offset;
        if cumulative_weight >= random_weight {
            return item.clone();
        }
    }

    panic!("Weighted random selection failed.");
}

async fn run_1v1_simulation(
    nn_path_1: &Path,
    nn_path_2: &Path,
    display_simulation: bool,
) -> Result<(f32, f32), Error> {
    // Construct the score file path
    let base_folder = nn_path_1.parent().unwrap();
    let nn_1_id = nn_path_1.file_stem().unwrap().to_str().unwrap();
    let nn_2_id = nn_path_2.file_stem().unwrap().to_str().unwrap();
    let score_file = base_folder.join(format!("{}_vs_{}.txt", nn_1_id, nn_2_id));

    // Check if score file already exists before running the simulation
    if score_file.exists() {
        let round_score = read_score_from_file(&score_file, nn_1_id)
            .await
            .with_context(|| format!("Failed to read score from file: {:?}", score_file))?;

        let opposing_score = read_score_from_file(&score_file, nn_2_id)
            .await
            .with_context(|| format!("Failed to read score from file: {:?}", score_file))?;

        debug!(
            "{} scored {}, while {} scored {}",
            nn_1_id, round_score, nn_2_id, opposing_score
        );

        return Ok((round_score, opposing_score));
    }

    // Check if the opposite round score has been determined
    let opposite_score_file = base_folder.join(format!("{}_vs_{}.txt", nn_2_id, nn_1_id));
    if opposite_score_file.exists() {
        let round_score = read_score_from_file(&opposite_score_file, nn_1_id)
            .await
            .with_context(|| {
                format!("Failed to read score from file: {:?}", opposite_score_file)
            })?;

        let opposing_score = read_score_from_file(&opposite_score_file, nn_2_id)
            .await
            .with_context(|| {
                format!("Failed to read score from file: {:?}", opposite_score_file)
            })?;

        debug!(
            "{} scored {}, while {} scored {}",
            nn_1_id, round_score, nn_2_id, opposing_score
        );

        return Ok((round_score, opposing_score));
    }

    // Run simulation until score file is generated
    let config1_arg = format!("-NN1Config=\"{}\"", nn_path_1.to_str().unwrap());
    let config2_arg = format!("-NN2Config=\"{}\"", nn_path_2.to_str().unwrap());
    let disable_unreal_rendering_arg = "-nullrhi".to_string();

    trace!(
        "Executing the following command {} {} {} {}",
        GAME_EXECUTABLE_PATH,
        config1_arg,
        config2_arg,
        disable_unreal_rendering_arg
    );

    trace!("Running simulation for {} vs {}", nn_1_id, nn_2_id);

    let _output = if display_simulation {
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

    trace!(
        "Simulation completed for {} vs {}: {}",
        nn_1_id,
        nn_2_id,
        score_file.exists()
    );

    // Read the score from the file
    if score_file.exists() {
        let round_score = read_score_from_file(&score_file, nn_1_id)
            .await
            .with_context(|| format!("Failed to read score from file: {:?}", score_file))?;

        let opposing_score = read_score_from_file(&score_file, nn_2_id)
            .await
            .with_context(|| format!("Failed to read score from file: {:?}", score_file))?;

        debug!(
            "{} scored {}, while {} scored {}",
            nn_1_id, round_score, nn_2_id, opposing_score
        );

        Ok((round_score, opposing_score))
    } else {
        warn!("Score file not found: {:?}", score_file);
        Ok((0.0, 0.0))
    }
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
                            return parts[1]
                                .trim()
                                .parse::<f32>()
                                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e));
                        }
                    }
                }

                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    "NN ID not found in scores file",
                ));
            }
            Err(e)
                if e.kind() == io::ErrorKind::WouldBlock
                    || e.kind() == io::ErrorKind::PermissionDenied
                    || e.kind() == io::ErrorKind::Other =>
            {
                if attempts >= 5 {
                    // Attempt 5 times before giving up.
                    return Err(e);
                }

                attempts += 1;
                // wait 1 second to ensure the file is written
                tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
            }
            Err(e) => return Err(e),
        }
    }
}

#[cfg(test)]
pub mod test {
    use super::*;

    #[test]
    fn test_weighted_random_selection() {
        let weights = vec![
            (43, -4.0403514),
            (26, -2.9386168),
            (44, -2.8106647),
            (46, -1.3942022),
            (23, 0.99386656),
            (41, -2.2198126),
            (48, 1.2195103),
            (42, -3.4927247),
            (7, -1.092067),
            (0, -0.3878999),
            (49, -4.156101),
            (34, -0.33209237),
            (30, -2.7059758),
            (2, -2.251783),
            (20, -0.5811202),
            (10, -3.047954),
            (6, -4.3464293),
            (39, -3.7280478),
            (1, -3.4291298),
            (11, -2.0568254),
            (24, -1.5701149),
            (8, -1.5029285),
            (3, -2.4728038),
            (4, 3.7312133),
            (25, -1.227466),
        ]
        .into_iter()
        .collect();

        let mut ids = vec![
            43, 26, 44, 46, 23, 41, 48, 42, 7, 0, 49, 34, 30, 2, 20, 10, 6, 39, 1, 11, 24, 8, 3, 4,
            25,
        ];

        for _ in 0..10000 {
            let id = weighted_random_selection(&weights);

            ids = ids.into_iter().filter(|&x| x != id).collect();

            assert!(weights.contains_key(&id));
        }

        assert_eq!(ids.len(), 0);
    }
}
