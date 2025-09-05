extern crate fann;

pub mod fighter_context;
pub mod neural_network_utility;

use anyhow::{anyhow, Context};
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
const POPULATION: usize = 200;

const NEURAL_NETWORK_INPUTS: usize = 22;
const NEURAL_NETWORK_OUTPUTS: usize = 8;

const NEURAL_NETWORK_HIDDEN_LAYERS_MIN: usize = 1;
const NEURAL_NETWORK_HIDDEN_LAYERS_MAX: usize = 2;

const NEURAL_NETWORK_HIDDEN_LAYER_SIZE_MIN: usize = 3;
const NEURAL_NETWORK_HIDDEN_LAYER_SIZE_MAX: usize = 50;

const NEURAL_NETWORK_INITIAL_WEIGHT_MAX: f32 = 0.5;

const NEURAL_NETWORK_MINOR_MUTATION_RATE_MAX: f32 = 0.3;
const NEURAL_NETWORK_MUTATION_WEIGHT_MAX: f32 = 1.0;

const NEURAL_NETWORK_MAJOR_MUTATION_RATE_MAX: f32 = 1.0;

const NEURAL_NETWORK_CROSSBREED_SEGMENTS_MIN: usize = 2;
const NEURAL_NETWORK_CROSSBREED_SEGMENTS_MAX: usize = 6;

const OFFSHOOT_GENERATIONAL_LENIENCE: u64 = 10;
const MAINLINE_GENERATIONAL_LENIENCE: u64 = 20;

const SIMULATION_ROUNDS: usize = 5;
const SURVIVAL_RATE_MIN: f32 = 0.1;
const SURVIVAL_RATE_MAX: f32 = 0.9;
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
    pub nn_shapes: Vec<HashMap<u64, Vec<u32>>>,
    pub crossbreed_segments: usize,
    pub weight_initialization_range: Range<f32>,
    pub minor_mutation_rate: f32,
    pub major_mutation_rate: f32,
    pub mutation_weight_range: Range<f32>,
    // Shows how individuals are mapped from one generation to the next
    pub id_mapping: Vec<HashMap<u64, u64>>,
    pub lerp_amount: f32,
    pub generational_lenience: u64,
    pub survival_rate: f32,
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
        let weight_initialization_amplitude = thread_rng().gen_range(0.0..NEURAL_NETWORK_INITIAL_WEIGHT_MAX);
        let weight_initialization_range = -weight_initialization_amplitude..weight_initialization_amplitude;

        // Create the first generation in this folder
        for i in 0..POPULATION {
            // Filenames are stored in the format of "xxxxxx_fighter_nn_0.net", "xxxxxx_fighter_nn_1.net", etc. Where xxxxxx is the folder name
            let nn = gen_folder
                .join(format!("{:06}_fighter_nn_{}", context.id, i))
                .with_extension("net");

            // Randomly generate a neural network shape based on constants
            let hidden_layers = thread_rng()
                .gen_range(NEURAL_NETWORK_HIDDEN_LAYERS_MIN..=NEURAL_NETWORK_HIDDEN_LAYERS_MAX);
            let mut nn_shape = vec![NEURAL_NETWORK_INPUTS as u32];
            for _ in 0..hidden_layers {
                nn_shape.push(thread_rng().gen_range(
                    NEURAL_NETWORK_HIDDEN_LAYER_SIZE_MIN..=NEURAL_NETWORK_HIDDEN_LAYER_SIZE_MAX,
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
            NEURAL_NETWORK_CROSSBREED_SEGMENTS_MIN..=NEURAL_NETWORK_CROSSBREED_SEGMENTS_MAX,
        );
        if crossbreed_segments % 2 == 0 {
            crossbreed_segments += 1;
        }

        let mutation_weight_amplitude = thread_rng().gen_range(0.0..NEURAL_NETWORK_MUTATION_WEIGHT_MAX);

        Ok(Box::new(FighterNN {
            id: context.id,
            folder,
            population_size: POPULATION,
            generation: 0,
            scores: vec![],
            nn_shapes: vec![nn_shapes],
            // we need crossbreed segments to be even
            crossbreed_segments,
            weight_initialization_range,
            minor_mutation_rate: thread_rng().gen_range(0.0..NEURAL_NETWORK_MINOR_MUTATION_RATE_MAX),
            major_mutation_rate: thread_rng().gen_range(0.0..NEURAL_NETWORK_MAJOR_MUTATION_RATE_MAX),
            mutation_weight_range: -mutation_weight_amplitude..mutation_weight_amplitude,
            id_mapping: vec![],
            lerp_amount: 0.0,
            generational_lenience: OFFSHOOT_GENERATIONAL_LENIENCE,
            survival_rate: thread_rng().gen_range(SURVIVAL_RATE_MIN..SURVIVAL_RATE_MAX),
        }))
    }

    async fn simulate(
        &mut self,
        context: GeneticNodeContext<Self::Context>,
    ) -> Result<bool, Error> {
        debug!("Context: {:?}", context);
        let mut matches = Vec::new();
        let mut allotted_simulations = Vec::new();
        for i in 0..self.population_size {
            allotted_simulations.push((i, SIMULATION_ROUNDS));
        }

        while !allotted_simulations.is_empty() {
            let primary_id = {
                let id = thread_rng().gen_range(0..allotted_simulations.len());
                let (i, _) = allotted_simulations[id];
                // Decrement the number of simulations left for this nn
                allotted_simulations[id].1 -= 1;
                // Remove the nn from the list if it has no more simulations left
                if allotted_simulations[id].1 == 0 {
                    allotted_simulations.remove(id);
                }
                i
            };


            let secondary_id = loop {
                if allotted_simulations.is_empty() || allotted_simulations.len() == 1 {
                    // Select a random id
                    let random_id = loop {
                        let id = thread_rng().gen_range(0..self.population_size);
                        if id != primary_id {
                            allotted_simulations.clear();
                            break id;
                        }
                    };

                    break random_id;
                }

                let id = thread_rng().gen_range(0..allotted_simulations.len());
                let (i, _) = allotted_simulations[id];

                if i != primary_id {
                    // Decrement the number of simulations left for this nn
                    allotted_simulations[id].1 -= 1;
                    // Remove the nn from the list if it has no more simulations left
                    if allotted_simulations[id].1 == 0 {
                        allotted_simulations.remove(id);
                    }
                    break i;
                }
            };

            matches.push((primary_id, secondary_id));
        }

        debug!("Matches determined");
        trace!("Matches: {:?}", matches);

        // Create a channel to send the scores back to the main thread
        let mut tasks = Vec::new();

        for (primary_id, secondary_id) in matches.iter() {
            let task = {
                let self_clone = self.clone();
                let semaphore_clone = context.gemla_context.shared_semaphore.clone();
                let display_simulation_semaphore = context.gemla_context.visible_simulations.clone();
            
                let folder = self_clone.folder.clone();
                let generation = self_clone.generation;
        
                let primary_nn = self_clone
                    .folder
                    .join(format!("{}", self_clone.generation))
                    .join(self_clone.get_individual_id(*primary_id as u64))
                    .with_extension("net");
                let secondary_nn = folder
                    .join(format!("{}", generation))
                    .join(self_clone.get_individual_id(*secondary_id as u64))
                    .with_extension("net");
        
                // Introducing a new scope for acquiring permits and running simulations
                let simulation_result = async move {
                    let permit = semaphore_clone.acquire_owned().await
                        .with_context(|| "Failed to acquire semaphore permit")?;
        
                    let display_simulation = match display_simulation_semaphore.try_acquire_owned() {
                        Ok(s) => Some(s),
                        Err(_) => None,
                    };
        
                    let (primary_score, secondary_score) = if let Some(display_simulation) = display_simulation {
                        let result = run_1v1_simulation(&primary_nn, &secondary_nn, true).await?;
                        drop(display_simulation); // Explicitly dropping resources no longer needed
                        result
                    } else {
                        run_1v1_simulation(&primary_nn, &secondary_nn, false).await?
                    };
        
                    drop(permit); // Explicitly dropping resources no longer needed

                    debug!(
                        "{} vs {} -> {} vs {}",
                        primary_id, secondary_id, primary_score, secondary_score
                    );
        
                    Ok((*primary_id, primary_score, *secondary_id, secondary_score))
                }; // Await the scoped async block immediately
        
                // The result of the simulation, whether Ok or Err, is returned here.
                // This ensures tx is dropped when the block exits, regardless of success or failure.
                simulation_result
            };

            tasks.push(task);
        }

        debug!("Tasks created");

        let results: Vec<Result<(usize, f32, usize, f32), Error>> = join_all(tasks).await;

        debug!("Tasks completed");

        // resolve results for any errors
        let mut scores = HashMap::new();
        for result in results.into_iter() {
            let (primary_id, primary_score, secondary_id, secondary_score) = result.with_context(|| "Failed to run simulation")?;

            // If score exists, add the new score to the existing score
            if let Some((existing_score, count)) = scores.get_mut(&(primary_id as u64)) {
                *existing_score += primary_score;
                *count += 1;
            } else {
                scores.insert(primary_id as u64, (primary_score, 1));
            }

            // If score exists, add the new score to the existing score
            if let Some((existing_score, count)) = scores.get_mut(&(secondary_id as u64)) {
                *existing_score += secondary_score;
                *count += 1;
            } else {
                scores.insert(secondary_id as u64, (secondary_score, 1));
            }
        }

        // Average scores for each individual
        let mut final_scores = HashMap::new();
        for (i, (score, count)) in scores.iter() {
            final_scores.insert(*i, *score / *count as f32);
        }

        self.scores.push(final_scores);

        Ok(should_continue(&self.scores, self.generational_lenience)?)
    }

    async fn mutate(&mut self, _context: GeneticNodeContext<Self::Context>) -> Result<(), Error> {
        let survivor_count = (self.population_size as f32 * self.survival_rate) as usize;
        let mut nn_sizes = Vec::new();
        let mut id_mapping = HashMap::new();

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
            id_mapping.insert(**nn_id, i as u64);
            fs::copy(&nn, &new_nn)?;
            nn_sizes.push(
                self.nn_shapes[self.generation as usize]
                    .get(nn_id)
                    .unwrap()
                    .clone(),
            );
        }

        let weights: HashMap<u64, f32> = scores_to_keep.iter().map(|(k, v)| (**k, **v)).collect();

        debug!("scores: {:?}", scores_to_keep);

        let mut tasks = Vec::new();

        // Take the remaining nn's and create new nn's by the following:
        for i in 0..(self.population_size - survivor_count) {
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
                new_fann.save(new_nn).with_context(|| "Failed to save nn")?;

                Ok::<Vec<u32>, Error>(new_fann.get_layer_sizes())
            });

            tasks.push(future);
        }

        let results = join_all(tasks).await;

        for result in results.into_iter() {
            let new_size = result.with_context(|| "Failed to create new nn")??;
            nn_sizes.push(new_size);
        }

        // Use the index of nn_sizes to generate the id for the nn_sizes HashMap
        let nn_sizes_map = nn_sizes
            .into_iter()
            .enumerate()
            .map(|(i, v)| (i as u64, v))
            .collect::<HashMap<_, _>>();

        self.generation += 1;
        self.nn_shapes.push(nn_sizes_map);
        self.id_mapping.push(id_mapping);

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

                debug!("{} vs {} -> {} vs {}", left_nn_id, right_nn_id, left_score, right_score);

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

                let nn_shape = source.nn_shapes[source.generation as usize]
                    .get(nn_id)
                    .unwrap();

                nn_shapes.insert((start_idx + i) as u64, nn_shape.clone());
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

        let survival_rate = left.survival_rate.lerp(right.survival_rate, lerp_amount);

        debug!("survival_rate: {}", survival_rate);

        Ok(Box::new(FighterNN {
            id: *id,
            folder,
            generation: 0,
            population_size: nn_shapes.len(),
            scores: vec![],
            crossbreed_segments,
            nn_shapes: vec![nn_shapes],
            weight_initialization_range,
            minor_mutation_rate,
            major_mutation_rate,
            mutation_weight_range,
            id_mapping: vec![],
            lerp_amount,
            // generational_lenience: left.generational_lenience + MAINLINE_GENERATIONAL_LENIENCE,
            generational_lenience: MAINLINE_GENERATIONAL_LENIENCE,
            survival_rate,
        }))
    }
}

impl FighterNN {
    pub fn get_individual_id(&self, nn_id: u64) -> String {
        format!("{:06}_fighter_nn_{}", self.id, nn_id)
    }
}

fn should_continue(scores: &[HashMap<u64, f32>], lenience: u64) -> Result<bool, Error> {
    if scores.len() < lenience as usize {
        return Ok(true);
    }

    let mut highest_q3_value = f32::MIN;
    let mut generation_with_highest_q3 = 0;

    let mut highest_median = f32::MIN;
    let mut generation_with_highest_median = 0;

    for (generation_index, generation) in scores.iter().enumerate() {
        let mut scores: Vec<f32> = generation.values().copied().collect();
        scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let q3_index = (scores.len() as f32 * 0.75).ceil() as usize - 1;
        let q3_value = scores
            .get(q3_index)
            .ok_or(anyhow!("Failed to get Q3 value"))?;

        if *q3_value > highest_q3_value {
            highest_q3_value = *q3_value;
            generation_with_highest_q3 = generation_index;
        }

        let median_index = (scores.len() as f32 * 0.5).ceil() as usize - 1;
        let median_value = scores
            .get(median_index)
            .ok_or(anyhow!("Failed to get median value"))?;

        if *median_value > highest_median {
            highest_median = *median_value;
            generation_with_highest_median = generation_index;
        }
    }

    let highest_generation_index = scores.len() - 1;
    let result = highest_generation_index - generation_with_highest_q3 < lenience as usize
        && highest_generation_index - generation_with_highest_median < lenience as usize;

    debug!(
        "Highest Q3 value: {} at generation {}, Highest Median value: {} at generation {}, Continuing? {}",
        highest_q3_value, generation_with_highest_q3 + 1, highest_median, generation_with_highest_median + 1, result
    );

    Ok(result)
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

        trace!(
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

        trace!(
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

        trace!(
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
            Err(_) =>
            {
                if attempts >= 2 {
                    // Attempt 5 times before giving up.
                    return Ok(-100.0);
                }

                attempts += 1;
                // wait 1 second to ensure the file is written
                tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;
            }
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

    #[test]
    fn test_should_continue() {
        let scores = vec![
            // Generation 0
            [
                (37, -7.1222725),
                (12, -3.6037624),
                (27, -5.202844),
                (21, -6.3283415),
                (4, -6.0053186),
                (8, -4.040202),
                (13, -4.0050435),
                (17, -5.8206105),
                (40, -7.5448103),
                (42, -8.027704),
                (15, -5.1600137),
                (10, -7.9063845),
                (1, -6.9830275),
                (7, -3.3323112),
                (16, -6.1065326),
                (23, -6.417853),
                (25, -6.410652),
                (14, -6.5887403),
                (3, -6.3966584),
                (19, 0.1242948),
                (28, -4.806827),
                (18, -6.3310747),
                (30, -5.8972425),
                (31, -6.398958),
                (22, -7.042196),
                (29, -5.7098813),
                (9, -8.931531),
                (33, -5.9806275),
                (6, -6.5489874),
                (26, -5.892653),
                (34, -6.4281516),
                (35, -5.5369387),
                (38, -5.495344),
                (43, 0.9552175),
                (44, -6.2549844),
                (45, -8.42142),
                (24, -7.121878),
                (47, -5.373896),
                (48, -6.445716),
                (39, -6.053849),
                (11, -5.8320975),
                (49, -10.014197),
                (46, -7.0919595),
                (20, -6.033137),
                (5, -6.3501267),
                (32, -4.203919),
                (2, -5.743471),
                (36, -8.493466),
                (41, -7.60419),
                (0, -7.388545),
            ],
            // Generation 1
            [
                (18, -6.048934),
                (39, -1.1448132),
                (48, -7.921489),
                (38, -6.0117235),
                (27, -6.30289),
                (9, -6.5567093),
                (29, -5.905172),
                (25, -4.2305975),
                (40, -5.1198816),
                (24, -7.232001),
                (46, -6.5581756),
                (20, -6.7987585),
                (8, -9.346154),
                (2, -7.6944494),
                (3, -6.487195),
                (16, -8.379641),
                (32, -7.292016),
                (33, -7.91467),
                (41, -7.4449363),
                (21, -6.0500197),
                (19, -5.357873),
                (10, -6.9984064),
                (7, -5.6824636),
                (13, -8.154273),
                (45, -7.8713655),
                (47, -5.279138),
                (49, -1.915852),
                (6, -2.682654),
                (30, -5.566201),
                (1, -1.829716),
                (11, -7.7527223),
                (12, -10.379072),
                (15, -4.866212),
                (35, -8.091223),
                (36, -8.137203),
                (42, -7.2846284),
                (44, -4.7636213),
                (28, -6.518874),
                (34, 1.9858776),
                (43, -10.140268),
                (0, -3.5068736),
                (17, -2.3913155),
                (26, -6.1766686),
                (22, -9.119884),
                (14, -7.470778),
                (5, -5.925585),
                (23, -6.004782),
                (31, -2.696432),
                (4, -2.4887466),
                (37, -5.5321026),
            ],
            // Generation 2
            [
                (25, -8.760574),
                (0, -2.5970187),
                (9, -4.270929),
                (11, -0.27550858),
                (20, -6.7012835),
                (30, 2.3309054),
                (4, -7.0107384),
                (31, -7.5239167),
                (41, -2.337672),
                (6, -3.4384027),
                (16, -7.9485044),
                (37, -7.3155503),
                (38, -7.4812994),
                (3, -3.958924),
                (42, -7.738173),
                (43, -6.500585),
                (22, -6.318394),
                (17, -5.7882595),
                (45, -8.782414),
                (49, -8.84129),
                (23, -10.222613),
                (26, -6.06804),
                (32, -6.4851217),
                (33, -7.3542376),
                (34, -2.8723297),
                (27, -7.1350646),
                (8, -2.7956052),
                (18, -5.0000043),
                (10, -1.5138103),
                (2, 0.10560961),
                (7, -1.4954948),
                (35, -7.7015786),
                (36, -8.602789),
                (47, -8.117584),
                (28, -9.151132),
                (39, -8.035833),
                (13, -6.2601876),
                (15, -9.050044),
                (19, -5.465233),
                (44, -8.494604),
                (5, -6.9012084),
                (12, -9.458872),
                (21, -5.980685),
                (14, -7.7407913),
                (46, -0.701484),
                (24, -9.477325),
                (29, -6.6444407),
                (1, -3.4681067),
                (40, -5.4685316),
                (48, 0.22965483),
            ],
            // Generation 3
            [
                (11, -5.7744265),
                (12, 0.10171394),
                (18, -8.503949),
                (3, -1.9760166),
                (17, -7.895561),
                (20, -8.515409),
                (45, -1.9184738),
                (6, -5.6488137),
                (46, -6.1171823),
                (49, -7.006673),
                (29, -3.6479561),
                (37, -4.025724),
                (42, -4.1281996),
                (9, -2.7060657),
                (33, 0.18799233),
                (15, -7.8216696),
                (23, -11.02603),
                (22, -10.132984),
                (7, -6.432255),
                (38, -7.2159233),
                (10, -2.195277),
                (2, -6.7676725),
                (27, -1.8040345),
                (34, -11.214028),
                (40, -6.1334066),
                (35, -9.410227),
                (44, -0.14929143),
                (47, -7.3865366),
                (41, -9.200221),
                (26, -6.1885824),
                (13, -5.5693216),
                (31, -8.184256),
                (39, -8.06583),
                (24, -11.773471),
                (25, -15.231514),
                (14, -5.4468412),
                (30, -5.494699),
                (21, -10.619481),
                (28, -7.322004),
                (16, -7.4136076),
                (8, -3.2260292),
                (32, -8.187313),
                (19, -5.9347467),
                (43, -0.112977505),
                (5, -1.9279568),
                (48, -3.8396995),
                (0, -9.317253),
                (4, -1.8099403),
                (1, -5.4981036),
                (36, -3.5487309),
            ],
            // Generation 4
            [
                (28, -6.2057357),
                (40, -6.9324327),
                (46, -0.5130272),
                (23, -7.9489794),
                (47, -7.3411865),
                (20, -8.930363),
                (26, -3.238875),
                (41, -7.376683),
                (48, -0.83026105),
                (27, -10.048681),
                (36, -5.1788163),
                (30, -8.002236),
                (9, -7.4656434),
                (4, -3.8850121),
                (16, -3.1768656),
                (11, 1.0195583),
                (44, -8.7163315),
                (45, -6.7038856),
                (33, -6.974304),
                (22, -10.026589),
                (13, -4.342838),
                (12, -6.69588),
                (31, -2.2994905),
                (14, -7.9772606),
                (32, -10.55702),
                (38, -5.668454),
                (34, -10.026564),
                (37, -8.128912),
                (42, -10.7178335),
                (17, -5.18195),
                (49, -9.900299),
                (21, -12.4000635),
                (8, -1.8514707),
                (29, -3.365313),
                (39, -5.588918),
                (43, -8.482417),
                (1, -4.390686),
                (35, -5.604909),
                (24, -7.1810236),
                (25, -5.9158974),
                (19, -4.5733366),
                (0, -5.68081),
                (3, -2.8414884),
                (6, -1.5809858),
                (7, -9.295659),
                (5, -3.7936096),
                (10, -4.088697),
                (2, -2.3494315),
                (15, -7.3323736),
                (18, -7.7137175),
            ],
            // Generation 5
            [
                (1, -2.7719336),
                (37, -6.097855),
                (39, -4.1296787),
                (2, -5.4538774),
                (34, -11.808794),
                (40, -9.822159),
                (3, -7.884645),
                (42, -14.777964),
                (32, -2.6564443),
                (16, -5.2442584),
                (9, -6.2919874),
                (48, -2.4359574),
                (25, -11.707236),
                (33, -5.5483084),
                (35, -0.3632618),
                (7, -4.3673687),
                (27, -8.139543),
                (12, -9.019396),
                (17, -0.029791832),
                (24, -8.63045),
                (18, -11.925819),
                (20, -9.040375),
                (44, -10.296264),
                (47, -15.95397),
                (23, -12.38116),
                (21, 0.18342426),
                (38, -7.695002),
                (6, -8.710346),
                (28, -2.8542902),
                (5, -2.077858),
                (10, -3.638583),
                (8, -7.360152),
                (15, -7.1610765),
                (29, -4.8372035),
                (45, -11.499393),
                (13, -3.8436065),
                (22, -5.472387),
                (11, -4.259357),
                (26, -4.847328),
                (4, -2.0376666),
                (36, -7.5392637),
                (41, -5.3857164),
                (19, -8.576212),
                (14, -8.267895),
                (30, -4.0456495),
                (31, -3.806975),
                (43, -7.9901657),
                (46, -7.181662),
                (0, -7.502816),
                (49, -7.3067017),
            ],
            // Generation 6
            [
                (17, -9.793276),
                (27, -2.8843281),
                (38, -8.737534),
                (8, -1.5083166),
                (16, -8.267393),
                (42, -8.055011),
                (47, -2.0843022),
                (14, -3.9945045),
                (30, -10.208374),
                (26, -3.2439823),
                (49, -2.5527742),
                (25, -10.359426),
                (9, -4.4744225),
                (19, -7.2775927),
                (3, -7.282045),
                (36, -8.503307),
                (40, -12.083569),
                (22, -3.7249084),
                (18, -7.5065627),
                (41, -3.3326488),
                (44, -2.76882),
                (45, -12.154654),
                (24, -2.8332536),
                (5, -5.2674284),
                (4, -4.105483),
                (10, -6.930478),
                (20, -3.7845988),
                (2, -4.4593267),
                (28, -0.3003047),
                (29, -6.5971193),
                (32, -5.0542274),
                (33, -9.068264),
                (43, -7.124672),
                (46, -8.358111),
                (23, -5.551978),
                (11, -7.7810373),
                (35, -7.4763336),
                (34, -10.868844),
                (39, -10.51066),
                (7, -4.376377),
                (48, -9.093265),
                (6, -0.20033613),
                (1, -6.125786),
                (12, -8.243349),
                (0, -7.1646323),
                (13, -3.7055316),
                (15, -6.295897),
                (21, -5.929867),
                (31, -7.2123885),
                (37, -2.482071),
            ],
            // Generation 7
            [
                (30, -12.467585),
                (14, -5.1706576),
                (40, -9.03964),
                (18, -5.7730474),
                (41, -9.061858),
                (20, -2.8577142),
                (24, -3.3558655),
                (42, -7.902747),
                (43, -6.1566644),
                (21, -5.4271364),
                (23, -7.1462164),
                (44, -7.9898252),
                (11, -2.493559),
                (31, -4.6718645),
                (48, -12.774545),
                (8, -7.252562),
                (35, -1.6866531),
                (49, -4.437603),
                (45, -7.164916),
                (7, -4.613396),
                (32, -8.156101),
                (39, -10.887325),
                (0, -0.18116185),
                (47, -4.998584),
                (10, -8.914183),
                (13, -0.8690014),
                (27, -0.3714923),
                (28, -12.002966),
                (9, -6.2789965),
                (26, -0.46416503),
                (2, -9.865377),
                (29, -8.443848),
                (46, -6.3264246),
                (3, -7.807205),
                (4, -6.8240366),
                (5, -6.843891),
                (12, -5.6381693),
                (15, -4.6679296),
                (36, -6.8010025),
                (16, -8.222928),
                (25, -10.326822),
                (34, -6.0182467),
                (37, -8.713378),
                (38, -7.549215),
                (17, -7.247555),
                (22, -13.296148),
                (33, -8.542955),
                (19, -7.254419),
                (1, -2.8472056),
                (6, -5.898753),
            ],
            // Generation 8
            [
                (7, -3.6624274),
                (4, -2.9281456),
                (39, -5.9176188),
                (13, -8.0644045),
                (16, -2.0319564),
                (49, -10.309226),
                (3, -0.21671781),
                (37, -8.295551),
                (44, -16.496105),
                (46, -6.2466326),
                (47, -3.5928986),
                (19, -9.298591),
                (1, -7.937351),
                (15, -8.218504),
                (6, -6.945601),
                (25, -8.446054),
                (12, -5.8477135),
                (14, -3.9165816),
                (17, -2.4864268),
                (20, -7.97737),
                (22, -5.347026),
                (0, -6.0739775),
                (32, -6.7568192),
                (36, -4.730008),
                (28, -9.923819),
                (38, -8.677519),
                (42, -4.668519),
                (48, 0.14014988),
                (5, -8.3167),
                (8, -2.5030074),
                (21, -1.8195568),
                (27, -6.111103),
                (45, -12.708131),
                (35, -8.089076),
                (11, -6.0151362),
                (34, -13.688166),
                (33, -11.375975),
                (2, -4.1082373),
                (24, -4.0867376),
                (10, -4.2828474),
                (41, -9.174506),
                (43, -1.1505331),
                (29, -3.7704785),
                (18, -4.9493446),
                (30, -3.727829),
                (31, -6.490308),
                (9, -6.0947385),
                (40, -9.492185),
                (26, -13.629112),
                (23, -9.773454),
            ],
            // Generation 9
            [
                (12, -1.754871),
                (41, 2.712658),
                (24, -4.0929146),
                (18, -4.9418926),
                (44, -9.325021),
                (8, -6.4423165),
                (1, -0.0946085),
                (5, -3.0156248),
                (14, -5.29519),
                (34, -10.763539),
                (11, -7.304751),
                (20, -6.8397574),
                (22, -5.6720686),
                (23, -7.829904),
                (7, -3.8627372),
                (6, -3.1108487),
                (16, -8.803584),
                (36, -13.916307),
                (21, -10.142917),
                (37, -12.171498),
                (45, -13.004938),
                (19, -3.7237267),
                (47, -6.0189786),
                (17, -4.612711),
                (15, -5.3010545),
                (30, -5.671092),
                (46, -13.300519),
                (25, -8.2948),
                (3, -10.556543),
                (42, -7.041272),
                (48, -9.797744),
                (9, -5.6163936),
                (26, -6.665021),
                (27, -7.074666),
                (4, -1.5992731),
                (2, -6.4931273),
                (29, -3.9785416),
                (31, -12.222026),
                (10, -2.3970482),
                (40, -6.204074),
                (49, -7.025599),
                (28, -8.562909),
                (13, -6.2592154),
                (32, -10.465271),
                (33, -7.7043953),
                (35, -6.4584246),
                (38, -2.9016697),
                (39, -1.5256255),
                (43, -10.858711),
                (0, -4.720929),
            ],
            //Generation 10
            [
                (2, -5.1676617),
                (3, -4.521774),
                (29, -7.3104324),
                (23, -6.550776),
                (26, -10.467587),
                (18, 1.6576093),
                (33, -2.564094),
                (20, -3.2697926),
                (35, -13.577334),
                (37, -6.0147185),
                (17, -4.07909),
                (0, -9.630419),
                (38, -7.011383),
                (12, -10.686635),
                (43, -8.94728),
                (48, -9.350017),
                (30, -7.3335466),
                (13, -7.7690034),
                (4, -2.3488472),
                (14, -7.2594194),
                (21, -9.08367),
                (34, -7.7497597),
                (8, -6.2317214),
                (27, -8.440135),
                (22, -4.4437346),
                (32, -2.194015),
                (28, -6.6919556),
                (40, -8.840385),
                (42, -9.781796),
                (15, -7.3304253),
                (49, -8.720987),
                (19, -9.044103),
                (6, -5.715863),
                (41, -8.395639),
                (36, -3.995482),
                (25, -9.1373005),
                (5, -7.5690002),
                (1, -6.0397635),
                (16, -8.231512),
                (10, -6.5344634),
                (44, -7.749376),
                (7, -9.302668),
                (31, -10.868391),
                (39, -2.7578635),
                (47, -6.964238),
                (24, -4.033315),
                (11, -8.211409),
                (45, -10.472969),
                (9, -7.1529093),
                (46, -9.653514),
            ],
        ];

        // Transform scores into a vector of hashmaps instead
        let scores: Vec<HashMap<u64, f32>> = scores
            .iter()
            .map(|gen_scores| gen_scores.iter().cloned().collect())
            .collect();

        assert!(
            should_continue(scores[..0].as_ref(), 5)
                .expect("Failed to determine if the simulation should continue")
                == true
        );
        assert!(
            should_continue(scores[..1].as_ref(), 5)
                .expect("Failed to determine if the simulation should continue")
                == true
        );
        assert!(
            should_continue(scores[..2].as_ref(), 5)
                .expect("Failed to determine if the simulation should continue")
                == true
        );
        assert!(
            should_continue(scores[..3].as_ref(), 5)
                .expect("Failed to determine if the simulation should continue")
                == true
        );
        assert!(
            should_continue(scores[..4].as_ref(), 5)
                .expect("Failed to determine if the simulation should continue")
                == true
        );
        assert!(
            should_continue(scores[..5].as_ref(), 5)
                .expect("Failed to determine if the simulation should continue")
                == true
        );
        assert!(
            should_continue(scores[..6].as_ref(), 5)
                .expect("Failed to determine if the simulation should continue")
                == true
        );
        assert!(
            should_continue(scores[..7].as_ref(), 5)
                .expect("Failed to determine if the simulation should continue")
                == true
        );
        assert!(
            should_continue(scores[..8].as_ref(), 5)
                .expect("Failed to determine if the simulation should continue")
                == false
        );
        assert!(
            should_continue(scores[..9].as_ref(), 5)
                .expect("Failed to determine if the simulation should continue")
                == false
        );
        assert!(
            should_continue(scores[..10].as_ref(), 5)
                .expect("Failed to determine if the simulation should continue")
                == false
        );
    }
}
