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
            let nn = gen_folder.join(format!("{:06}_fighter_nn_{}.net", context.id, i));

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
                    .join(self_clone.get_individual_id(i as u64));
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
                        .join(self_clone.get_individual_id(random_nn_index as u64));
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
        sorted_scores.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());
        let to_keep = sorted_scores[survivor_count..]
            .iter()
            .map(|(k, _)| *k)
            .collect::<Vec<_>>();

        // Save the remaining 5 nn's to the new generation folder
        for (i, nn_id) in to_keep.iter().enumerate().take(survivor_count) {
            let nn = self
                .folder
                .join(format!("{}", self.generation))
                .join(format!("{:06}_fighter_nn_{}.net", self.id, nn_id));
            let new_nn = new_gen_folder.join(format!("{:06}_fighter_nn_{}.net", self.id, i));
            fs::copy(&nn, &new_nn)?;
        }

        // Take the remaining 5 nn's and create 5 new nn's by the following:
        for i in 0..survivor_count {
            let nn_id = to_keep[i];
            let nn = self
                .folder
                .join(format!("{}", self.generation))
                .join(format!("{:06}_fighter_nn_{}.net", self.id, nn_id));
            let fann = Fann::from_file(&nn).with_context(|| "Failed to load nn")?;

            // Load another nn from the current generation and cross breed it with the current nn
            let cross_nn = self
                .folder
                .join(format!("{}", self.generation))
                .join(format!(
                    "{:06}_fighter_nn_{}.net",
                    self.id,
                    to_keep[thread_rng().gen_range(0..survivor_count)]
                ));
            let cross_fann =
                Fann::from_file(&cross_nn).with_context(|| "Failed to load cross nn")?;

            let mut new_fann = crossbreed(self, &fann, &cross_fann, self.crossbreed_segments)?;

            // For each weight in the 5 new nn's there is a 20% chance of a minor mutation (a random number between -0.1 and 0.1 is added to the weight)
            // And a 5% chance of a major mutation a new neuron is randomly added to a hidden layer
            let mut connections = new_fann.get_connections(); // Vector of connections
            for c in &mut connections {
                if thread_rng().gen_range(0.0..1.0) < self.minor_mutation_rate {
                    trace!("Minor mutation on connection {:?}", c);
                    c.weight += thread_rng().gen_range(self.weight_initialization_range.clone());
                    trace!("New weight: {}", c.weight);
                }
            }

            new_fann.set_connections(&connections);

            if thread_rng().gen_range(0.0..1.0) < self.major_mutation_rate {
                new_fann = major_mutation(&new_fann, self.weight_initialization_range.clone())?;
            }

            // Save the new nn's to the new generation folder
            let new_nn = new_gen_folder.join(format!(
                "{:06}_fighter_nn_{}.net",
                self.id,
                i + survivor_count
            ));
            new_fann
                .save(&new_nn)
                .with_context(|| "Failed to save nn")?;
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
            sorted_scores.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());
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

        for _ in 0..max(left.population_size, right.population_size) * SIMULATION_ROUNDS {
            let left_nn_id = left_scores[thread_rng().gen_range(0..left_scores.len())].0;
            let right_nn_id = right_scores[thread_rng().gen_range(0..right_scores.len())].0;

            let left_nn_path = left
                .folder
                .join(left.generation.to_string())
                .join(left.get_individual_id(left_nn_id));
            let right_nn_path = right
                .folder
                .join(right.generation.to_string())
                .join(right.get_individual_id(right_nn_id));
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

        let total_left_score = scores.iter().map(|(l, _)| l).sum::<f32>();
        let total_right_score = scores.iter().map(|(_, r)| r).sum::<f32>();

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

    // debug!("the following command {} {} {} {}", GAME_EXECUTABLE_PATH, config1_arg, config2_arg, disable_unreal_rendering_arg);

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
