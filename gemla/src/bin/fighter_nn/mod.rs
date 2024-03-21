extern crate fann;

use std::{fs::{self, File}, io::{self, BufRead, BufReader}, path::{Path, PathBuf}};
use fann::{ActivationFunc, Fann};
use futures::future::join_all;
use gemla::{core::genetic_node::{GeneticNode, GeneticNodeContext}, error::Error};
use rand::prelude::*;
use rand::distributions::{Distribution, Uniform};
use serde::{Deserialize, Serialize};
use anyhow::Context;
use uuid::Uuid;
use std::collections::HashMap;
use tokio::process::Command;
use async_trait::async_trait;

const BASE_DIR: &str = "F:\\\\vandomej\\Projects\\dootcamp-AI-Simulation\\Simulations";
const POPULATION: usize = 50;
const NEURAL_NETWORK_SHAPE: &[u32; 5] = &[14, 20, 20, 12, 8];
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
    
        // Create the first generation in this folder
        for i in 0..POPULATION {
            // Filenames are stored in the format of "xxxxxx_fighter_nn_0.net", "xxxxxx_fighter_nn_1.net", etc. Where xxxxxx is the folder name
            let nn = gen_folder.join(format!("{:06}_fighter_nn_{}.net", context.id, i));
            let mut fann = Fann::new(NEURAL_NETWORK_SHAPE)
                .with_context(|| "Failed to create nn")?;
            fann.randomize_weights(-0.8, 0.8);
            fann.set_activation_func_hidden(ActivationFunc::SigmoidSymmetric);
            fann.set_activation_func_output(ActivationFunc::SigmoidSymmetric);
            // This will overwrite any existing file with the same name
            fann.save(&nn)
                .with_context(|| format!("Failed to save nn at {:?}", nn))?;
        }
    
        Ok(Box::new(FighterNN {
            id: context.id,
            folder,
            population_size: POPULATION,
            generation: 0,
            scores: vec![HashMap::new()],
        }))
    }

    async fn simulate(&mut self, _context: GeneticNodeContext) -> Result<(), Error> {
        // For each nn in the current generation:
        for i in 0..self.population_size {
            // load the nn
            let nn = self.folder.join(format!("{}", self.generation)).join(format!("{:06}_fighter_nn_{}.net", self.id, i));
            let mut simulations = Vec::new();
    
            // Using the same original nn, repeat the simulation with 5 random nn's from the current generation concurrently
            for _ in 0..SIMULATION_ROUNDS {
                let random_nn_index = thread_rng().gen_range(0..self.population_size);
                let id = self.id.clone();
                let folder = self.folder.clone();
                let generation = self.generation;

                let random_nn = folder.join(format!("{}", generation)).join(format!("{:06}_fighter_nn_{}.net", id, random_nn_index));
                let nn_clone = nn.clone(); // Clone the path to use in the async block
    
                let config1_arg = format!("-NN1Config=\"{}\"", nn_clone.to_str().unwrap());
                let config2_arg = format!("-NN2Config=\"{}\"", random_nn.to_str().unwrap());
                let disable_unreal_rendering_arg = "-nullrhi".to_string();
    
                let future = async move {
                    // Construct the score file path
                    let nn_id = format!("{:06}_fighter_nn_{}", id, i);
                    let random_nn_id = format!("{:06}_fighter_nn_{}", id, random_nn_index);
                    let score_file_name = format!("{}_vs_{}.txt", nn_id, random_nn_id);
                    let score_file = folder.join(format!("{}", generation)).join(&score_file_name);

                    // Check if score file already exists before running the simulation
                    if score_file.exists() {
                        let round_score = read_score_from_file(&score_file, &nn_id)
                            .with_context(|| format!("Failed to read score from file: {:?}", score_file_name))?;
                        return Ok::<f32, Error>(round_score);
                    }

                    // Check if the opposite round score has been determined
                    let opposite_score_file = folder.join(format!("{}", generation)).join(format!("{}_vs_{}.txt", random_nn_id, nn_id));
                    if opposite_score_file.exists() {
                        let round_score = read_score_from_file(&opposite_score_file, &nn_id)
                            .with_context(|| format!("Failed to read score from file: {:?}", opposite_score_file))?;
                        return Ok::<f32, Error>(1.0 - round_score);
                    }

                    if thread_rng().gen_range(0..100) < 4 {
                        let _output = Command::new(GAME_EXECUTABLE_PATH)
                            .arg(&config1_arg)
                            .arg(&config2_arg)
                            .output()
                            .await
                            .expect("Failed to execute game");
                    } else {
                        let _output = Command::new(GAME_EXECUTABLE_PATH)
                            .arg(&config1_arg)
                            .arg(&config2_arg)
                            .arg(&disable_unreal_rendering_arg)
                            .output()
                            .await
                            .expect("Failed to execute game");
                    }
    
                    // Read the score from the file
                    let round_score = read_score_from_file(&score_file, &nn_id)
                        .with_context(|| format!("Failed to read score from file: {:?}", score_file_name))?;

                    Ok::<f32, Error>(round_score)
                };
    
                simulations.push(future);
            }
    
            // Wait for all simulation rounds to complete
            let results: Result<Vec<f32>, Error> = join_all(simulations).await.into_iter().collect();
    
            let score = results?.into_iter().sum::<f32>() / SIMULATION_ROUNDS as f32;
            trace!("NN {:06}_fighter_nn_{} scored {}", self.id, i, score);
            self.scores[self.generation as usize].insert(i as u64, score);
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
            let mut fann = Fann::from_file(&nn)
                .with_context(|| format!("Failed to load nn"))?;

            // Load another nn from the current generation and cross breed it with the current nn
            let cross_nn = self.folder.join(format!("{}", self.generation)).join(format!("{:06}_fighter_nn_{}.net", self.id, to_keep[thread_rng().gen_range(0..survivor_count)]));
            let cross_fann = Fann::from_file(&cross_nn)
                .with_context(|| format!("Failed to load cross nn"))?;

            let mut connections = fann.get_connections(); // Vector of connections
            let cross_connections = cross_fann.get_connections(); // Vector of connections
            let segment_count: usize = 3; // For example, choose 3 segments to swap
            let segment_distribution = Uniform::from(1..connections.len() / segment_count); // Ensure segments are not too small

            let mut start_points = vec![];

            for _ in 0..segment_count {
                let start_point = segment_distribution.sample(&mut rand::thread_rng());
                start_points.push(start_point);
            }
            start_points.sort_unstable(); // Ensure segments are in order
            
            for (j, &start) in start_points.iter().enumerate() {
                let end = if j < segment_count - 1 {
                    start_points[j + 1]
                } else {
                    connections.len()
                };

                // Swap segments
                for k in start..end {
                    connections[k] = cross_connections[k].clone();
                }
            }

            fann.set_connections(&connections);

            // For each weight in the 5 new nn's there is a 20% chance of a minor mutation (a random number between -0.1 and 0.1 is added to the weight)
            // And a 5% chance of a major mutation (a random number between -0.3 and 0.3 is added to the weight)
            let mut connections = fann.get_connections(); // Vector of connections
            for c in &mut connections {
                if thread_rng().gen_range(0..100) < 20 {
                    c.weight += thread_rng().gen_range(-0.1..0.1);
                }
                // else if thread_rng().gen_range(0..100) < 5 {
                //     c.weight += thread_rng().gen_range(-0.3..0.3);
                // }
            }
            fann.set_connections(&connections);

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
        }))
    }
}

fn read_score_from_file(file_path: &Path, nn_id: &str) -> Result<f32, io::Error> {
    let file = File::open(file_path)?;
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

    Err(io::Error::new(
        io::ErrorKind::NotFound,
        "NN ID not found in scores file",
    ))
}