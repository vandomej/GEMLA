extern crate fann;

use std::{fs, path::PathBuf};
use fann::{ActivationFunc, Fann};
use gemla::{core::genetic_node::{GeneticNode, GeneticNodeContext}, error::Error};
use rand::prelude::*;
use rand::distributions::{Distribution, Uniform};
use serde::{Deserialize, Serialize};
use anyhow::Context;
use uuid::Uuid;
use std::collections::HashMap;

const BASE_DIR: &str = "F:\\\\vandomej\\Projects\\dootcamp-AI-Simulation\\Simulations";
const POPULATION: usize = 100;
const NEURAL_NETWORK_SHAPE: &[u32; 3] = &[10, 10, 10];
const SIMULATION_ROUNDS: usize = 10;
const SURVIVAL_RATE: f32 = 0.5;

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

impl GeneticNode for FighterNN {
    // Check for the highest number of the folder name and increment it by 1
    fn initialize(context: &GeneticNodeContext) -> Result<Box<Self>, Error> {
        let base_path = PathBuf::from(BASE_DIR);

        let mut folder = base_path.join(format!("fighter_nn_{:06}", context.id));
        fs::create_dir(&folder)?;

        //Create a new directory for the first generation
        let gen_folder = folder.join("0");
        fs::create_dir(&gen_folder)?;

        // Create the first generation in this folder
        for i in 0..POPULATION {
            // Filenames are stored in the format of "xxxxxx_fighter_nn_0.net", "xxxxxx_fighter_nn_1.net", etc. Where xxxxxx is the folder name
            let nn = gen_folder.join(format!("{:06}_fighter_nn_{}.net", context.id, i));
            let mut fann = Fann::new(NEURAL_NETWORK_SHAPE)
                .with_context(|| format!("Failed to create nn"))?;
            fann.set_activation_func_hidden(ActivationFunc::SigmoidSymmetric);
            fann.set_activation_func_output(ActivationFunc::SigmoidSymmetric);
            fann.save(&nn)
                .with_context(|| format!("Failed to save nn"))?;
        }

        Ok(Box::new(FighterNN {
            id: context.id,
            folder,
            population_size: POPULATION,
            generation: 0,
            scores: vec![HashMap::new()],
        }))
    }

    fn simulate(&mut self, _context: &GeneticNodeContext) -> Result<(), Error> {
        // For each nn in the current generation:
        for i in 0..self.population_size {
            // load the nn
            let nn = self.folder.join(format!("{}", self.generation)).join(format!("{:06}_fighter_nn_{}.net", self.id, i));
            let fann = Fann::from_file(&nn)
                .with_context(|| format!("Failed to load nn"))?;

            // Simulate the nn against the random nn
            let mut score = 0.0;

            // Using the same original nn, repeat the simulation with 5 random nn's from the current generation
            for _ in 0..SIMULATION_ROUNDS {
                let random_nn = self.folder.join(format!("{}", self.generation)).join(format!("{:06}_fighter_nn_{}.net", self.id, thread_rng().gen_range(0..self.population_size)));
                let random_fann = Fann::from_file(&random_nn)
                    .with_context(|| format!("Failed to load random nn"))?;

                let inputs: Vec<f32> = (0..10).map(|_| thread_rng().gen_range(-1.0..1.0)).collect();
                let outputs = fann.run(&inputs)
                    .with_context(|| format!("Failed to run nn"))?;
                let random_outputs = random_fann.run(&inputs)
                    .with_context(|| format!("Failed to run random nn"))?;
                
                // Average the difference between the outputs of the nn and random_nn and add the result  to score
                let mut round_score = 0.0;
                for (o, r) in outputs.iter().zip(random_outputs.iter()) {
                    round_score += o - r;
                }
                score += round_score / fann.get_num_output() as f32;

            }

            score /= 5.0;
            self.scores[self.generation as usize].insert(i as u64, score);
        }

        Ok(())
    }


    fn mutate(&mut self, _context: &GeneticNodeContext) -> Result<(), Error> {
        let survivor_count = (self.population_size as f32 * SURVIVAL_RATE) as usize;

        // Create the new generation folder
        let new_gen_folder = self.folder.join(format!("{}", self.generation + 1));
        fs::create_dir(&new_gen_folder)?;

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
                } else if thread_rng().gen_range(0..100) < 5 {
                    c.weight += thread_rng().gen_range(-0.3..0.3);
                }
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

        // Find next highest
        let folder = base_path.join(format!("fighter_nn_{:06}", id));
        fs::create_dir(&folder)?;

        //Create a new directory for the first generation
        let gen_folder = folder.join("0");
        fs::create_dir(&gen_folder)?;

        // Take the 5 nn's with the highest scores from the left nn's and save them to the new fighter folder
        let mut sorted_scores: Vec<_> = left.scores[left.generation as usize].iter().collect();
        sorted_scores.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());
        let mut remaining = sorted_scores[(left.population_size / 2)..].iter().map(|(k, _)| *k).collect::<Vec<_>>();
        for i in 0..(left.population_size / 2) {
            let nn = left.folder.join(format!("{}", left.generation)).join(format!("{:06}_fighter_nn_{}.net", left.id, remaining.pop().unwrap()));
            let new_nn = folder.join(format!("0")).join(format!("{:06}_fighter_nn_{}.net", id, i));
            trace!("From: {:?}, To: {:?}", &nn, &new_nn);
            fs::copy(&nn, &new_nn)
                .with_context(|| format!("Failed to copy left nn"))?;
        }

        // Take the 5 nn's with the highest scores from the right nn's and save them to the new fighter folder
        sorted_scores = right.scores[right.generation as usize].iter().collect();
        sorted_scores.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());
        remaining = sorted_scores[(right.population_size / 2)..].iter().map(|(k, _)| *k).collect::<Vec<_>>();
        for i in (right.population_size / 2)..right.population_size {
            let nn = right.folder.join(format!("{}", right.generation)).join(format!("{:06}_fighter_nn_{}.net", right.id, remaining.pop().unwrap()));
            let new_nn = folder.join(format!("0")).join(format!("{:06}_fighter_nn_{}.net", id, i));
            trace!("From: {:?}, To: {:?}", &nn, &new_nn);
            fs::copy(&nn, &new_nn)
                .with_context(|| format!("Failed to copy right nn"))?;
        }

        Ok(Box::new(FighterNN {
            id: *id,
            folder,
            generation: 0,
            population_size: POPULATION,
            scores: vec![HashMap::new()],
        }))
    }
}
