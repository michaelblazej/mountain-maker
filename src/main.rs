mod dla_2d;

use anyhow::Result;
use dla_2d::{DlaSimulation, DlaParameters};
use std::fs::File;
use std::io::{BufWriter, Write};

fn main() -> Result<()> {
    println!("Mountain Maker - 2D Diffusion Limited Aggregation");
    
    // Create a basic DLA simulation
    let mut params = DlaParameters::default();
    params.num_particles = 5000; // Use a smaller number for faster testing
    params.stickiness = 0.7;     // Higher stickiness means particles stick more easily
    
    // Set dimensions for the simulation
    let width = 200;
    let height = 200;
    
    // Create and run the DLA simulation
    println!("Creating DLA simulation with size {}x{}...", width, height);
    let mut simulation = DlaSimulation::with_params(width, height, params);
    
    println!("Running DLA simulation...");
    simulation.run()?;
    
    // Export the base simulation to a CSV file
    let output_path = "dla_base.csv";
    export_to_csv(&simulation, output_path)?;
    println!("Base DLA simulation exported to {}", output_path);
    
    // Create a new simulation with increased dimensions
    println!("Creating scaled-up simulation with 2.5x dimensions...");
    let scale_factor = 2.5;
    let scaled_simulation = DlaSimulation::with_increased_dimensions(&simulation, scale_factor);
    
    // Export the scaled simulation to a CSV file
    let output_path = "dla_scaled.csv";
    export_to_csv(&scaled_simulation, output_path)?;
    println!("Scaled DLA simulation exported to {}", output_path);
    
    // Print some statistics
    let (base_width, base_height) = simulation.get_dimensions();
    let (scaled_width, scaled_height) = scaled_simulation.get_dimensions();
    
    println!("\nStatistics:");
    println!("  Base simulation:  {} particles, {}x{} dimensions", 
             simulation.particles.len(), base_width, base_height);
    println!("  Scaled simulation: {} particles, {}x{} dimensions", 
             scaled_simulation.particles.len(), scaled_width, scaled_height);
    
    println!("\nDone! You can visualize the CSV files with a spreadsheet program ");
    println!("or using a visualization tool to see the patterns created.");
    
    Ok(())
}

/// Export a DLA simulation to a CSV file
fn export_to_csv(simulation: &DlaSimulation, path: &str) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    
    // Write header
    writeln!(writer, "x,y")?;
    
    // Write each particle position
    for particle in &simulation.particles {
        writeln!(writer, "{},{}", particle.x, particle.y)?;
    }
    
    Ok(())
}

/// Example of how to create a DLA simulation with different parameters
fn _example_parameter_variations() -> Result<()> {
    // Create a more sparse structure
    let _sparse_params = DlaParameters {
        num_particles: 3000,
        stickiness: 0.3,     // Lower stickiness creates more branching structures
        step_size: 1.5,      // Larger step size
        spawn_radius_factor: 1.8, // Spawn particles further away
        max_steps_per_particle: 10000,
    };
    
    // Create a denser structure
    let _dense_params = DlaParameters {
        num_particles: 8000,
        stickiness: 0.9,     // Higher stickiness creates more compact structures
        step_size: 0.5,      // Smaller step size
        spawn_radius_factor: 1.2, // Spawn particles closer to the structure
        max_steps_per_particle: 3000,
    };
    
    println!("Parameter variation examples defined");
    Ok(())
}
