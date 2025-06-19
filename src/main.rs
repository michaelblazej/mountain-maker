mod dla_2d;

use anyhow::Result;
use dla_2d::{DlaSimulation, DlaParameters, Array2D};
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
    
    // Convert to grid and export
    let grid = simulation.to_grid();
    let output_path = "dla_base_grid.txt";
    export_grid_to_file(&grid, output_path)?;
    println!("Base DLA grid exported to {}", output_path);
    
    // Print some statistics
    let (base_width, base_height) = simulation.get_dimensions();
    
    println!("\nStatistics:");
    println!("  Base simulation:  {} particles, {}x{} dimensions", 
             simulation.particles.len(), base_width, base_height);
    
    println!("\nDone! You can visualize the CSV files with a spreadsheet program ");
    println!("or using a visualization tool to see the patterns created.");
    
    Ok(())
}

/// Export a DLA simulation to a CSV file
fn export_to_csv(simulation: &DlaSimulation, path: &str) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    
    // Write header
    writeln!(writer, "x,y,index")?;
    
    // Write each particle position
    for (particle, idx) in &simulation.particles {
        writeln!(writer, "{},{},{}", particle.x, particle.y, idx)?;
    }
    
    Ok(())
}

/// Export a 2D grid to a text file
fn export_grid_to_file(grid: &Array2D, path: &str) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    
    // Write grid dimensions as a header comment
    writeln!(writer, "# Grid dimensions: {} x {}", grid.width(), grid.height())?;
    
    // Write each row of the grid
    for y in 0..grid.height() {
        let mut line = String::with_capacity(grid.width());
        for x in 0..grid.width() {
            if let Some(value) = grid.get(x, y) {
                if value > 0 {
                    line.push('#'); // Use '#' character for particles
                } else {
                    line.push('.');  // Use '.' for empty space
                }
            }
        }
        writeln!(writer, "{}", line)?;
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
