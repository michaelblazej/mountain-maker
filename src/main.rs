mod dla_2d;
mod blur;

use anyhow::Result;
use dla_2d::{DlaSimulation, DlaParameters, Array2D};
use blur::{upsample, upsample_and_blur, box_blur, mean_filter, gaussian_filter, BlurOptions};
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
    
    // Convert to grid and export at different resolutions
    // Standard resolution (1:1)
    let grid = simulation.to_grid(1);
    let output_path = "dla_base_grid.txt";
    export_grid_to_file(&grid, output_path)?;
    println!("Base DLA grid exported to {}", output_path);
    
    // Higher resolution with connected points
    let high_res_grid = simulation.to_grid(3);
    let output_path = "dla_base_grid_hires.txt";
    export_grid_to_file(&high_res_grid, output_path)?;
    println!("High-resolution DLA grid exported to {}", output_path);
    
    // Create an upsampled grid with linear interpolation
    let upsampled_grid = upsample(&grid, 4, Some(BlurOptions { strength: 0.7 }));
    let output_path = "dla_base_grid_upsampled.txt";
    export_grid_to_file(&upsampled_grid, output_path)?;
    println!("Upsampled DLA grid exported to {}", output_path);
    
    // Create a blurred version of the upsampled grid
    let blurred_grid = box_blur(&upsampled_grid, 1);
    let output_path = "dla_base_grid_blurred.txt";
    export_grid_to_file(&blurred_grid, output_path)?;
    println!("Blurred DLA grid exported to {}", output_path);
    
    // Combined upsampling and blurring in one step
    let smooth_grid = upsample_and_blur(&grid, 4, 1);
    let output_path = "dla_base_grid_smooth.txt";
    export_grid_to_file(&smooth_grid, output_path)?;
    println!("Smooth DLA grid exported to {}", output_path);
    
    // Apply mean filter convolution
    let mean_grid = mean_filter(&upsampled_grid, 3);
    let output_path = "dla_base_grid_mean.txt";
    export_grid_to_file(&mean_grid, output_path)?;
    println!("Mean filtered grid exported to {}", output_path);
    
    // Apply Gaussian filter convolution
    let gaussian_grid = gaussian_filter(&upsampled_grid, 5, 1.0);
    let output_path = "dla_base_grid_gaussian.txt";
    export_grid_to_file(&gaussian_grid, output_path)?;
    println!("Gaussian filtered grid exported to {}", output_path);
    
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
    writeln!(writer, "x,y,index,stuck_to")?;
    
    // Write each particle position
    for (particle, data) in &simulation.particles {
        writeln!(writer, "{},{},{},{}", particle.x, particle.y, data.index, data.stuck_to)?;
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
