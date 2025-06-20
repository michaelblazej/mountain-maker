mod dla_2d;
mod blur;
mod export;

use anyhow::Result;
use dla_2d::{DlaSimulation, DlaParameters, Array2D};
use blur::{upsample, upsample_and_blur, box_blur, mean_filter, gaussian_filter, BlurOptions};
use export::export_array_to_glb;
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
    
    // Start to build the mountains
    let blur_options = Some(BlurOptions{strength: 0.7});
    let blur_radius = 8; // Blur radius for upsample_and_blur
    let upsample_factor = 2; // Upscale factor for each iteration
    let num_steps = 4; // Number of iterations
    
    // Initialize with the base grid
    let mut current_grid = simulation.to_grid(1);
    let mut current_scale = 1;
    
    // Loop through the steps to iteratively refine the mountain
    for step in 1..=num_steps {
        // Calculate the scale for the fine grid in this step
        current_scale *= upsample_factor;
        
        // Step 1: Blur and upsample the current grid
        let grid_blur = upsample_and_blur(&current_grid, upsample_factor, blur_radius);
        
        // Step 2: Get a fine grid at the new scale
        let mut grid_fine = simulation.to_grid(current_scale);
        grid_fine.normalize(0.1);
        
        // Step 3: Combine the blurred and fine grids
        let grid_sum = grid_blur + grid_fine;

        let grid_sum  = box_blur(&grid_sum,blur_radius);
        
        // Step 4: Save the result
        let output_path = format!("dla_grid_step{}.txt", step);
        if let Err(e) = export_grid_to_file(&grid_sum, &output_path) {
            eprintln!("Error exporting grid for step {}: {}", step, e);
        } else {
            println!("Exported mountain grid for step {} to {}", step, output_path);
        }
        
        // Use this grid as the basis for the next iteration
        // Normalize the grid so maximum value is 1.0
        let mut next_grid = grid_sum.clone();
        next_grid.normalize(1.0);
        current_grid = next_grid;
    }
    
    // Export the final grid as a GLB file
    let final_grid = &current_grid;
    let output_glb_path = "mountain_mesh.glb";
    
    println!("Exporting 3D mesh to GLB file: {}", output_glb_path);
    // Scale factors can be adjusted to control the appearance of the terrain
    if let Err(e) = export_array_to_glb(final_grid, 1.0, 1.0, 10.0, output_glb_path) {
        eprintln!("Error exporting to GLB: {}", e);
    } else {
        println!("Successfully exported 3D mesh to: {}", output_glb_path);
    }
    
    println!("\nDone! You can visualize the CSV files with a spreadsheet program ");
    println!("or use a 3D viewer to open the GLB file and see the 3D terrain model.");
    
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

/// Export a 2D grid to a CSV file that visualize.py can read
fn export_grid_to_file(grid: &Array2D, path: &str) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    
    // Write grid dimensions as a header comment
    writeln!(writer, "# Grid dimensions: {} x {}", grid.width(), grid.height())?;
    
    // Write each row of the grid as CSV
    for y in 0..grid.height() {
        let mut line = String::new();
        
        for x in 0..grid.width() {
            // Add comma separator between values (except for first value)
            if x > 0 {
                line.push(',');
            }
            
            // Write the value directly (as int or float)
            if let Some(value) = grid.get(x, y) {
                // For cleaner visualizations, we'll keep 0.0 and 1.0 as integers
                if value == 0.0 || value == 1.0 {
                    line.push_str(&format!("{}", value as i32));
                } else {
                    // Format floating point values with limited precision
                    line.push_str(&format!("{:.3}", value));
                }
            } else {
                line.push('0');
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
