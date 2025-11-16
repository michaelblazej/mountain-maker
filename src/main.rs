mod dla_2d;
mod blur;
mod export;
mod textures;

use anyhow::Result;
use dla_2d::{DlaSimulation, DlaParameters, Array2D};
use export::export_array_to_glb;
use blur::{upsample_and_blur, box_blur, BlurOptions};
use clap::Parser;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

// Define the CLI with clap
#[derive(Parser)]
#[command(name = "Mountain Maker")]
#[command(author = "DLA Mountain Generator")]
#[command(version = "1.0")]
#[command(about = "Generate procedural 3D mountains using Diffusion Limited Aggregation", long_about = None)]
struct Cli {
    // DLA simulation parameters
    #[arg(long, default_value_t = 200)]
    #[arg(help = "Width of the simulation grid")]
    width: i32,

    #[arg(long, default_value_t = 200)]
    #[arg(help = "Height of the simulation grid")]
    height: i32,

    #[arg(long, default_value_t = 5000)]
    #[arg(help = "Number of particles to simulate")]
    particles: usize,

    #[arg(long, default_value_t = 0.7)]
    #[arg(help = "Particle stickiness (0.0-1.0)")]
    stickiness: f32,

    #[arg(long, default_value_t = 1.0)]
    #[arg(help = "Random walk step size")]
    step_size: f32,

    // Mountain generation parameters
    #[arg(long, default_value_t = 8)]
    #[arg(help = "Blur radius for smoothing")]
    blur_radius: usize,

    #[arg(long, default_value_t = 2)]
    #[arg(help = "Upscaling factor for each iteration")]
    upsample_factor: usize,

    #[arg(long, default_value_t = 4)]
    #[arg(help = "Number of refinement steps")]
    steps: usize,

    #[arg(long, default_value_t = 0.7)]
    #[arg(help = "Blur strength (0.0-1.0)")]
    blur_strength: f32,

    // Randomization control
    #[arg(long)]
    #[arg(help = "Random seed for reproducible terrain generation")]
    seed: Option<u64>,

    // Export parameters
    #[arg(short = 'o', long, default_value = "mountain_mesh.glb")]
    #[arg(help = "Output path for the GLB file")]    
    output: String,

    #[arg(long, default_value_t = 1000.0)]
    #[arg(help = "Desired X dimension (width) in world units")]
    dim_x: f32,

    #[arg(long, default_value_t = 1000.0)]
    #[arg(help = "Desired Y dimension (depth) in world units")]
    dim_y: f32,

    #[arg(long, default_value_t = 200.0)]
    #[arg(help = "Desired Z dimension (height) in world units")]
    dim_z: f32,

    #[arg(long, default_value_t = false)]
    #[arg(help = "Save intermediate step files")]
    save_steps: bool,

    #[arg(long)]
    #[arg(help = "Directory to save intermediate step files")]
    step_dir: Option<PathBuf>,

    // Texture parameters
    #[arg(long, default_value_t = 0.1)]
    #[arg(help = "Texture scale in world units (lower = more repetition)")]
    world_uv_scale: f32,

    #[arg(long, default_value_t = false)]
    #[arg(help = "Enable procedural texture generation")]
    enable_textures: bool,

    #[arg(long, default_value_t = 512)]
    #[arg(help = "Resolution of procedural textures (512, 1024, 2048)")]
    texture_resolution: u32,

    #[arg(long, default_value_t = false)]
    #[arg(help = "Generate UV test texture for debugging")]
    uv_test: bool,

    // Normal map parameters
    #[arg(long, default_value_t = false)]
    #[arg(help = "Enable procedural normal map generation (requires --enable-textures)")]
    enable_normal_maps: bool,

    #[arg(long, default_value_t = 1.0)]
    #[arg(help = "Normal map generation strength (0.5-2.0, higher = more pronounced bumps)")]
    normal_strength: f32,

    #[arg(long, default_value_t = 1.0)]
    #[arg(help = "glTF normal scale parameter (0.5-2.0, runtime bump intensity)")]
    normal_scale: f32,

    // Biome and PBR parameters
    #[arg(long, default_value = "alpine")]
    #[arg(help = "Mountain biome type: alpine, desert, volcanic, or arctic")]
    biome: String,

    #[arg(long, default_value_t = false)]
    #[arg(help = "Enable full PBR materials with roughness maps (requires --enable-textures)")]
    enable_pbr: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    
    println!("Mountain Maker - 2D Diffusion Limited Aggregation");
    
    // Create a DLA simulation with CLI parameters
    let mut params = DlaParameters::default();
    params.num_particles = cli.particles;
    params.stickiness = cli.stickiness;
    params.step_size = cli.step_size;
    
    // Create and run the DLA simulation
    println!("Creating DLA simulation with size {}x{}...", cli.width, cli.height);
    
    // Display seed and particle information
    println!("Running DLA simulation with {} particles...", cli.particles);
    match cli.seed {
        Some(seed) => println!("Using random seed: {}", seed),
        None => println!("Using random seed: [default random]"),
    }
    
    let mut simulation = DlaSimulation::with_params_and_seed(cli.width, cli.height, params, cli.seed);
    simulation.run()?;
    
    // Start to build the mountains with CLI parameters
    let _blur_options = Some(BlurOptions{strength: cli.blur_strength});
    let blur_radius = cli.blur_radius; 
    let upsample_factor = cli.upsample_factor;
    let num_steps = cli.steps;
    
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
        
        // Save fine grid if step saving is enabled
        if cli.save_steps {
            let step_dir = cli.step_dir.as_ref().map_or("".to_string(), |p| {
                format!("{}/", p.to_string_lossy())
            });
            let fine_output_path = format!("{}{}_fine_step{}.txt", step_dir, "dla_grid", step);
            if let Err(e) = export_grid_to_file(&grid_fine, &fine_output_path) {
                eprintln!("Error exporting fine grid for step {}: {}", step, e);
            } else {
                println!("Exported fine mountain grid for step {} to {}", step, fine_output_path);
            }
        }
        
        // Step 3: Combine the blurred and fine grids
        let grid_sum = grid_blur + grid_fine;

        let grid_sum = box_blur(&grid_sum, blur_radius);
        let grid_sum = box_blur(&grid_sum, blur_radius);
        
        // Step 4: Save the result if step saving is enabled
        if cli.save_steps {
            let step_dir = cli.step_dir.as_ref().map_or("".to_string(), |p| {
                format!("{}/", p.to_string_lossy())
            });
            let output_path = format!("{}dla_grid_step{}.txt", step_dir, step);
            if let Err(e) = export_grid_to_file(&grid_sum, &output_path) {
                eprintln!("Error exporting grid for step {}: {}", step, e);
            } else {
                println!("Exported mountain grid for step {} to {}", step, output_path);
            }
        }
        
        // Use this grid as the basis for the next iteration
        // Normalize the grid so maximum value is 1.0
        let mut next_grid = grid_sum.clone();
        next_grid.normalize(1.0);
        
        // Clamp negative values to zero
        for y in 0..next_grid.height() {
            for x in 0..next_grid.width() {
                let value = next_grid.get(x, y);
                if value < Some(0.0) {
                    next_grid.set(x, y, 0.0);
                }
            }
        }
        
        current_grid = next_grid;
    }
    
    // Export the final grid as a GLB file
    let final_grid = &current_grid;
    
    println!("Exporting 3D mesh to GLB file: {}", cli.output);
    
    // Calculate scale factors based on desired dimensions
    // The grid is normalized to 1.0, so we multiply by the desired dimension
    let grid_width = final_grid.width() as f32;
    let grid_height = final_grid.height() as f32;
    
    // Calculate scales to achieve the desired dimensions
    let scale_x = cli.dim_x / grid_width;
    let scale_y = cli.dim_y / grid_height;
    let scale_z = cli.dim_z; // Z is already normalized to 1.0 max
    
    println!("Using dimensions: {}x{}x{} world units", cli.dim_x, cli.dim_y, cli.dim_z);

    // Validate normal map parameters
    if cli.enable_normal_maps && !cli.enable_textures {
        eprintln!("Warning: --enable-normal-maps requires --enable-textures");
        eprintln!("Normal maps will be ignored.");
    }

    // Validate PBR parameters
    if cli.enable_pbr && !cli.enable_textures {
        eprintln!("Warning: --enable-pbr requires --enable-textures");
        eprintln!("PBR materials will be ignored.");
    }

    // Parse and validate biome
    use textures::BiomeType;
    let biome_type = BiomeType::from_str(&cli.biome);
    if biome_type.is_none() {
        eprintln!("Warning: Invalid biome '{}', using 'alpine' instead", cli.biome);
        eprintln!("Valid biomes: alpine, desert, volcanic, arctic");
    }
    let biome = biome_type.unwrap_or(BiomeType::Alpine);

    if cli.enable_textures {
        println!("Using {} biome textures", biome.name());
        if cli.enable_pbr {
            println!("PBR materials enabled (with roughness maps)");
        }
    }

    if let Err(e) = export_array_to_glb(
        final_grid,
        scale_x,
        scale_y,
        scale_z,
        &cli.output,
        cli.world_uv_scale,
        cli.enable_textures,
        cli.texture_resolution,
        cli.uv_test,
        cli.enable_normal_maps,
        cli.normal_strength,
        cli.normal_scale,
        biome,
        cli.enable_pbr,
    ) {
        eprintln!("Error exporting to GLB: {}", e);
    } else {
        println!("Successfully exported 3D mesh to: {}", cli.output);
    }
    
    println!("\nDone! Use a 3D viewer to open the GLB file and see the 3D terrain model.");
    
    Ok(())
}

/// Export a DLA simulation to a CSV file
#[allow(dead_code)]
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
