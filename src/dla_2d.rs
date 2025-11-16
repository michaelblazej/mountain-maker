use anyhow::Result;
use glam::Vec2;
use rand::prelude::*;
use rand::{RngCore, SeedableRng};
use rand::rngs::{StdRng, ThreadRng};

/// RNG type for DLA simulation that can be either random or seeded
#[derive(Debug, Clone)]
enum SimulationRng {
    ThreadRandom(ThreadRng),
    Seeded(StdRng),
}

impl RngCore for SimulationRng {
    fn next_u32(&mut self) -> u32 {
        match self {
            SimulationRng::ThreadRandom(rng) => rng.next_u32(),
            SimulationRng::Seeded(rng) => rng.next_u32(),
        }
    }

    fn next_u64(&mut self) -> u64 {
        match self {
            SimulationRng::ThreadRandom(rng) => rng.next_u64(),
            SimulationRng::Seeded(rng) => rng.next_u64(),
        }
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        match self {
            SimulationRng::ThreadRandom(rng) => rng.fill_bytes(dest),
            SimulationRng::Seeded(rng) => rng.fill_bytes(dest),
        }
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        match self {
            SimulationRng::ThreadRandom(rng) => rng.try_fill_bytes(dest),
            SimulationRng::Seeded(rng) => rng.try_fill_bytes(dest),
        }
    }
}
use std::collections::HashMap;
use std::ops::Add;

/// A simple 2D array representation
#[derive(Debug, Clone)]
pub struct Array2D {
    /// Width of the 2D array
    width: usize,
    /// Height of the 2D array
    height: usize,
    /// Data storage in row-major order (floating point)
    data: Vec<f32>,
}

impl Array2D {
    /// Create a new 2D array with the specified dimensions, filled with the given value
    pub fn new(width: usize, height: usize, initial_value: f32) -> Self {
        let data = vec![initial_value; width * height];
        Array2D {
            width,
            height,
            data,
        }
    }
    
    /// Get the value at the specified coordinates
    pub fn get(&self, x: usize, y: usize) -> Option<f32> {
        if x >= self.width || y >= self.height {
            return None;
        }
        
        let index = y * self.width + x;
        self.data.get(index).copied()
    }
    
    /// Set the value at the specified coordinates
    pub fn set(&mut self, x: usize, y: usize, value: f32) -> bool {
        if x >= self.width || y >= self.height {
            return false;
        }
        
        let index = y * self.width + x;
        if let Some(cell) = self.data.get_mut(index) {
            *cell = value;
            return true;
        }
        
        false
    }
    
    /// Get the width of the 2D array
    pub fn width(&self) -> usize {
        self.width
    }
    
    /// Get the height of the 2D array
    pub fn height(&self) -> usize {
        self.height
    }
    
    /// Get a reference to the underlying data
    #[allow(dead_code)]
    pub fn data(&self) -> &[f32] {
        &self.data
    }
    
    /// Normalize the array so the maximum value equals the specified target value
    /// 
    /// This scales all values proportionally so the maximum value in the array equals target_max
    /// If the maximum value is 0, no normalization is done to avoid division by zero
    pub fn normalize(&mut self, target_max: f32) -> &mut Self {
        // Find the current maximum value in the array
        if let Some(max_value) = self.data.iter().cloned().fold(None, |max, x| {
            match max {
                None => Some(x),
                Some(max) => Some(f32::max(max, x))
            }
        }) {
            // Only normalize if the maximum is greater than zero (to avoid division by zero)
            if max_value > 0.0 {
                // Apply scaling to all elements
                let scale_factor = target_max / max_value;
                for value in &mut self.data {
                    *value *= scale_factor;
                }
            }
        }
        
        // Return self for method chaining
        self
    }
}

// Implement the Add trait for Array2D, so we can use the + operator
impl Add for Array2D {
    type Output = Array2D;
    
    /// Add two arrays element-wise
    /// 
    /// # Panics
    /// 
    /// Panics if the arrays have different dimensions
    fn add(self, other: Array2D) -> Self::Output {
        // Check if the dimensions match
        assert_eq!(self.width, other.width, "Arrays must have the same width");
        assert_eq!(self.height, other.height, "Arrays must have the same height");
        
        // Create a new array with the same dimensions
        let mut result = Array2D::new(self.width, self.height, 0.0);
        
        // Add the values element-wise
        for i in 0..self.data.len() {
            result.data[i] = self.data[i] + other.data[i];
        }
        
        result
    }
}

/// A 2D point in the simulation grid
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GridPoint {
    pub x: i32,
    pub y: i32,
}

impl GridPoint {
    pub fn new(x: i32, y: i32) -> Self {
        GridPoint { x, y }
    }
    
    #[allow(dead_code)]
    pub fn to_vec2(&self) -> Vec2 {
        Vec2::new(self.x as f32, self.y as f32)
    }
    
    pub fn from_vec2(v: Vec2) -> Self {
        GridPoint::new(v.x.round() as i32, v.y.round() as i32)
    }
    
    #[allow(dead_code)]
    pub fn distance_squared(&self, other: &GridPoint) -> i32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }
}

/// Parameters controlling the DLA simulation behavior
#[derive(Debug, Clone)]
pub struct DlaParameters {
    /// Number of particles to simulate
    pub num_particles: usize,
    /// Stickiness factor (0.0-1.0) - higher means particles stick more easily
    pub stickiness: f32,
    /// Random walk step size
    pub step_size: f32,
    /// Radius multiplier for particle spawning
    pub spawn_radius_factor: f32,
    /// Maximum number of steps a particle can take before being discarded
    pub max_steps_per_particle: usize,
}

impl Default for DlaParameters {
    fn default() -> Self {
        DlaParameters {
            num_particles: 10000,
            stickiness: 0.6,
            step_size: 1.0,
            spawn_radius_factor: 1.5,
            max_steps_per_particle: 5000,
        }
    }
}

/// Particle data in the DLA simulation
#[derive(Debug, Clone, Copy)]
pub struct ParticleData {
    /// The index of when this particle was added
    pub index: i32,
    /// Reference to the particle it got stuck to (-1 for the initial seed)
    pub stuck_to: i32,
}

/// Represents the state of a 2D Diffusion Limited Aggregation simulation
#[derive(Debug, Clone)]
pub struct DlaSimulation {
    /// Settled particles forming the structure with their data
    pub particles: HashMap<GridPoint, ParticleData>,
    /// Bounds of the simulation space (min_x, min_y, max_x, max_y)
    pub bounds: (i32, i32, i32, i32),
    /// Parameters controlling the simulation
    pub params: DlaParameters,
    /// Random number generator for the simulation
    rng: SimulationRng,
    /// Current radius of the structure (for spawning new particles)
    radius: f32,
    /// Center of the simulation
    center: GridPoint,
    /// Maximum distance from center to any settled particle
    max_distance: f32,
}

impl DlaSimulation {
    /// Create a new DLA simulation with default parameters
    #[allow(dead_code)]
    pub fn new(width: i32, height: i32) -> Self {
        Self::with_seed(width, height, None)
    }

    /// Create a new DLA simulation with a specific random seed
    pub fn with_seed(width: i32, height: i32, seed: Option<u64>) -> Self {
        let center_x = width / 2;
        let center_y = height / 2;
        let center = GridPoint { x: center_x, y: center_y };

        // Create the first "seed" particle at the center
        let mut particles = HashMap::new();
        particles.insert(center.clone(), ParticleData { 
            index: 0, 
            stuck_to: -1,
        });
        
        // Initialize RNG based on seed
        let rng = match seed {
            Some(seed_value) => SimulationRng::Seeded(StdRng::seed_from_u64(seed_value)),
            None => SimulationRng::ThreadRandom(thread_rng())
        };

        Self {
            particles,
            bounds: (0, 0, width, height),
            params: DlaParameters::default(),
            rng,
            radius: 1.0,
            center,
            max_distance: 0.0,
        }
    }

    
    /// Create a new DLA simulation with custom parameters
    #[allow(dead_code)]
    pub fn with_params(width: i32, height: i32, params: DlaParameters) -> Self {
        Self::with_params_and_seed(width, height, params, None)
    }
    
    /// Create a new DLA simulation with custom parameters and a specific random seed
    pub fn with_params_and_seed(width: i32, height: i32, params: DlaParameters, seed: Option<u64>) -> Self {
        let mut simulation = Self::with_seed(width, height, seed);
        simulation.params = params;
        simulation
    }

    /// Create a new DLA simulation based on an existing one but with increased dimensions
    #[allow(dead_code)]
    pub fn with_increased_dimensions(source: &DlaSimulation, scale_factor: f32) -> Self {
        let (old_min_x, old_min_y, old_max_x, old_max_y) = source.bounds;
        let old_width = old_max_x - old_min_x;
        let old_height = old_max_y - old_min_y;
        
        let new_width = (old_width as f32 * scale_factor) as i32;
        let new_height = (old_height as f32 * scale_factor) as i32;
        
        // Create new simulation with scaled dimensions
        let mut new_simulation = DlaSimulation::new(new_width, new_height);
        new_simulation.params = source.params.clone();
        
        // Scale and copy particles from source simulation
        for (p, particle_data) in &source.particles {
            // Calculate relative position from old center
            let rel_x = p.x - source.center.x;
            let rel_y = p.y - source.center.y;
            
            // Scale the position
            let scaled_x = (rel_x as f32 * scale_factor).round() as i32;
            let scaled_y = (rel_y as f32 * scale_factor).round() as i32;
            
            // Add to new center
            let new_x = new_simulation.center.x + scaled_x;
            let new_y = new_simulation.center.y + scaled_y;
            
            // Add particle to new simulation with the same data
            let new_particle = GridPoint::new(new_x, new_y);
            new_simulation.particles.insert(new_particle, *particle_data);
            
            // Update max distance
            let dist = ((scaled_x * scaled_x + scaled_y * scaled_y) as f32).sqrt();
            if dist > new_simulation.max_distance {
                new_simulation.max_distance = dist;
            }
        }
        
        // Update radius for spawning
        new_simulation.radius = new_simulation.max_distance + 5.0;
        
        new_simulation
    }
    
    /// Run the DLA simulation to completion
    pub fn run(&mut self) -> Result<()> {
        println!("Starting 2D DLA simulation with {} particles...", self.params.num_particles);
        
        for i in 0..self.params.num_particles {
            if i % 1000 == 0 && i > 0 {
                println!("Processed {} particles", i);
            }
            
            self.add_particle(i as i32 + 1)?; // +1 because we already used 0 for the seed
            
            // Update spawn radius based on current structure size
            self.update_radius();
        }
        
        println!("DLA simulation complete with {} settled particles", self.particles.len());
        Ok(())
    }
    
    /// Add a single particle to the simulation
    pub fn add_particle(&mut self, index: i32) -> Result<()> {
        // Generate random position on the perimeter
        let particle_pos = self.generate_random_start_position();
        
        // Perform random walk until particle settles or leaves bounds
        self.random_walk(particle_pos, index)
    }
    
    /// Generate a random starting position around the current structure
    fn generate_random_start_position(&mut self) -> Vec2 {
        // Random position on a circle around the current structure
        let angle = self.rng.r#gen::<f32>() * 2.0 * std::f32::consts::PI;
        
        // Use current radius multiplied by spawn factor
        let spawn_radius = self.radius * self.params.spawn_radius_factor;
        
        let center_vec = Vec2::new(self.center.x as f32, self.center.y as f32);
        
        let x = center_vec.x + spawn_radius * angle.cos();
        let y = center_vec.y + spawn_radius * angle.sin();
        
        Vec2::new(x, y)
    }
    
    /// Perform random walk for a particle until it settles or leaves bounds
    fn random_walk(&mut self, mut position: Vec2, index: i32) -> Result<()> {        
        for _ in 0..self.params.max_steps_per_particle {
            // Check if particle is out of bounds
            if !self.is_in_bounds(position) {
                return Ok(()); // Particle escaped
            }
            
            // Check if particle should settle
            if self.should_settle(position) {
                // Find which particle this one is sticking to
                let stuck_to_particle = self.find_nearby_particle(position);
                
                // Convert to grid position and add to settled particles with its data
                let grid_pos = GridPoint::from_vec2(position);
                self.particles.insert(grid_pos, ParticleData { 
                    index, 
                    stuck_to: stuck_to_particle 
                });
                
                // Update max distance if needed
                let distance = ((grid_pos.x - self.center.x).pow(2) + 
                                (grid_pos.y - self.center.y).pow(2)) as f32;
                let distance = distance.sqrt();
                
                if distance > self.max_distance {
                    self.max_distance = distance;
                }
                
                return Ok(());
            }
            
            // Random walk step
            position = self.take_random_step(position);
        }
        
        Ok(()) // Particle didn't settle within max steps
    }
    
    /// Take a random step with the configured step size
    fn take_random_step(&mut self, position: Vec2) -> Vec2 {
        let angle = self.rng.r#gen::<f32>() * 2.0 * std::f32::consts::PI;
        let step_x = angle.cos() * self.params.step_size;
        let step_y = angle.sin() * self.params.step_size;
        
        Vec2::new(position.x + step_x, position.y + step_y)
    }
    
    /// Check if a position is within the simulation bounds
    fn is_in_bounds(&self, position: Vec2) -> bool {
        let (min_x, min_y, max_x, max_y) = self.bounds;
        
        position.x >= min_x as f32 && position.x < max_x as f32 &&
        position.y >= min_y as f32 && position.y < max_y as f32
    }
    
    /// Check if a particle should settle at the given position
    fn should_settle(&mut self, position: Vec2) -> bool {
        // Find if there's a nearby particle to stick to
        self.find_nearby_particle(position) >= 0
    }
    
    /// Find the index of a nearby particle this one might stick to
    /// Returns the index of that particle, or -1 if none found
    fn find_nearby_particle(&mut self, position: Vec2) -> i32 {
        // Convert to grid position
        let grid_pos = GridPoint::from_vec2(position);
        
        // Check nearby grid points for settled particles
        for dx in -1..=1 {
            for dy in -1..=1 {
                if dx == 0 && dy == 0 {
                    continue; // Skip the current position
                }
                
                let check_pos = GridPoint::new(grid_pos.x + dx, grid_pos.y + dy);
                
                if let Some(particle_data) = self.particles.get(&check_pos) {
                    // Apply stickiness factor
                    if self.rng.r#gen::<f32>() < self.params.stickiness {
                        return particle_data.index;
                    }
                }
            }
        }
        
        -1 // No particle found to stick to
    }
    
    /// Update the radius based on the current structure size
    fn update_radius(&mut self) {
        self.radius = self.max_distance + 5.0;
    }
    
    /// Get the dimensions of the structure (width, height)
    #[allow(dead_code)]
    pub fn get_dimensions(&self) -> (i32, i32) {
        let mut min_x = i32::MAX;
        let mut min_y = i32::MAX;
        let mut max_x = i32::MIN;
        let mut max_y = i32::MIN;
        
        for (p, _index) in &self.particles {
            min_x = min_x.min(p.x);
            min_y = min_y.min(p.y);
            max_x = max_x.max(p.x);
            max_y = max_y.max(p.y);
        }
        
        (max_x - min_x + 1, max_y - min_y + 1)
    }
    
    /// Get the bounds of the particles (min_x, min_y, max_x, max_y)
    pub fn get_particle_bounds(&self) -> (i32, i32, i32, i32) {
        let mut min_x = i32::MAX;
        let mut min_y = i32::MAX;
        let mut max_x = i32::MIN;
        let mut max_y = i32::MIN;
        
        for (p, _) in &self.particles {
            min_x = min_x.min(p.x);
            min_y = min_y.min(p.y);
            max_x = max_x.max(p.x);
            max_y = max_y.max(p.y);
        }
        
        (min_x, min_y, max_x, max_y)
    }
    
    /// Convert the DLA simulation to a 2D grid (Array2D) where cells with particles are 1 and others are 0
    /// 
    /// # Arguments
    /// * `resolution` - The resolution factor. Higher values create a finer grid.
    ///                  A resolution of 1 creates a 1-to-1 mapping from particles to grid cells.
    ///                  A resolution of 2 creates a grid with 2x2 cells per particle spacing, etc.
    ///                  Connected particles will have their gaps filled in at higher resolutions.
    pub fn to_grid(&self, resolution: usize) -> Array2D {
        // Get the bounds of the particles
        let (min_x, min_y, max_x, max_y) = self.get_particle_bounds();
        
        // Calculate grid dimensions with the resolution factor
        let width = ((max_x - min_x + 1) * resolution as i32) as usize;
        let height = ((max_y - min_y + 1) * resolution as i32) as usize;
        
        // Create a new grid filled with zeros
        let mut grid = Array2D::new(width, height, 0.0);
        
        // Calculate the center of the particle structure
        let center_x = (min_x + max_x) as f32 / 2.0;
        let center_y = (min_y + max_y) as f32 / 2.0;
        
        // Calculate the maximum distance from center for normalization
        let mut max_distance = 0.0f32;
        for (p, _) in &self.particles {
            let dx = p.x as f32 - center_x;
            let dy = p.y as f32 - center_y;
            let distance = (dx * dx + dy * dy).sqrt();
            max_distance = max_distance.max(distance);
        }
        max_distance = max_distance* 0.4;
        
        // First pass: Mark all particle positions with distance-based values
        let mut particle_positions = Vec::new();
        for (p, data) in &self.particles {
            // Map particle coordinates to high-resolution grid indices
            let grid_x = ((p.x - min_x) * resolution as i32) as usize;
            let grid_y = ((p.y - min_y) * resolution as i32) as usize;
            
            // Calculate distance from center and normalize it
            let dx = p.x as f32 - center_x;
            let dy = p.y as f32 - center_y;
            let distance = (dx * dx + dy * dy).sqrt();
            let normalized_distance = if max_distance > 0.0 { distance / max_distance } else { 0.0 };
            
            // Set grid value based on distance: closer to center = higher value, further = closer to 0
            let value = 1.0 - normalized_distance;
            grid.set(grid_x, grid_y, value);
            particle_positions.push((grid_x, grid_y, data.index));
        }
        
        // Second pass: Connect particles in multiple ways to create a more connected appearance
        if resolution > 1 {
            // Create a mapping from particle index to grid position
            let mut index_to_position = HashMap::new();
            for (x, y, index) in &particle_positions {
                index_to_position.insert(*index, (*x, *y));
            }
            
            // Store all particle positions for nearest neighbor processing
            let mut all_positions = Vec::new();
            for (p, _) in &self.particles {
                let x = ((p.x - min_x) * resolution as i32) as usize;
                let y = ((p.y - min_y) * resolution as i32) as usize;
                all_positions.push((x, y));
            }
            
            // 1. Connect each particle with the one it stuck to
            for (p, data) in &self.particles {
                // Skip the seed particle
                if data.stuck_to < 0 {
                    continue;
                }
                
                // Get positions of this particle and the one it stuck to
                let child_x = ((p.x - min_x) * resolution as i32) as usize;
                let child_y = ((p.y - min_y) * resolution as i32) as usize;
                
                if let Some(&(parent_x, parent_y)) = index_to_position.get(&data.stuck_to) {
                    // Draw a thicker line between the two points for better connectivity
                    self.draw_thick_line(&mut grid, child_x, child_y, parent_x, parent_y, resolution / 2);
                }
            }
            
            // 2. Connect nearby particles when they are within a certain distance
            let connection_threshold = resolution * 3; // Adjust this threshold to control aggressiveness
            
            for i in 0..all_positions.len() {
                let (x1, y1) = all_positions[i];
                
                // Check against a subset of other positions to avoid excessive processing
                // This is a simplified approach - in a real application, you might use a spatial partitioning
                // structure like a quadtree for better performance with large numbers of particles
                for j in (i + 1)..all_positions.len() {
                    let (x2, y2) = all_positions[j];
                    
                    // Calculate Manhattan distance (faster than Euclidean)
                    let dist = ((x1 as isize - x2 as isize).abs() + (y1 as isize - y2 as isize).abs()) as usize;
                    
                    // If particles are close but not directly adjacent, connect them
                    // Skip if they share the same x or y coordinate (avoiding straight horizontal/vertical lines)
                    if dist > 1 && dist <= connection_threshold && x1 != x2 && y1 != y2 {
                        self.draw_line(&mut grid, x1, y1, x2, y2);
                    }
                }
            } 
            
            // 3. Fill in small gaps
            self.fill_gaps(&mut grid);
        }
        
        grid
    }
    
    /// Convert the DLA simulation to a 2D grid with default resolution (1:1 mapping)
    #[allow(dead_code)]
    pub fn to_grid_default(&self) -> Array2D {
        self.to_grid(1)
    }
    
    /// Draw a line between two points in the grid using Bresenham's line algorithm
    fn draw_line(&self, grid: &mut Array2D, x0: usize, y0: usize, x1: usize, y1: usize) {
        // Convert to isize for calculations
        let x0 = x0 as isize;
        let y0 = y0 as isize;
        let x1 = x1 as isize;
        let y1 = y1 as isize;
        
        let dx = (x1 - x0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let dy = -((y1 - y0).abs()); // Negative because y increases downward in our grid
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        
        let mut x = x0;
        let mut y = y0;
        
        loop {
            // Set the current pixel if it's within bounds
            if x >= 0 && y >= 0 && x < grid.width() as isize && y < grid.height() as isize {
                grid.set(x as usize, y as usize, 1.0);
            }
            
            // Check if we've reached the end point
            if x == x1 && y == y1 {
                break;
            }
            
            let e2 = 2 * err;
            
            // Handle horizontal movement
            if e2 >= dy {
                if x == x1 {
                    break;
                }
                err += dy;
                x += sx;
            }
            
            // Handle vertical movement
            if e2 <= dx {
                if y == y1 {
                    break;
                }
                err += dx;
                y += sy;
            }
        }
    }
    
    /// Draw a thicker line between two points by drawing multiple parallel lines
    fn draw_thick_line(&self, grid: &mut Array2D, x0: usize, y0: usize, x1: usize, y1: usize, thickness: usize) {
        // First draw the main line
        self.draw_line(grid, x0, y0, x1, y1);
        
        // Skip additional processing for thickness of 1
        if thickness <= 1 {
            return;
        }
        
        // Convert to isize for calculations
        let x0 = x0 as isize;
        let y0 = y0 as isize;
        let x1 = x1 as isize;
        let y1 = y1 as isize;
        
        // Calculate the direction vector of the line
        let dx = x1 - x0;
        let dy = y1 - y0;
        
        // Calculate the length of the line
        let length = ((dx * dx + dy * dy) as f64).sqrt();
        
        // Skip if length is 0 or too small
        if length < 0.1 {
            return;
        }
        
        // Calculate the normalized perpendicular vector
        let px = -dy as f64 / length;
        let py = dx as f64 / length;
        
        // Draw additional lines parallel to the main line
        let radius = thickness as isize / 2;
        for offset in 1..=radius {
            // Offset in the positive perpendicular direction
            let ox1 = (x0 as f64 + px * offset as f64).round() as isize;
            let oy1 = (y0 as f64 + py * offset as f64).round() as isize;
            let ox2 = (x1 as f64 + px * offset as f64).round() as isize;
            let oy2 = (y1 as f64 + py * offset as f64).round() as isize;
            
            // Draw the positive offset line if coordinates are valid
            if ox1 >= 0 && oy1 >= 0 && ox2 >= 0 && oy2 >= 0 && 
               ox1 < grid.width() as isize && oy1 < grid.height() as isize && 
               ox2 < grid.width() as isize && oy2 < grid.height() as isize {
                self.draw_line(grid, ox1 as usize, oy1 as usize, ox2 as usize, oy2 as usize);
            }
            
            // Offset in the negative perpendicular direction
            let ox1 = (x0 as f64 - px * offset as f64).round() as isize;
            let oy1 = (y0 as f64 - py * offset as f64).round() as isize;
            let ox2 = (x1 as f64 - px * offset as f64).round() as isize;
            let oy2 = (y1 as f64 - py * offset as f64).round() as isize;
            
            // Draw the negative offset line if coordinates are valid
            if ox1 >= 0 && oy1 >= 0 && ox2 >= 0 && oy2 >= 0 && 
               ox1 < grid.width() as isize && oy1 < grid.height() as isize && 
               ox2 < grid.width() as isize && oy2 < grid.height() as isize {
                self.draw_line(grid, ox1 as usize, oy1 as usize, ox2 as usize, oy2 as usize);
            }
        }
    }
    
    /// Fill small gaps in the grid to create a more connected appearance
    fn fill_gaps(&self, grid: &mut Array2D) {
        let width = grid.width();
        let height = grid.height();
        
        // Create a copy of the grid to check against while we modify the original
        let original = grid.clone();
        
        // Iterate through all cells in the grid
        for y in 1..height-1 {
            for x in 1..width-1 {
                // Skip cells that are already filled
                if original.get(x, y) == Some(1.0) {
                    continue;
                }
                
                // Count the number of filled neighbors
                let mut filled_count = 0;
                
                // Check 8 surrounding neighbors
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        if dx == 0 && dy == 0 {
                            continue; // Skip the center cell
                        }
                        
                        let nx = (x as isize + dx) as usize;
                        let ny = (y as isize + dy) as usize;
                        
                        if nx < width && ny < height && original.get(nx, ny) == Some(1.0) {
                            filled_count += 1;
                        }
                    }
                }
                
                // Fill this cell if it has enough filled neighbors
                // This creates a more connected appearance by filling in small gaps
                if filled_count >= 5 { // Threshold can be adjusted for different connectivity levels
                    grid.set(x, y, 1.0);
                }
            }
        }
    }
}
