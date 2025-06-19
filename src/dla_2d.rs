use anyhow::Result;
use glam::Vec2;
use rand::prelude::*;
use std::collections::HashSet;

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
    
    pub fn to_vec2(&self) -> Vec2 {
        Vec2::new(self.x as f32, self.y as f32)
    }
    
    pub fn from_vec2(v: Vec2) -> Self {
        GridPoint::new(v.x.round() as i32, v.y.round() as i32)
    }
    
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

/// Represents the state of a 2D Diffusion Limited Aggregation simulation
#[derive(Debug, Clone)]
pub struct DlaSimulation {
    /// Settled particles forming the structure
    pub particles: HashSet<GridPoint>,
    /// Bounds of the simulation space (min_x, min_y, max_x, max_y)
    pub bounds: (i32, i32, i32, i32),
    /// Parameters controlling the simulation
    pub params: DlaParameters,
    /// Random number generator for the simulation
    rng: ThreadRng,
    /// Current radius of the structure (for spawning new particles)
    radius: f32,
    /// Center of the simulation
    center: GridPoint,
    /// Maximum distance from center to any settled particle
    max_distance: f32,
}

impl DlaSimulation {
    /// Create a new DLA simulation with the given dimensions
    pub fn new(width: i32, height: i32) -> Self {
        let center_x = width / 2;
        let center_y = height / 2;
        let center = GridPoint::new(center_x, center_y);
        
        let mut simulation = DlaSimulation {
            particles: HashSet::new(),
            bounds: (0, 0, width, height),
            params: DlaParameters::default(),
            rng: thread_rng(),
            radius: 1.0,
            center,
            max_distance: 0.0,
        };
        
        // Add a single seed particle at the center
        simulation.particles.insert(center);
        
        simulation
    }
    
    /// Create a new DLA simulation with custom parameters
    pub fn with_params(width: i32, height: i32, params: DlaParameters) -> Self {
        let mut simulation = Self::new(width, height);
        simulation.params = params;
        simulation
    }

    /// Create a new DLA simulation based on an existing one but with increased dimensions
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
        for p in &source.particles {
            // Calculate relative position from old center
            let rel_x = p.x - source.center.x;
            let rel_y = p.y - source.center.y;
            
            // Scale the position
            let scaled_x = (rel_x as f32 * scale_factor).round() as i32;
            let scaled_y = (rel_y as f32 * scale_factor).round() as i32;
            
            // Add to new center
            let new_x = new_simulation.center.x + scaled_x;
            let new_y = new_simulation.center.y + scaled_y;
            
            // Add particle to new simulation
            let new_particle = GridPoint::new(new_x, new_y);
            new_simulation.particles.insert(new_particle);
            
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
            
            self.add_particle()?;
            
            // Update spawn radius based on current structure size
            self.update_radius();
        }
        
        println!("DLA simulation complete with {} settled particles", self.particles.len());
        Ok(())
    }
    
    /// Add a single particle to the simulation
    pub fn add_particle(&mut self) -> Result<()> {
        // Generate random position on the perimeter
        let particle_pos = self.generate_random_start_position();
        
        // Perform random walk until particle settles or leaves bounds
        self.random_walk(particle_pos)
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
    fn random_walk(&mut self, mut position: Vec2) -> Result<()> {        
        for _ in 0..self.params.max_steps_per_particle {
            // Check if particle is out of bounds
            if !self.is_in_bounds(position) {
                return Ok(()); // Particle escaped
            }
            
            // Check if particle should settle
            if self.should_settle(position) {
                // Convert to grid position and add to settled particles
                let grid_pos = GridPoint::from_vec2(position);
                self.particles.insert(grid_pos);
                
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
        // Convert to grid position
        let grid_pos = GridPoint::from_vec2(position);
        
        // Check nearby grid points for settled particles
        for dx in -1..=1 {
            for dy in -1..=1 {
                if dx == 0 && dy == 0 {
                    continue; // Skip the current position
                }
                
                let check_pos = GridPoint::new(grid_pos.x + dx, grid_pos.y + dy);
                
                if self.particles.contains(&check_pos) {
                    // Apply stickiness factor
                    if self.rng.r#gen::<f32>() < self.params.stickiness {
                        return true;
                    }
                }
            }
        }
        
        false
    }
    
    /// Update the radius based on the current structure size
    fn update_radius(&mut self) {
        self.radius = self.max_distance + 5.0;
    }
    
    /// Get the dimensions of the structure (width, height)
    pub fn get_dimensions(&self) -> (i32, i32) {
        let mut min_x = i32::MAX;
        let mut min_y = i32::MAX;
        let mut max_x = i32::MIN;
        let mut max_y = i32::MIN;
        
        for p in &self.particles {
            min_x = min_x.min(p.x);
            min_y = min_y.min(p.y);
            max_x = max_x.max(p.x);
            max_y = max_y.max(p.y);
        }
        
        (max_x - min_x + 1, max_y - min_y + 1)
    }
}
