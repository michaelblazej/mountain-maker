/// Procedural texture generation module for Mountain Maker
///
/// This module provides functions to generate tileable, realistic textures
/// and tangent-space normal maps for terrain rendering using multi-octave noise functions.

use image::{RgbaImage, Rgba};
use noise::{NoiseFn, Perlin};
use crate::dla_2d::Array2D;

// ============================================================================
// Normal Map Helper Functions
// ============================================================================

/// Encode a tangent-space normal vector to RGB color
///
/// Converts normal components from [-1, 1] to [0, 255] using glTF encoding.
/// Flat surface (0, 0, 1) encodes to (128, 128, 255).
///
/// # Arguments
/// * `nx` - X component of normal vector (tangent direction)
/// * `ny` - Y component of normal vector (bitangent direction)
/// * `nz` - Z component of normal vector (surface normal)
///
/// # Returns
/// * `[u8; 4]` - RGBA bytes with RGB encoding normal, A=255
fn encode_tangent_normal(nx: f32, ny: f32, nz: f32) -> [u8; 4] {
    // Normalize the input vector
    let length = (nx * nx + ny * ny + nz * nz).sqrt();
    let (nx, ny, nz) = if length > 0.0 {
        (nx / length, ny / length, nz / length)
    } else {
        (0.0, 0.0, 1.0) // Default to flat normal
    };

    // Encode: map [-1, 1] to [0, 255] with proper rounding
    let r = (((nx + 1.0) / 2.0 * 255.0).round()).clamp(0.0, 255.0) as u8;
    let g = (((ny + 1.0) / 2.0 * 255.0).round()).clamp(0.0, 255.0) as u8;
    let b = (((nz + 1.0) / 2.0 * 255.0).round()).clamp(0.0, 255.0) as u8;

    [r, g, b, 255]
}

/// Sample multi-octave Perlin noise for height
///
/// Combines multiple octaves of noise using fractional Brownian motion (fBM)
/// with standard persistence (0.5) and lacunarity (2.0).
///
/// # Arguments
/// * `perlin` - Perlin noise generator
/// * `x` - X coordinate in noise space
/// * `y` - Y coordinate in noise space
/// * `z` - Z offset for 3D noise sampling
/// * `octaves` - Number of noise octaves to combine
/// * `frequency` - Initial frequency multiplier
/// * `amplitude` - Initial amplitude multiplier
///
/// # Returns
/// * `f64` - Combined noise value
fn sample_height_noise(
    perlin: &Perlin,
    x: f64,
    y: f64,
    z: f64,
    octaves: u32,
    mut frequency: f64,
    mut amplitude: f64,
) -> f64 {
    let mut total = 0.0;
    for _ in 0..octaves {
        total += amplitude * perlin.get([x * frequency, y * frequency, z]);
        amplitude *= 0.5; // Persistence
        frequency *= 2.0; // Lacunarity
    }
    total
}

/// Calculate height gradient using central difference method for torus topology
///
/// Uses central difference to compute partial derivatives of noise function,
/// ensuring seamless tiling through torus coordinate mapping.
///
/// # Arguments
/// * `perlin` - Perlin noise generator
/// * `nx` - X coordinate in torus space
/// * `ny` - Y coordinate in torus space
/// * `size` - Texture size (used to calculate epsilon)
/// * `octaves` - Number of noise octaves
/// * `z_offset` - Z offset for 3D noise
/// * `scale` - Frequency scale for noise
/// * `epsilon` - Step size for gradient calculation
/// * `strength` - Multiplier for gradient magnitude
///
/// # Returns
/// * `(f32, f32)` - Partial derivatives (du, dv)
fn torus_gradient(
    perlin: &Perlin,
    nx: f64,
    ny: f64,
    _size: u32,
    octaves: u32,
    z_offset: f64,
    scale: f64,
    epsilon: f64,
    strength: f32,
) -> (f32, f32) {
    // Sample height at neighboring points using torus topology
    let h_right = sample_height_noise(perlin, nx + epsilon, ny, z_offset, octaves, scale, 1.0);
    let h_left = sample_height_noise(perlin, nx - epsilon, ny, z_offset, octaves, scale, 1.0);
    let h_up = sample_height_noise(perlin, nx, ny + epsilon, z_offset, octaves, scale, 1.0);
    let h_down = sample_height_noise(perlin, nx, ny - epsilon, z_offset, octaves, scale, 1.0);

    // Central difference for partial derivatives
    let du = ((h_right - h_left) / (2.0 * epsilon)) as f32 * strength;
    let dv = ((h_up - h_down) / (2.0 * epsilon)) as f32 * strength;

    (du, dv)
}

// ============================================================================
// Color Texture Generation Functions
// ============================================================================

/// Generate a rock texture with brown/grey tones and natural variation
///
/// Uses multi-octave Perlin noise to create realistic rocky patterns.
/// The texture is tileable (seamless wrapping) using torus topology.
///
/// # Arguments
/// * `size` - Resolution of the texture (typically 512, 1024, or 2048)
///
/// # Returns
/// * `RgbaImage` - RGBA image compatible with mesh-tools
#[allow(dead_code)]
pub fn generate_rock_texture(size: u32) -> RgbaImage {
    let mut img = RgbaImage::new(size, size);
    let perlin = Perlin::new(42); // Fixed seed for consistent textures

    // Color palette for rock
    let dark_grey = [60.0, 55.0, 50.0];
    let med_grey = [100.0, 95.0, 85.0];
    let brown = [120.0, 100.0, 70.0];
    let light_grey = [140.0, 135.0, 125.0];

    for y in 0..size {
        for x in 0..size {
            // Use torus topology for seamless tiling
            let (nx, ny) = torus_coords(x, y, size);

            // Multi-octave noise for realistic detail
            let mut noise_value = 0.0;
            let mut amplitude = 1.0;
            let mut frequency = 1.0;
            let octaves = 5;

            for _ in 0..octaves {
                noise_value += amplitude * perlin.get([nx * frequency, ny * frequency, 0.0]);
                amplitude *= 0.5;
                frequency *= 2.0;
            }

            // Normalize to 0-1 range
            noise_value = (noise_value + 1.0) / 2.0;

            // Add another noise layer for texture variation
            let detail_noise = perlin.get([nx * 8.0, ny * 8.0, 10.0]);
            let detail = ((detail_noise + 1.0) / 2.0) as f32;

            // Blend colors based on noise values (cast to f32)
            let noise_value_f32 = noise_value as f32;
            let color = if noise_value < 0.3 {
                // Dark grey for cracks and shadows
                blend_colors(&dark_grey, &med_grey, noise_value_f32 / 0.3)
            } else if noise_value < 0.6 {
                // Medium grey to brown
                blend_colors(&med_grey, &brown, (noise_value_f32 - 0.3) / 0.3)
            } else {
                // Brown to light grey for highlights
                blend_colors(&brown, &light_grey, (noise_value_f32 - 0.6) / 0.4)
            };

            // Apply detail variation
            let r = (color[0] * (0.9 + detail * 0.2)).clamp(0.0, 255.0) as u8;
            let g = (color[1] * (0.9 + detail * 0.2)).clamp(0.0, 255.0) as u8;
            let b = (color[2] * (0.9 + detail * 0.2)).clamp(0.0, 255.0) as u8;

            img.put_pixel(x, y, Rgba([r, g, b, 255]));
        }
    }

    img
}

/// Generate a grass texture with green tones and natural variation
///
/// Uses multi-octave Perlin noise to create realistic grass patterns.
/// The texture is tileable (seamless wrapping) using torus topology.
///
/// # Arguments
/// * `size` - Resolution of the texture (typically 512, 1024, or 2048)
///
/// # Returns
/// * `RgbaImage` - RGBA image compatible with mesh-tools
pub fn generate_grass_texture(size: u32) -> RgbaImage {
    let mut img = RgbaImage::new(size, size);
    let perlin = Perlin::new(123); // Different seed for grass

    // Color palette for grass
    let dark_green = [40.0, 70.0, 30.0];
    let med_green = [60.0, 110.0, 45.0];
    let light_green = [80.0, 140.0, 60.0];
    let yellow_green = [100.0, 150.0, 50.0];

    for y in 0..size {
        for x in 0..size {
            // Use torus topology for seamless tiling
            let (nx, ny) = torus_coords(x, y, size);

            // Multi-octave noise for realistic detail
            let mut noise_value = 0.0;
            let mut amplitude = 1.0;
            let mut frequency = 1.0;
            let octaves = 4;

            for _ in 0..octaves {
                noise_value += amplitude * perlin.get([nx * frequency, ny * frequency, 5.0]);
                amplitude *= 0.5;
                frequency *= 2.0;
            }

            // Normalize to 0-1 range
            noise_value = (noise_value + 1.0) / 2.0;

            // Add fine detail for grass blades
            let blade_noise = perlin.get([nx * 16.0, ny * 16.0, 15.0]);
            let blade_detail = ((blade_noise + 1.0) / 2.0) as f32;

            // Blend colors based on noise values (cast to f32)
            let noise_value_f32 = noise_value as f32;
            let color = if noise_value < 0.25 {
                // Dark green patches
                blend_colors(&dark_green, &med_green, noise_value_f32 / 0.25)
            } else if noise_value < 0.6 {
                // Medium green to light green
                blend_colors(&med_green, &light_green, (noise_value_f32 - 0.25) / 0.35)
            } else {
                // Light green to yellow-green highlights
                blend_colors(&light_green, &yellow_green, (noise_value_f32 - 0.6) / 0.4)
            };

            // Apply blade detail
            let r = (color[0] * (0.85 + blade_detail * 0.3)).clamp(0.0, 255.0) as u8;
            let g = (color[1] * (0.85 + blade_detail * 0.3)).clamp(0.0, 255.0) as u8;
            let b = (color[2] * (0.85 + blade_detail * 0.3)).clamp(0.0, 255.0) as u8;

            img.put_pixel(x, y, Rgba([r, g, b, 255]));
        }
    }

    img
}

/// Generate a snow texture with white/light blue tones and subtle variation
///
/// Uses multi-octave Perlin noise to create realistic snowy patterns.
/// The texture is tileable (seamless wrapping) using torus topology.
///
/// # Arguments
/// * `size` - Resolution of the texture (typically 512, 1024, or 2048)
///
/// # Returns
/// * `RgbaImage` - RGBA image compatible with mesh-tools
#[allow(dead_code)]
pub fn generate_snow_texture(size: u32) -> RgbaImage {
    let mut img = RgbaImage::new(size, size);
    let perlin = Perlin::new(789); // Different seed for snow

    // Color palette for snow
    let pure_white = [255.0, 255.0, 255.0];
    let light_blue = [235.0, 245.0, 255.0];
    let ice_blue = [220.0, 235.0, 250.0];
    let shadow_grey = [200.0, 210.0, 220.0];

    for y in 0..size {
        for x in 0..size {
            // Use torus topology for seamless tiling
            let (nx, ny) = torus_coords(x, y, size);

            // Multi-octave noise for subtle snow variation
            let mut noise_value = 0.0;
            let mut amplitude = 1.0;
            let mut frequency = 1.0;
            let octaves = 4;

            for _ in 0..octaves {
                noise_value += amplitude * perlin.get([nx * frequency, ny * frequency, 20.0]);
                amplitude *= 0.5;
                frequency *= 2.0;
            }

            // Normalize to 0-1 range
            noise_value = (noise_value + 1.0) / 2.0;

            // Add sparkle effect with high-frequency noise
            let sparkle_noise = perlin.get([nx * 32.0, ny * 32.0, 25.0]);
            let sparkle = (((sparkle_noise + 1.0) / 2.0).powf(3.0)) as f32; // Cubic for sharp sparkles

            // Blend colors based on noise values (cast to f32)
            let noise_value_f32 = noise_value as f32;
            let color = if noise_value < 0.2 {
                // Shadow areas
                blend_colors(&shadow_grey, &ice_blue, noise_value_f32 / 0.2)
            } else if noise_value < 0.5 {
                // Ice blue to light blue
                blend_colors(&ice_blue, &light_blue, (noise_value_f32 - 0.2) / 0.3)
            } else {
                // Light blue to pure white
                blend_colors(&light_blue, &pure_white, (noise_value_f32 - 0.5) / 0.5)
            };

            // Apply sparkle effect
            let r = (color[0] + sparkle * 20.0).clamp(0.0, 255.0) as u8;
            let g = (color[1] + sparkle * 20.0).clamp(0.0, 255.0) as u8;
            let b = (color[2] + sparkle * 20.0).clamp(0.0, 255.0) as u8;

            img.put_pixel(x, y, Rgba([r, g, b, 255]));
        }
    }

    img
}

/// Generate a UV test texture with checkerboard pattern and colored quadrants
///
/// Useful for debugging and visualizing UV mapping. Shows clear patterns
/// that help identify stretching, seams, and mapping issues.
///
/// # Arguments
/// * `size` - Resolution of the texture (typically 512, 1024, or 2048)
///
/// # Returns
/// * `RgbaImage` - RGBA image compatible with mesh-tools
pub fn generate_uv_test_texture(size: u32) -> RgbaImage {
    let mut img = RgbaImage::new(size, size);

    let checker_size = size / 16; // 16x16 checkerboard

    for y in 0..size {
        for x in 0..size {
            // Determine which quadrant we're in
            let quad_x = (x * 2) / size;
            let quad_y = (y * 2) / size;

            // Base color by quadrant
            let base_color = match (quad_x, quad_y) {
                (0, 0) => [255, 0, 0],     // Red - top left
                (1, 0) => [0, 255, 0],     // Green - top right
                (0, 1) => [0, 0, 255],     // Blue - bottom left
                (1, 1) => [255, 255, 0],   // Yellow - bottom right
                _ => [128, 128, 128],      // Grey (shouldn't happen)
            };

            // Checkerboard pattern
            let checker_x = x / checker_size;
            let checker_y = y / checker_size;
            let is_dark = (checker_x + checker_y) % 2 == 0;

            let r = if is_dark { base_color[0] / 2 } else { base_color[0] };
            let g = if is_dark { base_color[1] / 2 } else { base_color[1] };
            let b = if is_dark { base_color[2] / 2 } else { base_color[2] };

            // Add white borders at quadrant boundaries
            let border_width = size / 64;
            let is_border = x.abs_diff(size / 2) < border_width || y.abs_diff(size / 2) < border_width;

            if is_border {
                img.put_pixel(x, y, Rgba([255, 255, 255, 255]));
            } else {
                img.put_pixel(x, y, Rgba([r, g, b, 255]));
            }
        }
    }

    img
}

/// Convert pixel coordinates to torus topology for seamless tiling
///
/// Maps 2D coordinates to a 3D torus surface, ensuring that noise
/// values wrap seamlessly at texture boundaries.
///
/// # Arguments
/// * `x` - X coordinate in texture space (0..size)
/// * `y` - Y coordinate in texture space (0..size)
/// * `size` - Texture size
///
/// # Returns
/// * `(f64, f64)` - Normalized coordinates for noise sampling
fn torus_coords(x: u32, y: u32, size: u32) -> (f64, f64) {
    use std::f64::consts::PI;

    // Normalize to 0-1
    let u = x as f64 / size as f64;
    let v = y as f64 / size as f64;

    // Map to torus using parametric equations
    // This ensures seamless wrapping in both dimensions
    let angle_u = u * 2.0 * PI;
    let angle_v = v * 2.0 * PI;

    // Use a simple 2D circular mapping that preserves detail
    let nx = angle_u.cos() + angle_v.cos() * 0.5;
    let ny = angle_u.sin() + angle_v.sin() * 0.5;

    (nx, ny)
}

/// Blend two RGB colors with linear interpolation
///
/// # Arguments
/// * `color1` - First color as [r, g, b] in 0-255 range
/// * `color2` - Second color as [r, g, b] in 0-255 range
/// * `t` - Blend factor (0.0 = color1, 1.0 = color2)
///
/// # Returns
/// * `[f32; 3]` - Blended color in 0-255 range
fn blend_colors(color1: &[f32; 3], color2: &[f32; 3], t: f32) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    [
        color1[0] * (1.0 - t) + color2[0] * t,
        color1[1] * (1.0 - t) + color2[1] * t,
        color1[2] * (1.0 - t) + color2[2] * t,
    ]
}

// ============================================================================
// Normal Map Generation Functions
// ============================================================================

/// Generate a flat normal map (for testing/debugging)
///
/// All normals point straight up: RGB = (128, 128, 255), representing
/// normal vector (0, 0, 1) in tangent space. Useful for verifying
/// normal map pipeline without surface detail.
///
/// # Arguments
/// * `size` - Resolution of the normal map
///
/// # Returns
/// * `RgbaImage` - Flat tangent-space normal map
pub fn generate_flat_normal_map(size: u32) -> RgbaImage {
    let mut img = RgbaImage::new(size, size);

    // Flat surface: normal points up (0, 0, 1) in tangent space
    // Encoded: (0.5, 0.5, 1.0) -> RGB (128, 128, 255)
    let flat_normal = encode_tangent_normal(0.0, 0.0, 1.0);

    for y in 0..size {
        for x in 0..size {
            img.put_pixel(x, y, Rgba(flat_normal));
        }
    }

    img
}

/// Generate a rock normal map with cracks and surface detail
///
/// Uses multi-octave Perlin noise derivatives to create realistic
/// rocky surface perturbations. The normal map is tangent-space encoded
/// and seamlessly tileable using torus topology.
///
/// # Arguments
/// * `size` - Resolution of the normal map (typically 512, 1024, or 2048)
/// * `strength` - Normal strength multiplier (0.5-2.0, default 1.0)
///
/// # Returns
/// * `RgbaImage` - Tangent-space normal map (RGB encoded)
pub fn generate_rock_normal_map(size: u32, strength: f32) -> RgbaImage {
    let mut img = RgbaImage::new(size, size);
    let perlin = Perlin::new(42); // Match rock texture seed

    let octaves = 5; // Match rock texture octaves
    let z_offset = 0.0; // Match rock texture z_offset
    let scale = 1.0;
    let epsilon = 1.0 / (size as f64);

    for y in 0..size {
        for x in 0..size {
            // Use torus topology for seamless tiling
            let (nx, ny) = torus_coords(x, y, size);

            // Calculate height gradient using central difference
            let (du, dv) = torus_gradient(&perlin, nx, ny, size, octaves, z_offset, scale, epsilon, strength);

            // Construct tangent-space normal: (-du, -dv, 1.0) normalized
            let norm_x = -du;
            let norm_y = -dv;
            let norm_z = 1.0;

            // Encode to RGB
            let rgba = encode_tangent_normal(norm_x, norm_y, norm_z);
            img.put_pixel(x, y, Rgba(rgba));
        }
    }

    img
}

/// Generate a grass normal map with blade-like patterns
///
/// Creates directional grass blade normals for realistic lighting.
/// Uses multi-octave noise patterns to simulate grass orientation
/// and fine surface detail.
///
/// # Arguments
/// * `size` - Resolution of the normal map (typically 512, 1024, or 2048)
/// * `strength` - Normal strength multiplier (0.5-2.0, default 1.0)
///
/// # Returns
/// * `RgbaImage` - Tangent-space normal map (RGB encoded)
pub fn generate_grass_normal_map(size: u32, strength: f32) -> RgbaImage {
    let mut img = RgbaImage::new(size, size);
    let perlin = Perlin::new(123); // Match grass texture seed

    let octaves = 4; // Match grass texture octaves
    let z_offset = 5.0; // Match grass texture z_offset
    let scale = 1.0;
    let epsilon = 1.0 / (size as f64);

    for y in 0..size {
        for x in 0..size {
            // Use torus topology for seamless tiling
            let (nx, ny) = torus_coords(x, y, size);

            // Calculate height gradient using central difference
            let (du, dv) = torus_gradient(&perlin, nx, ny, size, octaves, z_offset, scale, epsilon, strength);

            // Construct tangent-space normal: (-du, -dv, 1.0) normalized
            let norm_x = -du;
            let norm_y = -dv;
            let norm_z = 1.0;

            // Encode to RGB
            let rgba = encode_tangent_normal(norm_x, norm_y, norm_z);
            img.put_pixel(x, y, Rgba(rgba));
        }
    }

    img
}

// ============================================================================
// Biome Configuration System
// ============================================================================

/// Supported mountain biome types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BiomeType {
    /// Alpine mountains: grass base, rocky mid-slopes, snow peaks
    Alpine,
    /// Desert/arid mountains: minimal vegetation, red/tan rocks, little snow
    Desert,
    /// Volcanic mountains: dark basalt, ash deposits, dramatic peaks
    Volcanic,
    /// Arctic/glacial mountains: tundra base, extensive ice and snow
    Arctic,
}

impl BiomeType {
    /// Parse biome type from string (case-insensitive)
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "alpine" => Some(BiomeType::Alpine),
            "desert" | "arid" => Some(BiomeType::Desert),
            "volcanic" | "volcano" => Some(BiomeType::Volcanic),
            "arctic" | "glacial" => Some(BiomeType::Arctic),
            _ => None,
        }
    }

    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            BiomeType::Alpine => "Alpine",
            BiomeType::Desert => "Desert",
            BiomeType::Volcanic => "Volcanic",
            BiomeType::Arctic => "Arctic",
        }
    }
}

/// Configuration for a biome's material properties
#[derive(Debug, Clone)]
pub struct BiomeConfig {
    /// Biome type
    pub biome_type: BiomeType,
    /// Rock texture variant to use
    pub rock_type: RockType,
    /// Snow coverage factor (0.0 = minimal, 1.0 = extensive)
    pub snow_coverage: f32,
    /// Snow texture variant to use
    pub snow_type: SnowType,
    /// Base/low altitude material type
    pub base_material: BaseMaterialType,
    /// PBR roughness value for rock (0.0-1.0)
    pub rock_roughness: f32,
    /// PBR roughness value for snow (0.0-1.0)
    pub snow_roughness: f32,
}

/// Types of rock textures available
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RockType {
    /// Grey granite (default rock texture)
    Granite,
    /// Red/tan sedimentary rock
    Sandstone,
    /// Dark volcanic basalt
    Basalt,
    /// Light-colored limestone
    Limestone,
}

/// Types of snow textures available
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnowType {
    /// Fresh, powdery snow (bright white)
    Fresh,
    /// Glacial ice (blue-tinted, smooth)
    Ice,
    /// Mixed ice and snow
    IcySnow,
}

/// Base material types for low altitudes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaseMaterialType {
    /// Green grass/vegetation
    Grass,
    /// Sparse alpine vegetation
    Alpine,
    /// Desert soil/sand
    Desert,
    /// Volcanic soil and ash
    Volcanic,
    /// Arctic tundra
    Tundra,
}

impl BiomeConfig {
    /// Create configuration for Alpine biome
    pub fn alpine() -> Self {
        BiomeConfig {
            biome_type: BiomeType::Alpine,
            rock_type: RockType::Granite,
            snow_coverage: 0.6,
            snow_type: SnowType::Fresh,
            base_material: BaseMaterialType::Grass,
            rock_roughness: 0.85,
            snow_roughness: 0.4,
        }
    }

    /// Create configuration for Desert biome
    pub fn desert() -> Self {
        BiomeConfig {
            biome_type: BiomeType::Desert,
            rock_type: RockType::Sandstone,
            snow_coverage: 0.1,
            snow_type: SnowType::Fresh,
            base_material: BaseMaterialType::Desert,
            rock_roughness: 0.75,
            snow_roughness: 0.5,
        }
    }

    /// Create configuration for Volcanic biome
    pub fn volcanic() -> Self {
        BiomeConfig {
            biome_type: BiomeType::Volcanic,
            rock_type: RockType::Basalt,
            snow_coverage: 0.3,
            snow_type: SnowType::IcySnow,
            base_material: BaseMaterialType::Volcanic,
            rock_roughness: 0.9,
            snow_roughness: 0.35,
        }
    }

    /// Create configuration for Arctic biome
    pub fn arctic() -> Self {
        BiomeConfig {
            biome_type: BiomeType::Arctic,
            rock_type: RockType::Granite,
            snow_coverage: 0.9,
            snow_type: SnowType::Ice,
            base_material: BaseMaterialType::Tundra,
            rock_roughness: 0.8,
            snow_roughness: 0.25,
        }
    }

    /// Get biome configuration by type
    pub fn from_biome_type(biome: BiomeType) -> Self {
        match biome {
            BiomeType::Alpine => Self::alpine(),
            BiomeType::Desert => Self::desert(),
            BiomeType::Volcanic => Self::volcanic(),
            BiomeType::Arctic => Self::arctic(),
        }
    }
}

// ============================================================================
// Enhanced Rock Texture Generation
// ============================================================================

/// Generate rock texture based on type
pub fn generate_rock_texture_typed(size: u32, rock_type: RockType) -> RgbaImage {
    match rock_type {
        RockType::Granite => generate_granite_texture(size),
        RockType::Sandstone => generate_sandstone_texture(size),
        RockType::Basalt => generate_basalt_texture(size),
        RockType::Limestone => generate_limestone_texture(size),
    }
}

/// Generate granite texture with grey tones and mineral variation
///
/// Enhanced version with better color variation, lichen patches, and
/// geological detail. Uses multi-octave noise for realistic appearance.
pub fn generate_granite_texture(size: u32) -> RgbaImage {
    let mut img = RgbaImage::new(size, size);
    let perlin = Perlin::new(42);
    let detail_perlin = Perlin::new(100);

    // Enhanced color palette with more variation
    let dark_grey = [45.0, 45.0, 50.0];
    let med_grey = [85.0, 85.0, 90.0];
    let light_grey = [130.0, 130.0, 135.0];
    let quartz = [150.0, 150.0, 155.0];
    let lichen_green = [80.0, 100.0, 70.0];
    let rust_brown = [110.0, 85.0, 60.0];

    for y in 0..size {
        for x in 0..size {
            let (nx, ny) = torus_coords(x, y, size);

            // Base granite noise (multiple octaves)
            let mut base_noise = 0.0;
            let mut amplitude = 1.0;
            let mut frequency = 1.0;
            for _ in 0..5 {
                base_noise += amplitude * perlin.get([nx * frequency, ny * frequency, 0.0]);
                amplitude *= 0.5;
                frequency *= 2.0;
            }
            base_noise = (base_noise + 1.0) / 2.0;

            // Mineral speckles (high frequency)
            let speckle = perlin.get([nx * 20.0, ny * 20.0, 5.0]);
            let is_mineral = speckle > 0.7;

            // Lichen patches (lower frequency)
            let lichen_noise = detail_perlin.get([nx * 3.0, ny * 3.0, 10.0]);
            let has_lichen = lichen_noise > 0.5;

            // Rust/weathering stains
            let rust_noise = detail_perlin.get([nx * 2.0, ny * 2.0, 20.0]);
            let has_rust = rust_noise > 0.6;

            // Base rock color (cast to f32)
            let base_noise_f32 = base_noise as f32;
            let base_color = if base_noise < 0.3 {
                blend_colors(&dark_grey, &med_grey, base_noise_f32 / 0.3)
            } else if base_noise < 0.7 {
                blend_colors(&med_grey, &light_grey, (base_noise_f32 - 0.3) / 0.4)
            } else {
                blend_colors(&light_grey, &quartz, (base_noise_f32 - 0.7) / 0.3)
            };

            // Apply modifiers
            let final_color = if is_mineral {
                // Bright quartz/feldspar crystals
                blend_colors(&base_color, &quartz, 0.6)
            } else if has_lichen {
                // Green lichen patches
                blend_colors(&base_color, &lichen_green, 0.4)
            } else if has_rust {
                // Rust/iron staining
                blend_colors(&base_color, &rust_brown, 0.3)
            } else {
                base_color
            };

            let r = (final_color[0]).clamp(0.0, 255.0) as u8;
            let g = (final_color[1]).clamp(0.0, 255.0) as u8;
            let b = (final_color[2]).clamp(0.0, 255.0) as u8;

            img.put_pixel(x, y, Rgba([r, g, b, 255]));
        }
    }

    img
}

/// Generate sandstone texture with layered sedimentary patterns
///
/// Red/tan colored rock with horizontal stratification typical of
/// desert and arid mountain environments.
pub fn generate_sandstone_texture(size: u32) -> RgbaImage {
    let mut img = RgbaImage::new(size, size);
    let perlin = Perlin::new(200);
    let layer_perlin = Perlin::new(201);

    // Desert sandstone color palette
    let dark_red = [140.0, 80.0, 50.0];
    let red_brown = [170.0, 100.0, 60.0];
    let tan = [200.0, 150.0, 100.0];
    let light_tan = [220.0, 180.0, 130.0];
    let desert_yellow = [210.0, 190.0, 140.0];

    for y in 0..size {
        for x in 0..size {
            let (nx, ny) = torus_coords(x, y, size);

            // Horizontal stratification (emphasize Y-axis patterns)
            let layer_height = ny * 8.0 + layer_perlin.get([nx * 2.0, ny * 0.5, 0.0]);
            let layer_value = (layer_height.sin() + 1.0) / 2.0;

            // Base texture noise
            let mut base_noise = 0.0;
            let mut amplitude = 1.0;
            let mut frequency = 1.0;
            for _ in 0..4 {
                base_noise += amplitude * perlin.get([nx * frequency, ny * frequency, 10.0]);
                amplitude *= 0.5;
                frequency *= 2.0;
            }
            base_noise = (base_noise + 1.0) / 2.0;

            // Combine layering with base texture
            let combined = layer_value * 0.6 + base_noise * 0.4;
            let combined_f32 = combined as f32;

            let color = if combined < 0.2 {
                blend_colors(&dark_red, &red_brown, combined_f32 / 0.2)
            } else if combined < 0.5 {
                blend_colors(&red_brown, &tan, (combined_f32 - 0.2) / 0.3)
            } else if combined < 0.8 {
                blend_colors(&tan, &light_tan, (combined_f32 - 0.5) / 0.3)
            } else {
                blend_colors(&light_tan, &desert_yellow, (combined_f32 - 0.8) / 0.2)
            };

            // Add fine sand grain texture
            let grain = perlin.get([nx * 25.0, ny * 25.0, 15.0]);
            let grain_factor = (0.9 + ((grain + 1.0) / 2.0) * 0.2) as f32;

            let r = (color[0] * grain_factor).clamp(0.0, 255.0) as u8;
            let g = (color[1] * grain_factor).clamp(0.0, 255.0) as u8;
            let b = (color[2] * grain_factor).clamp(0.0, 255.0) as u8;

            img.put_pixel(x, y, Rgba([r, g, b, 255]));
        }
    }

    img
}

/// Generate basalt texture with dark volcanic rock patterns
///
/// Dark grey to black volcanic rock with occasional vesicles (gas bubbles)
/// and columnar joint patterns typical of cooled lava.
pub fn generate_basalt_texture(size: u32) -> RgbaImage {
    let mut img = RgbaImage::new(size, size);
    let perlin = Perlin::new(300);
    let vesicle_perlin = Perlin::new(301);

    // Volcanic basalt color palette (dark!)
    let black = [20.0, 20.0, 25.0];
    let dark_grey = [40.0, 40.0, 45.0];
    let basalt_grey = [60.0, 58.0, 62.0];
    let light_basalt = [75.0, 72.0, 78.0];
    let weathered = [85.0, 80.0, 75.0];

    for y in 0..size {
        for x in 0..size {
            let (nx, ny) = torus_coords(x, y, size);

            // Base basalt texture
            let mut base_noise = 0.0;
            let mut amplitude = 1.0;
            let mut frequency = 1.0;
            for _ in 0..5 {
                base_noise += amplitude * perlin.get([nx * frequency, ny * frequency, 20.0]);
                amplitude *= 0.5;
                frequency *= 2.0;
            }
            base_noise = (base_noise + 1.0) / 2.0;

            // Vesicles (gas bubble holes)
            let vesicle_noise = vesicle_perlin.get([nx * 15.0, ny * 15.0, 25.0]);
            let is_vesicle = vesicle_noise > 0.75;

            // Weathering (lighter areas)
            let weathering = vesicle_perlin.get([nx * 2.0, ny * 2.0, 30.0]);
            let is_weathered = weathering > 0.6;

            let base_noise_f32 = base_noise as f32;
            let base_color = if base_noise < 0.25 {
                blend_colors(&black, &dark_grey, base_noise_f32 / 0.25)
            } else if base_noise < 0.6 {
                blend_colors(&dark_grey, &basalt_grey, (base_noise_f32 - 0.25) / 0.35)
            } else {
                blend_colors(&basalt_grey, &light_basalt, (base_noise_f32 - 0.6) / 0.4)
            };

            let final_color = if is_vesicle {
                // Very dark holes
                blend_colors(&base_color, &black, 0.7)
            } else if is_weathered {
                // Lighter weathered patches
                blend_colors(&base_color, &weathered, 0.4)
            } else {
                base_color
            };

            let r = (final_color[0]).clamp(0.0, 255.0) as u8;
            let g = (final_color[1]).clamp(0.0, 255.0) as u8;
            let b = (final_color[2]).clamp(0.0, 255.0) as u8;

            img.put_pixel(x, y, Rgba([r, g, b, 255]));
        }
    }

    img
}

/// Generate limestone texture with light colors and fossil patterns
///
/// Light grey to white sedimentary rock, often with visible bedding planes
/// and occasional darker streaks from mineral deposits.
pub fn generate_limestone_texture(size: u32) -> RgbaImage {
    let mut img = RgbaImage::new(size, size);
    let perlin = Perlin::new(400);
    let bedding_perlin = Perlin::new(401);

    // Limestone color palette (light!)
    let white = [240.0, 240.0, 245.0];
    let light_grey = [210.0, 210.0, 215.0];
    let cream = [230.0, 225.0, 210.0];
    let grey = [180.0, 180.0, 185.0];
    let dark_vein = [120.0, 120.0, 125.0];

    for y in 0..size {
        for x in 0..size {
            let (nx, ny) = torus_coords(x, y, size);

            // Base limestone texture
            let mut base_noise = 0.0;
            let mut amplitude = 1.0;
            let mut frequency = 1.0;
            for _ in 0..4 {
                base_noise += amplitude * perlin.get([nx * frequency, ny * frequency, 30.0]);
                amplitude *= 0.5;
                frequency *= 2.0;
            }
            base_noise = (base_noise + 1.0) / 2.0;

            // Bedding planes (subtle horizontal layers)
            let bedding = bedding_perlin.get([nx * 1.0, ny * 6.0, 35.0]);
            let bedding_value = (bedding + 1.0) / 2.0;

            // Dark mineral veins
            let vein_noise = perlin.get([nx * 12.0, ny * 12.0, 40.0]);
            let is_vein = vein_noise > 0.8;

            // Combine base texture with bedding
            let combined = base_noise * 0.7 + bedding_value * 0.3;
            let combined_f32 = combined as f32;

            let base_color = if combined < 0.3 {
                blend_colors(&grey, &light_grey, combined_f32 / 0.3)
            } else if combined < 0.6 {
                blend_colors(&light_grey, &cream, (combined_f32 - 0.3) / 0.3)
            } else {
                blend_colors(&cream, &white, (combined_f32 - 0.6) / 0.4)
            };

            let final_color = if is_vein {
                // Dark mineral veins
                blend_colors(&base_color, &dark_vein, 0.5)
            } else {
                base_color
            };

            let r = (final_color[0]).clamp(0.0, 255.0) as u8;
            let g = (final_color[1]).clamp(0.0, 255.0) as u8;
            let b = (final_color[2]).clamp(0.0, 255.0) as u8;

            img.put_pixel(x, y, Rgba([r, g, b, 255]));
        }
    }

    img
}

// ============================================================================
// Enhanced Snow and Ice Texture Generation
// ============================================================================

/// Generate snow texture based on type
pub fn generate_snow_texture_typed(size: u32, snow_type: SnowType) -> RgbaImage {
    match snow_type {
        SnowType::Fresh => generate_fresh_snow_texture(size),
        SnowType::Ice => generate_ice_texture(size),
        SnowType::IcySnow => generate_icy_snow_texture(size),
    }
}

/// Generate fresh snow texture with bright white and fluffy appearance
///
/// Bright white powdery snow with soft variation and gentle sparkle effects.
/// This is the classic "mountain peak" snow appearance.
pub fn generate_fresh_snow_texture(size: u32) -> RgbaImage {
    let mut img = RgbaImage::new(size, size);
    let perlin = Perlin::new(789);

    // Fresh snow color palette (very bright!)
    let pure_white = [255.0, 255.0, 255.0];
    let snow_white = [250.0, 250.0, 252.0];
    let light_blue = [240.0, 245.0, 252.0];
    let shadow = [220.0, 225.0, 235.0];

    for y in 0..size {
        for x in 0..size {
            let (nx, ny) = torus_coords(x, y, size);

            // Soft multi-octave noise for snow variation
            let mut noise_value = 0.0;
            let mut amplitude = 1.0;
            let mut frequency = 1.0;
            for _ in 0..4 {
                noise_value += amplitude * perlin.get([nx * frequency, ny * frequency, 20.0]);
                amplitude *= 0.5;
                frequency *= 2.0;
            }
            noise_value = (noise_value + 1.0) / 2.0;

            // Sparkle effect (high frequency, subtle)
            let sparkle = perlin.get([nx * 35.0, ny * 35.0, 25.0]);
            let sparkle_value = if sparkle > 0.8 {
                ((sparkle - 0.8) / 0.2) as f32
            } else {
                0.0
            };

            let noise_value_f32 = noise_value as f32;
            let base_color = if noise_value < 0.15 {
                blend_colors(&shadow, &light_blue, noise_value_f32 / 0.15)
            } else if noise_value < 0.5 {
                blend_colors(&light_blue, &snow_white, (noise_value_f32 - 0.15) / 0.35)
            } else {
                blend_colors(&snow_white, &pure_white, (noise_value_f32 - 0.5) / 0.5)
            };

            // Apply sparkle
            let r = (base_color[0] + sparkle_value * 5.0).clamp(0.0, 255.0) as u8;
            let g = (base_color[1] + sparkle_value * 5.0).clamp(0.0, 255.0) as u8;
            let b = (base_color[2] + sparkle_value * 5.0).clamp(0.0, 255.0) as u8;

            img.put_pixel(x, y, Rgba([r, g, b, 255]));
        }
    }

    img
}

/// Generate glacial ice texture with blue tint and smooth appearance
///
/// Blue-tinted compressed ice typical of glaciers and ice cliffs.
/// Very smooth with internal crystal structure visible.
pub fn generate_ice_texture(size: u32) -> RgbaImage {
    let mut img = RgbaImage::new(size, size);
    let perlin = Perlin::new(800);
    let crack_perlin = Perlin::new(801);

    // Glacial ice color palette (blue-tinted!)
    let deep_blue = [180.0, 205.0, 230.0];
    let ice_blue = [200.0, 220.0, 240.0];
    let light_ice = [220.0, 235.0, 248.0];
    let white_ice = [235.0, 245.0, 252.0];

    for y in 0..size {
        for x in 0..size {
            let (nx, ny) = torus_coords(x, y, size);

            // Smooth ice texture (fewer octaves than snow)
            let mut noise_value = 0.0;
            let mut amplitude = 1.0;
            let mut frequency = 1.0;
            for _ in 0..3 {
                noise_value += amplitude * perlin.get([nx * frequency, ny * frequency, 30.0]);
                amplitude *= 0.5;
                frequency *= 2.0;
            }
            noise_value = (noise_value + 1.0) / 2.0;

            // Internal cracks and crystal boundaries
            let crack_noise = crack_perlin.get([nx * 8.0, ny * 8.0, 35.0]);
            let is_crack = crack_noise > 0.75;

            // Ice bubbles (small dark spots)
            let bubble_noise = perlin.get([nx * 18.0, ny * 18.0, 40.0]);
            let has_bubble = bubble_noise > 0.8;

            let noise_value_f32 = noise_value as f32;
            let base_color = if noise_value < 0.25 {
                blend_colors(&deep_blue, &ice_blue, noise_value_f32 / 0.25)
            } else if noise_value < 0.6 {
                blend_colors(&ice_blue, &light_ice, (noise_value_f32 - 0.25) / 0.35)
            } else {
                blend_colors(&light_ice, &white_ice, (noise_value_f32 - 0.6) / 0.4)
            };

            let final_color = if has_bubble {
                // Darken for air bubbles
                blend_colors(&base_color, &deep_blue, 0.6)
            } else if is_crack {
                // Lighten for crystal boundaries
                blend_colors(&base_color, &white_ice, 0.3)
            } else {
                base_color
            };

            let r = (final_color[0]).clamp(0.0, 255.0) as u8;
            let g = (final_color[1]).clamp(0.0, 255.0) as u8;
            let b = (final_color[2]).clamp(0.0, 255.0) as u8;

            img.put_pixel(x, y, Rgba([r, g, b, 255]));
        }
    }

    img
}

/// Generate mixed ice and snow texture
///
/// Combination of icy patches and snow, typical of wind-blown peaks
/// or partially melted/refrozen snow. Medium blue tint.
pub fn generate_icy_snow_texture(size: u32) -> RgbaImage {
    let mut img = RgbaImage::new(size, size);
    let perlin = Perlin::new(850);
    let mix_perlin = Perlin::new(851);

    // Mixed ice/snow palette
    let ice_blue = [210.0, 225.0, 240.0];
    let snow_white = [245.0, 248.0, 252.0];
    let pure_white = [255.0, 255.0, 255.0];
    let shadow_blue = [200.0, 215.0, 230.0];

    for y in 0..size {
        for x in 0..size {
            let (nx, ny) = torus_coords(x, y, size);

            // Determine ice vs snow ratio at this pixel
            let mix_noise = mix_perlin.get([nx * 4.0, ny * 4.0, 0.0]);
            let ice_factor = ((mix_noise + 1.0) / 2.0) as f32;

            // Base texture
            let mut noise_value = 0.0;
            let mut amplitude = 1.0;
            let mut frequency = 1.0;
            for _ in 0..4 {
                noise_value += amplitude * perlin.get([nx * frequency, ny * frequency, 45.0]);
                amplitude *= 0.5;
                frequency *= 2.0;
            }
            noise_value = (noise_value + 1.0) / 2.0;

            // Choose color based on whether this area is more ice or snow
            let noise_value_f32 = noise_value as f32;
            let base_color = if ice_factor > 0.6 {
                // More icy areas
                if noise_value < 0.4 {
                    blend_colors(&shadow_blue, &ice_blue, noise_value_f32 / 0.4)
                } else {
                    blend_colors(&ice_blue, &snow_white, (noise_value_f32 - 0.4) / 0.6)
                }
            } else {
                // More snowy areas
                if noise_value < 0.5 {
                    blend_colors(&snow_white, &pure_white, noise_value_f32 / 0.5)
                } else {
                    pure_white
                }
            };

            let r = (base_color[0]).clamp(0.0, 255.0) as u8;
            let g = (base_color[1]).clamp(0.0, 255.0) as u8;
            let b = (base_color[2]).clamp(0.0, 255.0) as u8;

            img.put_pixel(x, y, Rgba([r, g, b, 255]));
        }
    }

    img
}

// ============================================================================
// Base Material Textures (for low altitudes / valleys)
// ============================================================================

/// Generate base material texture based on type
pub fn generate_base_material_texture(size: u32, material_type: BaseMaterialType) -> RgbaImage {
    match material_type {
        BaseMaterialType::Grass => generate_grass_texture(size),
        BaseMaterialType::Alpine => generate_alpine_vegetation_texture(size),
        BaseMaterialType::Desert => generate_desert_soil_texture(size),
        BaseMaterialType::Volcanic => generate_volcanic_soil_texture(size),
        BaseMaterialType::Tundra => generate_tundra_texture(size),
    }
}

/// Generate desert soil texture with sandy, arid appearance
///
/// Tan/beige sandy soil with sparse rocky patches, typical of arid environments.
pub fn generate_desert_soil_texture(size: u32) -> RgbaImage {
    let mut img = RgbaImage::new(size, size);
    let perlin = Perlin::new(500);

    // Desert soil palette
    let dark_tan = [150.0, 120.0, 80.0];
    let tan = [180.0, 150.0, 100.0];
    let light_tan = [200.0, 170.0, 120.0];
    let sand = [210.0, 185.0, 140.0];

    for y in 0..size {
        for x in 0..size {
            let (nx, ny) = torus_coords(x, y, size);

            let mut noise_value = 0.0;
            let mut amplitude = 1.0;
            let mut frequency = 1.0;
            for _ in 0..4 {
                noise_value += amplitude * perlin.get([nx * frequency, ny * frequency, 50.0]);
                amplitude *= 0.5;
                frequency *= 2.0;
            }
            noise_value = (noise_value + 1.0) / 2.0;

            let noise_value_f32 = noise_value as f32;
            let color = if noise_value < 0.3 {
                blend_colors(&dark_tan, &tan, noise_value_f32 / 0.3)
            } else if noise_value < 0.6 {
                blend_colors(&tan, &light_tan, (noise_value_f32 - 0.3) / 0.3)
            } else {
                blend_colors(&light_tan, &sand, (noise_value_f32 - 0.6) / 0.4)
            };

            let r = (color[0]).clamp(0.0, 255.0) as u8;
            let g = (color[1]).clamp(0.0, 255.0) as u8;
            let b = (color[2]).clamp(0.0, 255.0) as u8;

            img.put_pixel(x, y, Rgba([r, g, b, 255]));
        }
    }

    img
}

/// Generate volcanic soil texture with dark ash and lava deposits
///
/// Dark grey to brown soil with ash deposits, typical of volcanic regions.
pub fn generate_volcanic_soil_texture(size: u32) -> RgbaImage {
    let mut img = RgbaImage::new(size, size);
    let perlin = Perlin::new(550);

    // Volcanic soil palette
    let black_ash = [35.0, 30.0, 30.0];
    let dark_ash = [55.0, 50.0, 45.0];
    let grey_ash = [75.0, 70.0, 65.0];
    let brown_soil = [90.0, 75.0, 60.0];

    for y in 0..size {
        for x in 0..size {
            let (nx, ny) = torus_coords(x, y, size);

            let mut noise_value = 0.0;
            let mut amplitude = 1.0;
            let mut frequency = 1.0;
            for _ in 0..5 {
                noise_value += amplitude * perlin.get([nx * frequency, ny * frequency, 55.0]);
                amplitude *= 0.5;
                frequency *= 2.0;
            }
            noise_value = (noise_value + 1.0) / 2.0;

            let noise_value_f32 = noise_value as f32;
            let color = if noise_value < 0.3 {
                blend_colors(&black_ash, &dark_ash, noise_value_f32 / 0.3)
            } else if noise_value < 0.6 {
                blend_colors(&dark_ash, &grey_ash, (noise_value_f32 - 0.3) / 0.3)
            } else {
                blend_colors(&grey_ash, &brown_soil, (noise_value_f32 - 0.6) / 0.4)
            };

            let r = (color[0]).clamp(0.0, 255.0) as u8;
            let g = (color[1]).clamp(0.0, 255.0) as u8;
            let b = (color[2]).clamp(0.0, 255.0) as u8;

            img.put_pixel(x, y, Rgba([r, g, b, 255]));
        }
    }

    img
}

/// Generate alpine vegetation texture with sparse, hardy plants
///
/// Mix of sparse grass, rocks, and alpine flowers. Less lush than lowland grass.
pub fn generate_alpine_vegetation_texture(size: u32) -> RgbaImage {
    let mut img = RgbaImage::new(size, size);
    let perlin = Perlin::new(600);
    let rock_perlin = Perlin::new(601);

    // Alpine vegetation palette
    let rock_grey = [100.0, 95.0, 90.0];
    let brown_soil = [120.0, 100.0, 75.0];
    let sparse_green = [90.0, 120.0, 75.0];
    let alpine_green = [110.0, 140.0, 90.0];

    for y in 0..size {
        for x in 0..size {
            let (nx, ny) = torus_coords(x, y, size);

            // Rocky patches
            let rock_noise = rock_perlin.get([nx * 6.0, ny * 6.0, 60.0]);
            let is_rocky = rock_noise > 0.5;

            let mut noise_value = 0.0;
            let mut amplitude = 1.0;
            let mut frequency = 1.0;
            for _ in 0..4 {
                noise_value += amplitude * perlin.get([nx * frequency, ny * frequency, 65.0]);
                amplitude *= 0.5;
                frequency *= 2.0;
            }
            noise_value = (noise_value + 1.0) / 2.0;

            let noise_value_f32 = noise_value as f32;
            let color = if is_rocky {
                // Rocky areas
                blend_colors(&rock_grey, &brown_soil, noise_value_f32)
            } else {
                // Sparse vegetation
                if noise_value < 0.4 {
                    blend_colors(&brown_soil, &sparse_green, noise_value_f32 / 0.4)
                } else {
                    blend_colors(&sparse_green, &alpine_green, (noise_value_f32 - 0.4) / 0.6)
                }
            };

            let r = (color[0]).clamp(0.0, 255.0) as u8;
            let g = (color[1]).clamp(0.0, 255.0) as u8;
            let b = (color[2]).clamp(0.0, 255.0) as u8;

            img.put_pixel(x, y, Rgba([r, g, b, 255]));
        }
    }

    img
}

/// Generate tundra texture with moss, lichen, and sparse vegetation
///
/// Cold-climate ground cover with mosses, lichens, and minimal plants.
pub fn generate_tundra_texture(size: u32) -> RgbaImage {
    let mut img = RgbaImage::new(size, size);
    let perlin = Perlin::new(650);
    let lichen_perlin = Perlin::new(651);

    // Tundra palette
    let dark_moss = [60.0, 75.0, 55.0];
    let moss_green = [80.0, 95.0, 70.0];
    let lichen_grey = [110.0, 120.0, 100.0];
    let pale_green = [130.0, 140.0, 115.0];

    for y in 0..size {
        for x in 0..size {
            let (nx, ny) = torus_coords(x, y, size);

            // Lichen patches
            let lichen_noise = lichen_perlin.get([nx * 5.0, ny * 5.0, 70.0]);
            let has_lichen = lichen_noise > 0.55;

            let mut noise_value = 0.0;
            let mut amplitude = 1.0;
            let mut frequency = 1.0;
            for _ in 0..4 {
                noise_value += amplitude * perlin.get([nx * frequency, ny * frequency, 75.0]);
                amplitude *= 0.5;
                frequency *= 2.0;
            }
            noise_value = (noise_value + 1.0) / 2.0;

            let noise_value_f32 = noise_value as f32;
            let color = if has_lichen {
                // Light lichen areas
                blend_colors(&lichen_grey, &pale_green, noise_value_f32)
            } else {
                // Moss areas
                if noise_value < 0.5 {
                    blend_colors(&dark_moss, &moss_green, noise_value_f32 / 0.5)
                } else {
                    blend_colors(&moss_green, &lichen_grey, (noise_value_f32 - 0.5) / 0.5)
                }
            };

            let r = (color[0]).clamp(0.0, 255.0) as u8;
            let g = (color[1]).clamp(0.0, 255.0) as u8;
            let b = (color[2]).clamp(0.0, 255.0) as u8;

            img.put_pixel(x, y, Rgba([r, g, b, 255]));
        }
    }

    img
}

// ============================================================================
// PBR Roughness Map Generation
// ============================================================================

/// Generate roughness map for a given material type
///
/// Roughness maps are greyscale images where:
/// - 0 (black) = perfectly smooth/glossy
/// - 255 (white) = maximally rough/matte
///
/// # Arguments
/// * `size` - Resolution of the roughness map
/// * `base_roughness` - Base roughness value (0.0-1.0)
/// * `variation` - Amount of roughness variation (0.0-1.0)
/// * `seed` - Noise seed for variation pattern
///
/// # Returns
/// * `RgbaImage` - Greyscale roughness map (R=G=B=roughness, A=255)
pub fn generate_roughness_map(size: u32, base_roughness: f32, variation: f32, seed: u32) -> RgbaImage {
    let mut img = RgbaImage::new(size, size);
    let perlin = Perlin::new(seed);

    // Convert base roughness (0.0-1.0) to 0-255 range
    let base_value = (base_roughness * 255.0).clamp(0.0, 255.0);
    let variation_range = (variation * 128.0).clamp(0.0, 128.0);

    for y in 0..size {
        for x in 0..size {
            let (nx, ny) = torus_coords(x, y, size);

            // Multi-octave noise for variation
            let mut noise_value = 0.0;
            let mut amplitude = 1.0;
            let mut frequency = 1.0;
            for _ in 0..3 {
                noise_value += amplitude * perlin.get([nx * frequency, ny * frequency, 0.0]);
                amplitude *= 0.5;
                frequency *= 2.0;
            }
            noise_value = (noise_value + 1.0) / 2.0; // Normalize to 0-1

            // Apply variation around base roughness
            let final_roughness = base_value + (noise_value as f32 - 0.5) * variation_range * 2.0;
            let clamped = final_roughness.clamp(0.0, 255.0) as u8;

            // Greyscale: R=G=B=roughness
            img.put_pixel(x, y, Rgba([clamped, clamped, clamped, 255]));
        }
    }

    img
}

/// Generate roughness map for rock materials
pub fn generate_rock_roughness_map(size: u32, rock_type: RockType) -> RgbaImage {
    let (base, variation, seed) = match rock_type {
        RockType::Granite => (0.85, 0.15, 1042),
        RockType::Sandstone => (0.75, 0.20, 1200),
        RockType::Basalt => (0.90, 0.10, 1300),
        RockType::Limestone => (0.80, 0.18, 1400),
    };
    generate_roughness_map(size, base, variation, seed)
}

/// Generate roughness map for snow materials
pub fn generate_snow_roughness_map(size: u32, snow_type: SnowType) -> RgbaImage {
    let (base, variation, seed) = match snow_type {
        SnowType::Fresh => (0.45, 0.15, 1789),
        SnowType::Ice => (0.25, 0.10, 1800),
        SnowType::IcySnow => (0.35, 0.15, 1850),
    };
    generate_roughness_map(size, base, variation, seed)
}

/// Generate roughness map for base materials
pub fn generate_base_material_roughness_map(size: u32, material_type: BaseMaterialType) -> RgbaImage {
    let (base, variation, seed) = match material_type {
        BaseMaterialType::Grass => (0.90, 0.10, 1123),
        BaseMaterialType::Alpine => (0.85, 0.15, 1600),
        BaseMaterialType::Desert => (0.70, 0.20, 1500),
        BaseMaterialType::Volcanic => (0.85, 0.12, 1550),
        BaseMaterialType::Tundra => (0.80, 0.15, 1650),
    };
    generate_roughness_map(size, base, variation, seed)
}

/// Generate a snow normal map with crystalline microstructure
///
/// Subtle normal variations for icy, crystalline snow appearance.
/// Lower frequency than rock, with sparkle-like perturbations.
///
/// # Arguments
/// * `size` - Resolution of the normal map (typically 512, 1024, or 2048)
/// * `strength` - Normal strength multiplier (0.3-1.0, default 0.5 for smooth snow)
///
/// # Returns
/// * `RgbaImage` - Tangent-space normal map (RGB encoded)
pub fn generate_snow_normal_map(size: u32, strength: f32) -> RgbaImage {
    let mut img = RgbaImage::new(size, size);
    let perlin = Perlin::new(789); // Match snow texture seed

    let octaves = 4; // Match snow texture octaves
    let z_offset = 20.0; // Match snow texture z_offset
    let scale = 0.5; // Lower scale for smoother surface
    let epsilon = 1.0 / (size as f64);

    // Snow has subtle normals (smoother surface)
    let adjusted_strength = strength * 0.5;

    for y in 0..size {
        for x in 0..size {
            // Use torus topology for seamless tiling
            let (nx, ny) = torus_coords(x, y, size);

            // Calculate height gradient using central difference
            let (du, dv) = torus_gradient(&perlin, nx, ny, size, octaves, z_offset, scale, epsilon, adjusted_strength);

            // Construct tangent-space normal: (-du, -dv, 1.0) normalized
            let norm_x = -du;
            let norm_y = -dv;
            let norm_z = 1.0;

            // Encode to RGB
            let rgba = encode_tangent_normal(norm_x, norm_y, norm_z);
            img.put_pixel(x, y, Rgba(rgba));
        }
    }

    img
}

// ============================================================================
// Terrain Analysis Helper Functions
// ============================================================================

/// Find the minimum and maximum height values in a heightmap
///
/// # Arguments
/// * `heightmap` - The 2D height array to analyze
///
/// # Returns
/// * `(f32, f32)` - Tuple of (min_height, max_height)
pub fn find_height_range(heightmap: &Array2D) -> (f32, f32) {
    let mut min = f32::MAX;
    let mut max = f32::MIN;

    for y in 0..heightmap.height() {
        for x in 0..heightmap.width() {
            if let Some(h) = heightmap.get(x, y) {
                min = min.min(h);
                max = max.max(h);
            }
        }
    }

    (min, max)
}

/// Calculate terrain slope at a specific point in the heightmap
///
/// Slope is computed from the height gradients using central differences.
/// Returns a value from 0.0 (flat) to 1.0 (very steep).
///
/// # Arguments
/// * `heightmap` - The 2D height array
/// * `x` - X coordinate in the heightmap
/// * `y` - Y coordinate in the heightmap
///
/// # Returns
/// * `f32` - Slope magnitude (0.0 = flat, 1.0 = very steep)
pub fn calculate_slope(heightmap: &Array2D, x: usize, y: usize) -> f32 {
    let width = heightmap.width();
    let height = heightmap.height();

    // Get center height
    let h_center = heightmap.get(x, y).unwrap_or(0.0);

    // Get neighboring heights with bounds checking
    let h_left = if x > 0 {
        heightmap.get(x - 1, y).unwrap_or(h_center)
    } else {
        h_center
    };

    let h_right = if x < width - 1 {
        heightmap.get(x + 1, y).unwrap_or(h_center)
    } else {
        h_center
    };

    let h_up = if y > 0 {
        heightmap.get(x, y - 1).unwrap_or(h_center)
    } else {
        h_center
    };

    let h_down = if y < height - 1 {
        heightmap.get(x, y + 1).unwrap_or(h_center)
    } else {
        h_center
    };

    // Calculate gradients using central difference
    let dx = (h_right - h_left) / 2.0;
    let dy = (h_down - h_up) / 2.0;

    // Slope magnitude
    let slope = (dx * dx + dy * dy).sqrt();

    // Normalize to 0-1 range (clamp very steep slopes)
    (slope * 2.0).min(1.0)
}

/// Sample multi-octave position-based noise
///
/// This generates noise based on world coordinates (not texture coordinates)
/// to ensure spatial coherence across the terrain.
///
/// # Arguments
/// * `world_x` - X coordinate in world space
/// * `world_y` - Y coordinate in world space
/// * `noise_scale` - Scale factor for noise frequency (lower = larger features)
/// * `seed` - Random seed for noise generation
///
/// # Returns
/// * `f32` - Noise value normalized to 0.0-1.0 range
pub fn sample_position_noise(world_x: f32, world_y: f32, noise_scale: f32, seed: u32) -> f32 {
    let perlin = Perlin::new(seed);

    // Sample at world coordinates
    let x = (world_x as f64) * (noise_scale as f64);
    let y = (world_y as f64) * (noise_scale as f64);

    // Multi-octave noise (4 octaves)
    let mut value = 0.0;
    let mut amplitude = 1.0;
    let mut frequency = 1.0;

    for _ in 0..4 {
        value += perlin.get([x * frequency, y * frequency]) * amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }

    // Normalize to 0-1 range
    ((value + 1.0) / 2.0) as f32
}

/// Sample high-frequency detail noise
///
/// # Arguments
/// * `world_x` - X coordinate in world space
/// * `world_y` - Y coordinate in world space
/// * `seed` - Random seed for noise generation
///
/// # Returns
/// * `f32` - Detail noise value normalized to 0.0-1.0 range
pub fn sample_detail_noise(world_x: f32, world_y: f32, seed: u32) -> f32 {
    // Use higher frequency (0.5) for fine detail
    sample_position_noise(world_x, world_y, 0.5, seed)
}

// ============================================================================
// Material Color Helper Functions
// ============================================================================

/// Blend two RGBA colors based on a factor
///
/// # Arguments
/// * `c1` - First color
/// * `c2` - Second color
/// * `factor` - Blend factor (0.0 = all c1, 1.0 = all c2)
///
/// # Returns
/// * `Rgba<u8>` - Blended color
fn blend_colors_rgba(c1: &Rgba<u8>, c2: &Rgba<u8>, factor: f32) -> Rgba<u8> {
    let factor = factor.clamp(0.0, 1.0);
    Rgba([
        (c1[0] as f32 * (1.0 - factor) + c2[0] as f32 * factor) as u8,
        (c1[1] as f32 * (1.0 - factor) + c2[1] as f32 * factor) as u8,
        (c1[2] as f32 * (1.0 - factor) + c2[2] as f32 * factor) as u8,
        255,
    ])
}

/// Get color for valley/low elevation areas
///
/// # Arguments
/// * `config` - Biome configuration
/// * `noise` - Noise value for variation (0.0-1.0)
///
/// # Returns
/// * `Rgba<u8>` - Valley color
fn get_valley_color(config: &BiomeConfig, noise: f32) -> Rgba<u8> {
    match config.base_material {
        BaseMaterialType::Grass => {
            // Green with variation
            let base_green = Rgba([45, 100, 45, 255]);
            let dark_green = Rgba([30, 80, 30, 255]);
            blend_colors_rgba(&base_green, &dark_green, noise)
        }
        BaseMaterialType::Alpine => {
            // Sparse greenish-brown
            let alpine_green = Rgba([60, 90, 50, 255]);
            let alpine_brown = Rgba([80, 70, 50, 255]);
            blend_colors_rgba(&alpine_green, &alpine_brown, noise)
        }
        BaseMaterialType::Desert => {
            // Sandy browns
            let sand = Rgba([194, 154, 97, 255]);
            let dark_sand = Rgba([163, 123, 77, 255]);
            blend_colors_rgba(&sand, &dark_sand, noise)
        }
        BaseMaterialType::Volcanic => {
            // Dark volcanic soil
            let dark_ash = Rgba([50, 45, 40, 255]);
            let red_soil = Rgba([80, 50, 40, 255]);
            blend_colors_rgba(&dark_ash, &red_soil, noise)
        }
        BaseMaterialType::Tundra => {
            // Mossy greens and browns
            let moss = Rgba([70, 85, 60, 255]);
            let brown = Rgba([90, 80, 60, 255]);
            blend_colors_rgba(&moss, &brown, noise)
        }
    }
}

/// Get color for rock formations
///
/// # Arguments
/// * `config` - Biome configuration
/// * `noise` - Noise value for variation (0.0-1.0)
///
/// # Returns
/// * `Rgba<u8>` - Rock color
fn get_rock_color(config: &BiomeConfig, noise: f32) -> Rgba<u8> {
    match config.rock_type {
        RockType::Granite => {
            let grey = Rgba([128, 128, 120, 255]);
            let dark_grey = Rgba([90, 90, 85, 255]);
            blend_colors_rgba(&grey, &dark_grey, noise)
        }
        RockType::Sandstone => {
            let red_rock = Rgba([180, 100, 70, 255]);
            let tan_rock = Rgba([160, 120, 90, 255]);
            blend_colors_rgba(&red_rock, &tan_rock, noise)
        }
        RockType::Basalt => {
            let dark_basalt = Rgba([60, 60, 65, 255]);
            let med_basalt = Rgba([80, 80, 85, 255]);
            blend_colors_rgba(&dark_basalt, &med_basalt, noise)
        }
        RockType::Limestone => {
            let light_lime = Rgba([200, 195, 180, 255]);
            let med_lime = Rgba([180, 175, 160, 255]);
            blend_colors_rgba(&light_lime, &med_lime, noise)
        }
    }
}

/// Get color for snow/ice
///
/// # Arguments
/// * `_config` - Biome configuration (unused but kept for consistency)
/// * `noise` - Noise value for variation (0.0-1.0)
///
/// # Returns
/// * `Rgba<u8>` - Snow color
fn get_snow_color(_config: &BiomeConfig, noise: f32) -> Rgba<u8> {
    let white = Rgba([250, 250, 255, 255]);
    let light_blue = Rgba([230, 235, 245, 255]);
    blend_colors_rgba(&white, &light_blue, noise)
}

/// Blend valley and rock colors based on height and slope
///
/// # Arguments
/// * `config` - Biome configuration
/// * `height` - Normalized height (0.0-1.0)
/// * `slope` - Slope value (0.0-1.0)
/// * `noise` - Noise value for variation (0.0-1.0)
///
/// # Returns
/// * `Rgba<u8>` - Blended color
fn blend_valley_rock(
    config: &BiomeConfig,
    height: f32,
    slope: f32,
    noise: f32,
) -> Rgba<u8> {
    let valley_color = get_valley_color(config, noise);
    let rock_color = get_rock_color(config, noise);

    // Blend based on slope (steeper = more rock)
    // Use quadratic for smoother transition
    let rock_factor = slope.powf(2.0);

    // Also blend based on height within the mid-zone
    // Higher elevations in mid-zone get more rock
    let height_factor = (height - 0.25) / 0.4; // Map 0.25-0.65 to 0-1
    let combined_factor = (rock_factor * 0.7 + height_factor * 0.3).clamp(0.0, 1.0);

    blend_colors_rgba(&valley_color, &rock_color, combined_factor)
}

/// Add color variation based on detail noise
///
/// # Arguments
/// * `base_color` - Base color to add variation to
/// * `detail_noise` - Detail noise value (0.0-1.0)
///
/// # Returns
/// * `Rgba<u8>` - Color with variation added
fn add_color_variation(base_color: Rgba<u8>, detail_noise: f32) -> Rgba<u8> {
    // Map noise from 0-1 to -0.1 to +0.1 for subtle variation
    let variation = (detail_noise - 0.5) * 0.2;

    Rgba([
        ((base_color[0] as f32 * (1.0 + variation)).clamp(0.0, 255.0)) as u8,
        ((base_color[1] as f32 * (1.0 + variation)).clamp(0.0, 255.0)) as u8,
        ((base_color[2] as f32 * (1.0 + variation)).clamp(0.0, 255.0)) as u8,
        255,
    ])
}

/// Determine material color based on height, slope, and biome
///
/// # Arguments
/// * `height_normalized` - Normalized height (0.0 = lowest, 1.0 = highest)
/// * `slope` - Slope value (0.0 = flat, 1.0 = vertical)
/// * `biome` - Biome type
/// * `noise` - Noise value for variation (0.0-1.0)
///
/// # Returns
/// * `Rgba<u8>` - Material color
fn determine_material_color(
    height_normalized: f32,
    slope: f32,
    biome: BiomeType,
    noise: f32,
) -> Rgba<u8> {
    let config = BiomeConfig::from_biome_type(biome);

    // Define elevation bands with smooth transitions
    if height_normalized < 0.25 {
        // Valley zone (0.0-0.25): Base material
        get_valley_color(&config, noise)
    } else if height_normalized < 0.65 {
        // Mid zone (0.25-0.65): Mix of base and rock
        // Use slope to determine rock vs vegetation
        if slope > 0.6 {
            // Steep slope = more rock
            get_rock_color(&config, noise)
        } else {
            // Gentle slope = blend
            blend_valley_rock(&config, height_normalized, slope, noise)
        }
    } else if height_normalized > 0.8 {
        // High peaks (0.8-1.0): Snow
        get_snow_color(&config, noise)
    } else {
        // Upper slopes (0.65-0.8): Rock
        get_rock_color(&config, noise)
    }
}

// ============================================================================
// Terrain-Aware Texture Generation
// ============================================================================

/// Generate unique texture mapped to terrain heightmap
///
/// This function creates a procedurally generated texture that is uniquely
/// mapped to the terrain mesh without tiling. Each part of the mountain
/// gets a unique color based on its height, slope, and position.
///
/// # Arguments
/// * `heightmap` - The 2D height array representing the terrain
/// * `texture_width` - Width of the generated texture in pixels
/// * `texture_height` - Height of the generated texture in pixels
/// * `biome` - Biome type to determine material colors
/// * `scale_x` - World-space X scale factor
/// * `scale_y` - World-space Y scale factor
/// * `scale_z` - World-space Z scale factor (height)
/// * `noise_scale` - Scale for position-based noise (lower = larger features)
/// * `seed` - Random seed for noise generation
///
/// # Returns
/// * `RgbaImage` - Generated texture image
///
/// # Example
/// ```ignore
/// let texture = generate_terrain_texture(
///     &heightmap,
///     1024,
///     1024,
///     BiomeType::Alpine,
///     5.0,
///     5.0,
///     200.0,
///     0.01,
///     42,
/// );
/// ```
pub fn generate_terrain_texture(
    heightmap: &Array2D,
    texture_width: u32,
    texture_height: u32,
    biome: BiomeType,
    scale_x: f32,
    scale_y: f32,
    scale_z: f32,
    noise_scale: f32,
    seed: u32,
) -> RgbaImage {
    // Analyze heightmap for normalization
    let (min_height, max_height) = find_height_range(heightmap);
    let height_range = max_height - min_height;

    let mut texture = RgbaImage::new(texture_width, texture_height);

    // For each texture pixel
    for tex_y in 0..texture_height {
        for tex_x in 0..texture_width {
            // Map texture coords back to heightmap position
            let u = tex_x as f32 / (texture_width as f32 - 1.0);
            let v = tex_y as f32 / (texture_height as f32 - 1.0);

            let grid_x = (u * (heightmap.width() as f32 - 1.0)) as usize;
            let grid_y = (v * (heightmap.height() as f32 - 1.0)) as usize;

            // Sample terrain data at this position
            let height = heightmap.get(grid_x, grid_y).unwrap_or(0.0);
            let height_normalized = if height_range > 0.0 {
                (height - min_height) / height_range
            } else {
                0.0
            };

            // Calculate slope at this position
            let slope = calculate_slope(heightmap, grid_x, grid_y);

            // Get world position for noise sampling
            let world_x = (grid_x as f32 - (heightmap.width() as f32 / 2.0)) * scale_x;
            let world_y = (grid_y as f32 - (heightmap.height() as f32 / 2.0)) * scale_y;

            // Sample position-based noise (at world scale, not texture scale)
            let noise_value = sample_position_noise(world_x, world_y, noise_scale, seed);

            // Determine material color based on height and slope
            let base_color = determine_material_color(
                height_normalized,
                slope,
                biome,
                noise_value,
            );

            // Add fine detail variation
            let detail_noise = sample_detail_noise(world_x, world_y, seed + 1000);
            let final_color = add_color_variation(base_color, detail_noise);

            texture.put_pixel(tex_x, tex_y, final_color);
        }
    }

    texture
}

/// Generate terrain-aware normal map from heightmap
///
/// Creates a normal map that represents the actual terrain geometry
/// with detail enhancement based on noise.
///
/// # Arguments
/// * `heightmap` - The 2D height array representing the terrain
/// * `texture_width` - Width of the generated normal map in pixels
/// * `texture_height` - Height of the generated normal map in pixels
/// * `strength` - Normal map strength multiplier (higher = more pronounced bumps)
/// * `scale_x` - World-space X scale factor
/// * `scale_y` - World-space Y scale factor
/// * `seed` - Random seed for detail noise
///
/// # Returns
/// * `RgbaImage` - Generated normal map
pub fn generate_terrain_normal_map(
    heightmap: &Array2D,
    texture_width: u32,
    texture_height: u32,
    strength: f32,
    scale_x: f32,
    scale_y: f32,
    seed: u32,
) -> RgbaImage {
    let mut normal_map = RgbaImage::new(texture_width, texture_height);

    for tex_y in 0..texture_height {
        for tex_x in 0..texture_width {
            // Map to heightmap position
            let u = tex_x as f32 / (texture_width as f32 - 1.0);
            let v = tex_y as f32 / (texture_height as f32 - 1.0);

            let grid_x = (u * (heightmap.width() as f32 - 1.0)) as usize;
            let grid_y = (v * (heightmap.height() as f32 - 1.0)) as usize;

            // Calculate normal from heightmap with bounds checking
            let width = heightmap.width();
            let height = heightmap.height();

            let h_center = heightmap.get(grid_x, grid_y).unwrap_or(0.0);

            let h_left = if grid_x > 0 {
                heightmap.get(grid_x - 1, grid_y).unwrap_or(h_center)
            } else {
                h_center
            };

            let h_right = if grid_x < width - 1 {
                heightmap.get(grid_x + 1, grid_y).unwrap_or(h_center)
            } else {
                h_center
            };

            let h_up = if grid_y > 0 {
                heightmap.get(grid_x, grid_y - 1).unwrap_or(h_center)
            } else {
                h_center
            };

            let h_down = if grid_y < height - 1 {
                heightmap.get(grid_x, grid_y + 1).unwrap_or(h_center)
            } else {
                h_center
            };

            // Calculate gradients (scaled by strength)
            let dx = (h_right - h_left) * strength;
            let dy = (h_down - h_up) * strength;

            // Add fine detail from noise
            let world_x = (grid_x as f32 - (width as f32 / 2.0)) * scale_x;
            let world_y = (grid_y as f32 - (height as f32 / 2.0)) * scale_y;
            let detail = sample_detail_noise(world_x, world_y, seed) - 0.5;
            let detail_strength = 0.1;

            // Normal vector (tangent-space)
            let norm_x = -dx + detail * detail_strength;
            let norm_y = -dy + detail * detail_strength;
            let norm_z = 1.0;

            // Normalize
            let len = (norm_x * norm_x + norm_y * norm_y + norm_z * norm_z).sqrt();
            let (norm_x, norm_y, norm_z) = if len > 0.0 {
                (norm_x / len, norm_y / len, norm_z / len)
            } else {
                (0.0, 0.0, 1.0)
            };

            // Encode to RGB (0.5, 0.5, 1.0) = flat surface
            let r = (((norm_x + 1.0) / 2.0) * 255.0) as u8;
            let g = (((norm_y + 1.0) / 2.0) * 255.0) as u8;
            let b = (((norm_z + 1.0) / 2.0) * 255.0) as u8;

            normal_map.put_pixel(tex_x, tex_y, Rgba([r, g, b, 255]));
        }
    }

    normal_map
}

/// Generate terrain-aware roughness map
///
/// Creates a roughness map that varies based on terrain material
/// (vegetation, rock, snow) determined by height and slope.
///
/// # Arguments
/// * `heightmap` - The 2D height array representing the terrain
/// * `texture_width` - Width of the generated roughness map in pixels
/// * `texture_height` - Height of the generated roughness map in pixels
/// * `biome` - Biome type to determine material roughness values
/// * `scale_x` - World-space X scale factor
/// * `scale_y` - World-space Y scale factor
/// * `seed` - Random seed for roughness variation
///
/// # Returns
/// * `RgbaImage` - Generated roughness map (greyscale in RGB channels)
pub fn generate_terrain_roughness_map(
    heightmap: &Array2D,
    texture_width: u32,
    texture_height: u32,
    biome: BiomeType,
    scale_x: f32,
    scale_y: f32,
    seed: u32,
) -> RgbaImage {
    let (min_height, max_height) = find_height_range(heightmap);
    let height_range = max_height - min_height;
    let config = BiomeConfig::from_biome_type(biome);

    let mut roughness_map = RgbaImage::new(texture_width, texture_height);

    for tex_y in 0..texture_height {
        for tex_x in 0..texture_width {
            let u = tex_x as f32 / (texture_width as f32 - 1.0);
            let v = tex_y as f32 / (texture_height as f32 - 1.0);

            let grid_x = (u * (heightmap.width() as f32 - 1.0)) as usize;
            let grid_y = (v * (heightmap.height() as f32 - 1.0)) as usize;

            let height = heightmap.get(grid_x, grid_y).unwrap_or(0.0);
            let height_normalized = if height_range > 0.0 {
                (height - min_height) / height_range
            } else {
                0.0
            };

            let slope = calculate_slope(heightmap, grid_x, grid_y);

            // Determine roughness based on material
            let base_roughness = if height_normalized > 0.8 {
                // Snow/ice = low roughness (smooth, reflective)
                config.snow_roughness
            } else if slope > 0.6 || height_normalized > 0.5 {
                // Rock = high roughness
                config.rock_roughness
            } else {
                // Vegetation = medium roughness
                0.7
            };

            // Add variation from noise
            let world_x = (grid_x as f32 - (heightmap.width() as f32 / 2.0)) * scale_x;
            let world_y = (grid_y as f32 - (heightmap.height() as f32 / 2.0)) * scale_y;
            let noise = sample_position_noise(world_x, world_y, 0.05, seed);
            let roughness_varied = (base_roughness + (noise - 0.5) * 0.2).clamp(0.0, 1.0);

            // Encode as greyscale
            let grey = (roughness_varied * 255.0) as u8;
            roughness_map.put_pixel(tex_x, tex_y, Rgba([grey, grey, grey, 255]));
        }
    }

    roughness_map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_rock_texture() {
        let texture = generate_rock_texture(64);
        assert_eq!(texture.width(), 64);
        assert_eq!(texture.height(), 64);

        // Check that we have some variation (not all pixels the same)
        let first_pixel = texture.get_pixel(0, 0);
        let mut has_variation = false;
        for y in 0..64 {
            for x in 0..64 {
                if texture.get_pixel(x, y) != first_pixel {
                    has_variation = true;
                    break;
                }
            }
            if has_variation {
                break;
            }
        }
        assert!(has_variation, "Texture should have color variation");
    }

    #[test]
    fn test_generate_grass_texture() {
        let texture = generate_grass_texture(64);
        assert_eq!(texture.width(), 64);
        assert_eq!(texture.height(), 64);

        // Check that grass has green tones
        let pixel = texture.get_pixel(32, 32);
        assert!(pixel[1] > pixel[0], "Green channel should be higher than red");
        assert!(pixel[1] > pixel[2], "Green channel should be higher than blue");
    }

    #[test]
    fn test_generate_snow_texture() {
        let texture = generate_snow_texture(64);
        assert_eq!(texture.width(), 64);
        assert_eq!(texture.height(), 64);

        // Check that snow is generally bright
        let mut total_brightness = 0u32;
        for y in 0..64 {
            for x in 0..64 {
                let pixel = texture.get_pixel(x, y);
                total_brightness += pixel[0] as u32 + pixel[1] as u32 + pixel[2] as u32;
            }
        }
        let avg_brightness = total_brightness / (64 * 64 * 3);
        assert!(avg_brightness > 200, "Snow texture should be bright, got {}", avg_brightness);
    }

    #[test]
    fn test_generate_uv_test_texture() {
        let texture = generate_uv_test_texture(64);
        assert_eq!(texture.width(), 64);
        assert_eq!(texture.height(), 64);

        // Check that we have different colored quadrants
        let top_left = texture.get_pixel(16, 16);
        let top_right = texture.get_pixel(48, 16);
        let bottom_left = texture.get_pixel(16, 48);
        let bottom_right = texture.get_pixel(48, 48);

        // All quadrants should be different
        assert_ne!(top_left, top_right);
        assert_ne!(top_left, bottom_left);
        assert_ne!(top_left, bottom_right);
    }

    #[test]
    fn test_blend_colors() {
        let color1 = [0.0, 0.0, 0.0];
        let color2 = [255.0, 255.0, 255.0];

        let result = blend_colors(&color1, &color2, 0.5);
        assert_eq!(result[0], 127.5);
        assert_eq!(result[1], 127.5);
        assert_eq!(result[2], 127.5);

        let result = blend_colors(&color1, &color2, 0.0);
        assert_eq!(result[0], 0.0);

        let result = blend_colors(&color1, &color2, 1.0);
        assert_eq!(result[0], 255.0);
    }

    #[test]
    fn test_torus_coords() {
        let (nx1, ny1) = torus_coords(0, 0, 100);
        let (nx2, ny2) = torus_coords(100, 100, 100);

        // Coordinates should be finite
        assert!(nx1.is_finite());
        assert!(ny1.is_finite());
        assert!(nx2.is_finite());
        assert!(ny2.is_finite());

        // Opposite corners should have different but related coordinates
        // (torus topology makes them wrap around)
        assert!((nx1 - nx2).abs() < 3.0);
        assert!((ny1 - ny2).abs() < 3.0);
    }

    #[test]
    fn test_encode_tangent_normal_flat() {
        // Test flat normal (0, 0, 1)
        let flat = encode_tangent_normal(0.0, 0.0, 1.0);
        assert_eq!(flat[0], 128, "R channel should be 128 (neutral X)");
        assert_eq!(flat[1], 128, "G channel should be 128 (neutral Y)");
        assert_eq!(flat[2], 255, "B channel should be 255 (up Z)");
        assert_eq!(flat[3], 255, "Alpha should be 255");
    }

    #[test]
    fn test_encode_tangent_normal_directions() {
        // Test left-leaning normal (-1, 0, 0)
        let left = encode_tangent_normal(-1.0, 0.0, 0.0);
        assert_eq!(left[0], 0, "R should be 0 for -1 X");
        assert_eq!(left[1], 128, "G should be 128 for 0 Y");
        assert_eq!(left[2], 128, "B should be 128 for 0 Z");

        // Test right-leaning normal (1, 0, 0)
        let right = encode_tangent_normal(1.0, 0.0, 0.0);
        assert_eq!(right[0], 255, "R should be 255 for +1 X");
        assert_eq!(right[1], 128, "G should be 128 for 0 Y");
        assert_eq!(right[2], 128, "B should be 128 for 0 Z");
    }

    #[test]
    fn test_generate_flat_normal_map() {
        let normal_map = generate_flat_normal_map(64);
        assert_eq!(normal_map.width(), 64);
        assert_eq!(normal_map.height(), 64);

        // All pixels should be (128, 128, 255, 255) for flat surface
        for y in 0..64 {
            for x in 0..64 {
                let pixel = normal_map.get_pixel(x, y);
                assert_eq!(pixel[0], 128, "R channel should be 128 (neutral X)");
                assert_eq!(pixel[1], 128, "G channel should be 128 (neutral Y)");
                assert_eq!(pixel[2], 255, "B channel should be 255 (up Z)");
                assert_eq!(pixel[3], 255, "Alpha should be 255");
            }
        }
    }

    #[test]
    fn test_generate_rock_normal_map() {
        let normal_map = generate_rock_normal_map(64, 1.0);
        assert_eq!(normal_map.width(), 64);
        assert_eq!(normal_map.height(), 64);

        // Check that we have variation (not all pixels flat)
        let mut has_variation = false;
        for y in 0..64 {
            for x in 0..64 {
                let pixel = normal_map.get_pixel(x, y);
                // If any pixel differs from flat normal, we have variation
                if pixel[0] != 128 || pixel[1] != 128 || pixel[2] != 255 {
                    has_variation = true;
                    break;
                }
            }
            if has_variation {
                break;
            }
        }
        assert!(has_variation, "Normal map should have surface variation");

        // Verify all normals are generally upward-facing (blue channel dominant)
        for y in 0..64 {
            for x in 0..64 {
                let pixel = normal_map.get_pixel(x, y);
                // Blue channel should be > 128 for upward-facing normals
                assert!(pixel[2] > 100, "Z component should be positive (pixel at {},{} has B={})", x, y, pixel[2]);
            }
        }
    }

    #[test]
    fn test_generate_grass_normal_map() {
        let normal_map = generate_grass_normal_map(64, 1.0);
        assert_eq!(normal_map.width(), 64);
        assert_eq!(normal_map.height(), 64);

        // Check that we have variation
        let mut has_variation = false;
        for y in 0..64 {
            for x in 0..64 {
                let pixel = normal_map.get_pixel(x, y);
                if pixel[0] != 128 || pixel[1] != 128 || pixel[2] != 255 {
                    has_variation = true;
                    break;
                }
            }
            if has_variation {
                break;
            }
        }
        assert!(has_variation, "Grass normal map should have surface variation");
    }

    #[test]
    fn test_generate_snow_normal_map() {
        let normal_map = generate_snow_normal_map(64, 1.0);
        assert_eq!(normal_map.width(), 64);
        assert_eq!(normal_map.height(), 64);

        // Snow should have subtle variation (smoother than rock)
        // All normals should be mostly upward-facing
        let mut upward_count = 0;
        for y in 0..64 {
            for x in 0..64 {
                let pixel = normal_map.get_pixel(x, y);
                // Blue channel should be dominant for upward-facing normals
                if pixel[2] > 200 {
                    upward_count += 1;
                }
            }
        }
        // Most pixels should be strongly upward-facing for snow (smoother surface)
        assert!(upward_count > (64 * 64) * 3 / 4,
            "Snow normal map should be relatively smooth ({}% upward)",
            (upward_count * 100) / (64 * 64));
    }

    #[test]
    fn test_sample_height_noise() {
        let perlin = Perlin::new(42);
        let value = sample_height_noise(&perlin, 0.5, 0.5, 0.0, 4, 1.0, 1.0);

        // Should return a finite value
        assert!(value.is_finite(), "Noise value should be finite");

        // Multi-octave noise should be roughly in range [-2, 2] for 4 octaves
        // (sum of amplitudes: 1.0 + 0.5 + 0.25 + 0.125 = 1.875)
        assert!(value.abs() < 3.0, "Multi-octave noise should be reasonable, got {}", value);
    }
}
