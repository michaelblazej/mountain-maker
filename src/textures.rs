/// Procedural texture generation module for Mountain Maker
///
/// This module provides functions to generate tileable, realistic textures
/// and tangent-space normal maps for terrain rendering using multi-octave noise functions.

use image::{RgbaImage, Rgba};
use noise::{NoiseFn, Perlin};

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
