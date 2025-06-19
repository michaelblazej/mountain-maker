use crate::dla_2d::Array2D;

/// Module for blurring and interpolating grid data
pub struct BlurOptions {
    /// Strength of the blur (0.0 to 1.0)
    pub strength: f32,
}

impl Default for BlurOptions {
    fn default() -> Self {
        BlurOptions {
            strength: 0.5,
        }
    }
}

/// Linearly interpolate between two values
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

/// Upsample a grid using linear interpolation
/// 
/// * `grid` - The source grid to interpolate from
/// * `scale` - The scale factor for upsampling (e.g., 2 will double the dimensions)
/// * `options` - Optional blur settings
/// 
/// Returns a new grid with dimensions multiplied by scale and interpolated values
pub fn upsample(grid: &Array2D, scale: usize, options: Option<BlurOptions>) -> Array2D {
    // Use default options if none provided
    let options = options.unwrap_or_default();
    
    let src_width = grid.width();
    let src_height = grid.height();
    
    // Calculate new dimensions
    let dst_width = src_width * scale;
    let dst_height = src_height * scale;
    
    // Create target grid
    let mut result = Array2D::new(dst_width, dst_height, 0);
    
    // For each pixel in the destination grid
    for y in 0..dst_height {
        for x in 0..dst_width {
            // Map back to source coordinates
            let src_x = x as f32 / scale as f32;
            let src_y = y as f32 / scale as f32;
            
            // Get the four surrounding points in source grid
            let x0 = src_x.floor() as usize;
            let y0 = src_y.floor() as usize;
            let x1 = (x0 + 1).min(src_width - 1);
            let y1 = (y0 + 1).min(src_height - 1);
            
            // Calculate interpolation factors
            let tx = (src_x - x0 as f32) * options.strength;
            let ty = (src_y - y0 as f32) * options.strength;
            
            // Get source values, defaulting to 0 for out of bounds
            let v00 = grid.get(x0, y0).unwrap_or(0) as f32;
            let v01 = grid.get(x0, y1).unwrap_or(0) as f32;
            let v10 = grid.get(x1, y0).unwrap_or(0) as f32;
            let v11 = grid.get(x1, y1).unwrap_or(0) as f32;
            
            // Bilinear interpolation
            let top = lerp(v00, v10, tx);
            let bottom = lerp(v01, v11, tx);
            let value = lerp(top, bottom, ty);
            
            // Round and convert back to i32
            let value = if value >= 0.5 { 1 } else { 0 };
            result.set(x, y, value);
        }
    }
    
    result
}

/// Apply a simple box blur to the grid
/// 
/// * `grid` - The grid to blur
/// * `radius` - Blur radius
/// 
/// Returns a new blurred grid
pub fn box_blur(grid: &Array2D, radius: usize) -> Array2D {
    let width = grid.width();
    let height = grid.height();
    
    // Create result grid
    let mut result = Array2D::new(width, height, 0);
    
    // For each pixel in the grid
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0;
            let mut count = 0;
            
            // Sample the surrounding pixels within radius
            for dy in -(radius as isize)..=(radius as isize) {
                for dx in -(radius as isize)..=(radius as isize) {
                    let nx = x as isize + dx;
                    let ny = y as isize + dy;
                    
                    // Skip if out of bounds
                    if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
                        continue;
                    }
                    
                    if let Some(value) = grid.get(nx as usize, ny as usize) {
                        sum += value;
                        count += 1;
                    }
                }
            }
            
            // Calculate the average
            let avg = if count > 0 {
                sum as f32 / count as f32
            } else {
                0.0
            };
            
            // Set the result
            let value = if avg >= 0.5 { 1 } else { 0 };
            result.set(x, y, value);
        }
    }
    
    result
}

/// Upsample and then blur a grid in one operation
///
/// * `grid` - The source grid
/// * `scale` - Upsampling factor
/// * `blur_radius` - Radius for blur after upsampling
/// 
/// Returns a new grid that is upsampled and blurred
pub fn upsample_and_blur(grid: &Array2D, scale: usize, blur_radius: usize) -> Array2D {
    let upsampled = upsample(grid, scale, None);
    box_blur(&upsampled, blur_radius)
}
