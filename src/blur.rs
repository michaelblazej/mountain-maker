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
    let mut result = Array2D::new(dst_width, dst_height, 0.0);
    
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
            
            // Get source values, defaulting to 0.0 for out of bounds
            let v00 = grid.get(x0, y0).unwrap_or(0.0);
            let v01 = grid.get(x0, y1).unwrap_or(0.0);
            let v10 = grid.get(x1, y0).unwrap_or(0.0);
            let v11 = grid.get(x1, y1).unwrap_or(0.0);
            
            // Bilinear interpolation
            let top = lerp(v00, v10, tx);
            let bottom = lerp(v01, v11, tx);
            let value = lerp(top, bottom, ty);
            
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
    let mut result = Array2D::new(width, height, 0.0);
    
    // For each pixel in the grid
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
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
            result.set(x, y, avg);
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

/// A 2D convolution kernel
#[allow(dead_code)]
pub struct Kernel {
    /// The kernel data as a flattened array
    data: Vec<f32>,
    /// Width of the kernel
    width: usize,
    /// Height of the kernel
    height: usize,
}

impl Kernel {
    /// Create a new kernel with the given data and dimensions
    #[allow(dead_code)]
    pub fn new(data: Vec<f32>, width: usize, height: usize) -> Self {
        assert_eq!(data.len(), width * height, "Kernel data size must match width * height");
        Kernel { data, width, height }
    }
    
    /// Create a mean filter kernel of the specified size
    #[allow(dead_code)]
    pub fn mean(size: usize) -> Self {
        let mut data = vec![1.0; size * size];
        let sum: f32 = data.iter().sum();
        
        // Normalize to ensure the sum of all weights is 1
        for val in data.iter_mut() {
            *val /= sum;
        }
        
        Kernel::new(data, size, size)
    }
    
    /// Create a Gaussian kernel of the specified size
    #[allow(dead_code)]
    pub fn gaussian(size: usize, sigma: f32) -> Self {
        assert!(size % 2 == 1, "Kernel size must be odd");
        
        let mut data = vec![0.0; size * size];
        let center = (size / 2) as f32;
        let mut sum = 0.0;
        
        // Generate Gaussian values
        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - center;
                let dy = y as f32 - center;
                let distance_squared = dx * dx + dy * dy;
                let exponent = -distance_squared / (2.0 * sigma * sigma);
                let value = (exponent.exp()) / (2.0 * std::f32::consts::PI * sigma * sigma);
                
                data[y * size + x] = value;
                sum += value;
            }
        }
        
        // Normalize
        for val in data.iter_mut() {
            *val /= sum;
        }
        
        Kernel::new(data, size, size)
    }
    
    /// Get the kernel value at the given position
    #[allow(dead_code)]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.width + x]
    }
}

/// Apply a convolution kernel to a grid
/// 
/// * `grid` - The source grid
/// * `kernel` - The kernel to apply
/// 
/// Returns a new grid after convolution
#[allow(dead_code)]
pub fn convolve(grid: &Array2D, kernel: &Kernel) -> Array2D {
    let width = grid.width();
    let height = grid.height();
    let kw = kernel.width;
    let kh = kernel.height;
    
    // Ensure kernel size is odd
    assert!(kw % 2 == 1 && kh % 2 == 1, "Kernel dimensions must be odd");
    
    // Calculate padding needed
    let pad_x = kw / 2;
    let pad_y = kh / 2;
    
    // Create result grid
    let mut result = Array2D::new(width, height, 0.0);
    
    // For each pixel in the grid
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            
            // Apply kernel
            for ky in 0..kh {
                for kx in 0..kw {
                    // Calculate input coordinates with respect to kernel center
                    let ix = x as isize + (kx as isize - pad_x as isize);
                    let iy = y as isize + (ky as isize - pad_y as isize);
                    
                    // Handle borders (zero padding)
                    if ix < 0 || iy < 0 || ix >= width as isize || iy >= height as isize {
                        continue;
                    }
                    
                    // Get the input value and kernel weight
                    if let Some(value) = grid.get(ix as usize, iy as usize) {
                        sum += value * kernel.get(kx, ky);
                    }
                }
            }
            
            // Convert to binary result
            let value = if sum >= 0.5 { 1.0 } else { 0.0 };
            result.set(x, y, value);
        }
    }
    
    result
}

/// Apply a mean filter convolution to a grid
/// 
/// * `grid` - The source grid
/// * `size` - The size of the filter kernel (should be odd)
/// 
/// Returns a new grid after mean filtering
#[allow(dead_code)]
pub fn mean_filter(grid: &Array2D, size: usize) -> Array2D {
    let kernel = Kernel::mean(size);
    convolve(grid, &kernel)
}

/// Apply a Gaussian filter convolution to a grid
/// 
/// * `grid` - The source grid
/// * `size` - The size of the filter kernel (should be odd)
/// * `sigma` - Standard deviation of the Gaussian function
/// 
/// Returns a new grid after Gaussian filtering
#[allow(dead_code)]
pub fn gaussian_filter(grid: &Array2D, size: usize, sigma: f32) -> Array2D {
    let kernel = Kernel::gaussian(size, sigma);
    convolve(grid, &kernel)
}
