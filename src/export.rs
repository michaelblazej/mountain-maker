use crate::dla_2d::Array2D;
use anyhow::Result;
use mesh_tools::GltfBuilder;
use mesh_tools::Triangle;
use mesh_tools::compat::{Point3, Vector2, Vector3};

/// Exports the given Array2D as a 3D mesh in GLB format
/// 
/// # Arguments
/// * `array` - The 2D array containing height values
/// * `scale_x` - X-axis scale factor (horizontal grid spacing)
/// * `scale_y` - Y-axis scale factor (horizontal grid spacing) 
/// * `scale_z` - Z-axis scale factor (height multiplier)
/// * `output_path` - Path where the GLB file will be saved
/// 
/// # Returns
/// * `Result<()>` - Success or error
pub fn export_array_to_glb(
    array: &Array2D, 
    scale_x: f32,
    scale_y: f32, 
    scale_z: f32,
    output_path: &str
) -> Result<()> {
    // Create a new glTF builder
    let mut builder = GltfBuilder::new();
    
    // Get array dimensions
    let width = array.width();
    let height = array.height();
    
    // Calculate vertices, normals, UVs and indices
    let mut positions = Vec::with_capacity(width * height);
    let mut normals = Vec::with_capacity(width * height);
    let mut texcoords = Vec::with_capacity(width * height);
    let mut indices = Vec::with_capacity((width - 1) * (height - 1) * 2); // 2 triangles per grid cell
    
    // Generate vertex data
    for y in 0..height {
        for x in 0..width {
            // Get height value from the array
            let z_val = array.get(x, y).unwrap_or(0.0) * scale_z;
            
            // Calculate position in 3D space (centered around origin)
            let x_pos = (x as f32 - (width as f32 / 2.0)) * scale_x;
            let y_pos = (y as f32 - (height as f32 / 2.0)) * scale_y;
            
            // Add position using the mesh-tools Point3 type
            positions.push(mesh_tools::compat::point3::new(x_pos, z_val, y_pos));
            
            // Calculate normal
            let normal = calculate_normal(array, x, y, width, height, scale_x, scale_y, scale_z);
            normals.push(mesh_tools::compat::vector3::new(normal.0, normal.1, normal.2));
            
            // Add texture coordinates
            texcoords.push(mesh_tools::compat::vector2::new(
                x as f32 / (width as f32 - 1.0),
                y as f32 / (height as f32 - 1.0)
            ));
        }
    }
    
    // Generate triangle indices
    for y in 0..(height - 1) {
        for x in 0..(width - 1) {
            let top_left = (y * width + x) as u32;
            let top_right = (y * width + x + 1) as u32;
            let bottom_left = ((y + 1) * width + x) as u32;
            let bottom_right = ((y + 1) * width + x + 1) as u32;
            
            // First triangle of quad
            indices.push(Triangle::new(top_left, bottom_left, top_right));
            
            // Second triangle of quad
            indices.push(Triangle::new(top_right, bottom_left, bottom_right));
        }
    }
    
    // Create the mesh using the high-level API
    let mesh_index = builder.create_simple_mesh(
        Some("TerrainMesh".to_string()),
        &positions,
        &indices,
        Some(normals),
        Some(texcoords),
        None // No material
    );
    
    // Add a node referencing the mesh
    let node = builder.add_node(
        Some("Terrain".to_string()),
        Some(mesh_index),
        None, // Default position
        None, // Default rotation
        None, // Default scale
    );
    
    // Create a scene containing the node
    builder.add_scene(Some("Main Scene".to_string()), Some(vec![node]));
    
    // Export to GLB format
    builder.export_glb(output_path)?;
    
    Ok(())
}

/// Calculate surface normal at given point using central difference method
fn calculate_normal(
    array: &Array2D, 
    x: usize, 
    y: usize, 
    width: usize, 
    height: usize,
    scale_x: f32,
    scale_y: f32,
    scale_z: f32
) -> (f32, f32, f32) {
    let get_height = |x: isize, y: isize| -> f32 {
        if x >= 0 && y >= 0 && x < width as isize && y < height as isize {
            array.get(x as usize, y as usize).unwrap_or(0.0) * scale_z
        } else {
            0.0
        }
    };
    
    let x_i = x as isize;
    let y_i = y as isize;
    
    // Use central difference for calculating gradients
    let dx = (get_height(x_i + 1, y_i) - get_height(x_i - 1, y_i)) / (2.0 * scale_x);
    let dy = (get_height(x_i, y_i + 1) - get_height(x_i, y_i - 1)) / (2.0 * scale_y);
    
    // Cross product of tangent vectors gives us the normal
    let nx = -dx;
    let ny = -dy;
    let nz = 1.0;
    
    // Normalize the vector
    let length = (nx * nx + ny * ny + nz * nz).sqrt();
    if length > 0.0 {
        (nx / length, ny / length, nz / length)
    } else {
        (0.0, 0.0, 1.0) // Default to up vector if we can't calculate normal
    }
}

/// Exports the given Array2D as a 3D mesh in GLB format with default scale parameters
/// 
/// # Arguments
/// * `array` - The 2D array containing height values
/// * `output_path` - Path where the GLB file will be saved
/// 
/// # Returns
/// * `Result<()>` - Success or error
pub fn export_array_to_glb_default(array: &Array2D, output_path: &str) -> Result<()> {
    export_array_to_glb(array, 1.0, 1.0, 1.0, output_path)
}
