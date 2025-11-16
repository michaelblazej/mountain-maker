use crate::dla_2d::Array2D;
use crate::textures::{
    generate_grass_texture,
    generate_uv_test_texture,
    generate_grass_normal_map,
    generate_flat_normal_map,
};
use anyhow::Result;
use mesh_tools::GltfBuilder;
use mesh_tools::Triangle;
use mesh_tools::material::MaterialBuilder;

/// Exports the given Array2D as a 3D mesh in GLB format
///
/// # Arguments
/// * `array` - The 2D array containing height values
/// * `scale_x` - X-axis scale factor (horizontal grid spacing)
/// * `scale_y` - Y-axis scale factor (horizontal grid spacing)
/// * `scale_z` - Z-axis scale factor (height multiplier)
/// * `output_path` - Path where the GLB file will be saved
/// * `world_uv_scale` - Texture scale in world units (lower = more repetition)
/// * `enable_textures` - Whether to generate and embed procedural textures
/// * `texture_resolution` - Resolution of procedural textures (512, 1024, 2048)
/// * `uv_test` - Generate UV test texture instead of realistic textures
/// * `enable_normal_maps` - Whether to generate tangent-space normal maps
/// * `normal_strength` - Normal map generation strength (0.5-2.0)
/// * `normal_scale` - glTF normal scale parameter (0.5-2.0, runtime intensity)
///
/// # Returns
/// * `Result<()>` - Success or error
pub fn export_array_to_glb(
    array: &Array2D,
    scale_x: f32,
    scale_y: f32,
    scale_z: f32,
    output_path: &str,
    world_uv_scale: f32,
    enable_textures: bool,
    texture_resolution: u32,
    uv_test: bool,
    enable_normal_maps: bool,
    normal_strength: f32,
    normal_scale: f32,
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

            // Add position using the mesh-tools Point3 type - height on z-axis
            positions.push(mesh_tools::compat::point3::new(x_pos, y_pos, z_val));

            // Calculate normal
            let normal = calculate_normal(array, x, y, width, height, scale_x, scale_y, scale_z);
            normals.push(mesh_tools::compat::vector3::new(normal.0, normal.1, normal.2));

            // Add texture coordinates using world-space UVs
            // This ensures consistent texture density regardless of grid resolution
            let u = x_pos * world_uv_scale;
            let v = y_pos * world_uv_scale;
            texcoords.push(mesh_tools::compat::vector2::new(u, v));
        }
    }

    // Note: Vertex colors are not currently supported by mesh-tools create_custom_mesh
    // Height data is stored in the texture coordinates' z-component in future versions
    // For now, we rely on the texture for visual appearance
    
    // Generate triangle indices
    for y in 0..(height - 1) {
        for x in 0..(width - 1) {
            let top_left = (y * width + x) as u32;
            let top_right = (y * width + x + 1) as u32;
            let bottom_left = ((y + 1) * width + x) as u32;
            let bottom_right = ((y + 1) * width + x + 1) as u32;
            
            // First triangle of quad (counter-clockwise winding for upward-facing normal)
            indices.push(Triangle::new(top_left, top_right, bottom_left));

            // Second triangle of quad (counter-clockwise winding for upward-facing normal)
            indices.push(Triangle::new(top_right, bottom_right, bottom_left));
        }
    }
    
    // Create a terrain material (with or without textures)
    let terrain_material = if enable_textures {
        println!("Generating procedural texture ({}x{})...", texture_resolution, texture_resolution);

        // Generate procedural color texture
        let color_img = if uv_test {
            println!("Using UV test texture");
            generate_uv_test_texture(texture_resolution)
        } else {
            // Use grass as primary texture for now
            // Future enhancement: blend multiple textures based on height
            println!("Using procedural grass texture");
            generate_grass_texture(texture_resolution)
        };

        // Convert RgbaImage to DynamicImage and create color texture
        let color_texture_idx = builder.create_texture_from_image(
            Some("BaseColorTexture".to_string()),
            &image::DynamicImage::ImageRgba8(color_img),
            mesh_tools::texture::TextureFormat::PNG,
        )?;

        // Create material with or without normal maps
        let material = if enable_normal_maps {
            println!("Generating procedural normal map ({}x{})...", texture_resolution, texture_resolution);

            // Generate normal map
            let normal_img = if uv_test {
                println!("Using flat normal map for UV test");
                generate_flat_normal_map(texture_resolution)
            } else {
                println!("Using procedural grass normal map (strength: {})", normal_strength);
                generate_grass_normal_map(texture_resolution, normal_strength)
            };

            // Create normal texture
            let normal_texture_idx = builder.create_texture_from_image(
                Some("NormalTexture".to_string()),
                &image::DynamicImage::ImageRgba8(normal_img),
                mesh_tools::texture::TextureFormat::PNG,
            )?;

            // Create PBR material with both color and normal textures
            MaterialBuilder::new(Some("TerrainMaterial".to_string()))
                .with_base_color_texture(color_texture_idx, None)
                .with_normal_texture(normal_texture_idx, None, Some(normal_scale))
                .with_roughness_factor(0.8)
                .with_metallic_factor(0.0)
                .build()
        } else {
            // Color texture only (backward compatible)
            MaterialBuilder::new(Some("TerrainMaterial".to_string()))
                .with_base_color_texture(color_texture_idx, None)
                .with_roughness_factor(0.8)
                .with_metallic_factor(0.0)
                .build()
        };

        // Add material directly to the gltf materials vector
        let material_index = if let Some(materials) = &mut builder.gltf.materials {
            materials.push(material);
            materials.len() - 1
        } else {
            builder.gltf.materials = Some(vec![material]);
            0
        };

        material_index
    } else {
        // Fallback to solid color (backward compatibility)
        builder.create_basic_material(
            Some("TerrainMaterial".to_string()),
            [0.5, 0.8, 0.3, 1.0],  // Green-ish color for terrain
        )
    };

    // Create the mesh using custom mesh API
    let mesh_index = builder.create_custom_mesh(
        Some("TerrainMesh".to_string()),
        &positions,
        &indices,
        Some(normals),
        Some(vec![texcoords]),  // UV channel 0 (can add more channels later)
        Some(terrain_material),
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
    let dzx = (get_height(x_i + 1, y_i) - get_height(x_i - 1, y_i)) / (2.0 * scale_x);
    let dzy = (get_height(x_i, y_i + 1) - get_height(x_i, y_i - 1)) / (2.0 * scale_y);

    let vec_x = vec![1.0, 0.0, dzx];
    let vec_y = vec![0.0, 1.0, dzy];
    
    // Cross product of tangent vectors gives us the normal
    // Manual cross product calculation: vec_x × vec_y
    // For vectors a = [a1, a2, a3] and b = [b1, b2, b3]:
    // cross = [a2*b3 - a3*b2, a3*b1 - a1*b3, a1*b2 - a2*b1]
    let nx = vec_x[1] * vec_y[2] - vec_x[2] * vec_y[1]; // 0.0 * dzy - dzx * 2.0
    let ny = vec_x[2] * vec_y[0] - vec_x[0] * vec_y[2]; // dzx * 0.0 - 2.0 * dzy
    let nz = vec_x[0] * vec_y[1] - vec_x[1] * vec_y[0]; // 2.0 * 2.0 - 0.0 * 0.0
    
    // Normalize the vector
    let length = (nx * nx + ny * ny + nz * nz).sqrt();
    if length > 0.0 {
        (nx / length, ny / length, nz / length)
    } else {
        (0.0, 0.0, 1.0) // Default to up vector if we can't calculate normal
    }
}

/// Exports the given Array2D as a 3D mesh in GLB format with default parameters
///
/// # Arguments
/// * `array` - The 2D array containing height values
/// * `output_path` - Path where the GLB file will be saved
///
/// # Returns
/// * `Result<()>` - Success or error
#[allow(dead_code)]
pub fn export_array_to_glb_default(array: &Array2D, output_path: &str) -> Result<()> {
    export_array_to_glb(array, 1.0, 1.0, 1.0, output_path, 0.1, false, 512, false, false, 1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_normal_flat_surface() {
        // Create a flat 5x5 grid with height 1.0
        let array = Array2D::new(5, 5, 1.0);
        
        // Calculate normal at center point (2, 2)
        let (nx, ny, nz) = calculate_normal(&array, 2, 2, 5, 5, 1.0, 1.0, 1.0);
        
        // For a flat surface, normal should point straight up (0, 0, 1)
        // Allow for small floating point errors
        assert!((nx - 0.0).abs() < 1e-6, "nx should be close to 0, got {}", nx);
        assert!((ny - 0.0).abs() < 1e-6, "ny should be close to 0, got {}", ny);
        assert!((nz - 1.0).abs() < 1e-6, "nz should be close to 1, got {}", nz);
    }

    #[test]
    fn test_calculate_normal_sloped_surface() {
        // Create a 5x5 grid with a slope in x-direction
        let mut array = Array2D::new(5, 5, 0.0);
        
        // Set heights to create a slope: height increases with x
        for y in 0..5 {
            for x in 0..5 {
                array.set(x, y, x as f32);
            }
        }
        
        // Calculate normal at center point (2, 2)
        let (nx, ny, nz) = calculate_normal(&array, 2, 2, 5, 5, 1.0, 1.0, 1.0);
        
        // For a slope in x-direction, nx should be negative, ny should be 0, nz positive
        // The normal should be normalized
        let length = (nx * nx + ny * ny + nz * nz).sqrt();
        let test_val = (2.0f32).sqrt()/2.0;
        assert!((length - 1.0).abs() < 1e-6, "Normal should be normalized, length = {}", length);
        assert!((nx + test_val).abs() < 1e-6, "nx should be close to {}, got {}", test_val, nx);
        assert!((ny - 0.0).abs() < 1e-6, "ny should be close to 0, got {}", ny);
        assert!((nz - test_val).abs() < 1e-6, "nz should be close to {}, got {}",test_val, nz);
    }

    #[test]
    fn test_calculate_normal_pyramid() {
        // Create a 5x5 grid with a pyramid shape (peak at center)
        let mut array = Array2D::new(5, 5, 0.0);
        
        // Set heights to create a pyramid: height decreases with distance from center
        for y in 0..5 {
            for x in 0..5 {
                let dist_from_center = ((x as f32 - 2.0).powi(2) + (y as f32 - 2.0).powi(2)).sqrt();
                array.set(x, y, 5.0 - dist_from_center);
            }
        }
        
        // Calculate normal at center point (2, 2)
        let (nx, ny, nz) = calculate_normal(&array, 2, 2, 5, 5, 1.0, 5.0, 1.0);
        
        // The normal should be normalized
        let length = (nx * nx + ny * ny + nz * nz).sqrt();
        assert!((length - 1.0).abs() < 1e-6, "Normal should be normalized, length = {}", length);
        assert!(nz > 0.0, "nz should be positive for a peak, got {}", nz);
    }

    #[test]
    fn test_calculate_normal_edge_handling() {
        // Create a 3x3 grid to test edge cases
        let array = Array2D::new(3, 3, 1.0);

        // Test corner point (0, 0)
        let (nx, ny, nz) = calculate_normal(&array, 0, 0, 3, 3, 1.0, 1.0, 1.0);

        // Should still return a valid normalized normal
        let length = (nx * nx + ny * ny + nz * nz).sqrt();
        assert!((length - 1.0).abs() < 1e-6, "Normal should be normalized, length = {}", length);
    }

    #[test]
    fn test_exported_normals_point_upward() {
        use std::fs;

        // Create a flat 10x10 grid with uniform height 1.0
        let array = Array2D::new(10, 10, 1.0);

        // Export to a temporary GLB file
        let temp_path = "/tmp/test_flat_normals.glb";
        let result = export_array_to_glb(
            &array,
            1.0,  // scale_x
            1.0,  // scale_y
            1.0,  // scale_z
            temp_path,
            0.1,  // world_uv_scale
            false, // enable_textures
            512,  // texture_resolution
            false, // uv_test
            false, // enable_normal_maps
            1.0,  // normal_strength
            1.0,  // normal_scale
        );

        assert!(result.is_ok(), "Failed to export GLB: {:?}", result.err());

        // Verify file was created
        assert!(std::path::Path::new(temp_path).exists(), "GLB file was not created");

        // Load the GLB file and verify normals
        let data = fs::read(temp_path).expect("Failed to read GLB file");
        let gltf = gltf::Gltf::from_slice(&data).expect("Failed to parse GLB");

        // Get the blob containing binary data
        let blob = gltf.blob.as_ref().expect("No binary blob in GLB");

        // Get the first mesh
        let mesh = gltf.meshes().next().expect("No mesh found in GLB");
        let primitive = mesh.primitives().next().expect("No primitives found");

        // Get the normals accessor
        let normals_accessor = primitive.get(&gltf::Semantic::Normals)
            .expect("No normals found in mesh");

        // Get the buffer view for normals
        let buffer_view = normals_accessor.view().expect("No buffer view for normals");
        let buffer_offset = buffer_view.offset() + normals_accessor.offset();

        // Read normal values (they should all point roughly upward for a flat surface)
        // Note: We skip edge vertices as they may have artifacts from boundary handling
        let mut upward_count = 0;
        let mut interior_count = 0;
        let width = 10;
        let height = 10;

        for i in 0..normals_accessor.count() {
            let x = i % width;
            let y = i / width;

            // Skip edge vertices - only check interior vertices
            if x == 0 || y == 0 || x == width - 1 || y == height - 1 {
                continue;
            }

            let offset = buffer_offset + i * 12; // 3 floats * 4 bytes each
            let nx = f32::from_le_bytes([
                blob[offset],
                blob[offset + 1],
                blob[offset + 2],
                blob[offset + 3],
            ]);
            let ny = f32::from_le_bytes([
                blob[offset + 4],
                blob[offset + 5],
                blob[offset + 6],
                blob[offset + 7],
            ]);
            let nz = f32::from_le_bytes([
                blob[offset + 8],
                blob[offset + 9],
                blob[offset + 10],
                blob[offset + 11],
            ]);

            interior_count += 1;

            // For a flat surface, normals should point upward: (0, 0, 1)
            // Allow small tolerance for floating point errors
            if nx.abs() < 0.01 && ny.abs() < 0.01 && (nz - 1.0).abs() < 0.01 {
                upward_count += 1;
            }

            // Verify the normal is normalized
            let length = (nx * nx + ny * ny + nz * nz).sqrt();
            assert!((length - 1.0).abs() < 0.01,
                "Normal at ({}, {}) should be normalized, got length {}, normal = ({}, {}, {})",
                x, y, length, nx, ny, nz);

            // Verify Z component is positive (normals should point upward)
            assert!(nz > 0.99,
                "Normal at ({}, {}) Z component should be close to 1.0 for flat surface, got nz = {}, normal = ({}, {}, {})",
                x, y, nz, nx, ny, nz);

            // Verify X and Y components are near zero
            assert!(nx.abs() < 0.01,
                "Normal at ({}, {}) X component should be close to 0 for flat surface, got nx = {}, normal = ({}, {}, {})",
                x, y, nx, nx, ny, nz);
            assert!(ny.abs() < 0.01,
                "Normal at ({}, {}) Y component should be close to 0 for flat surface, got ny = {}, normal = ({}, {}, {})",
                x, y, ny, nx, ny, nz);
        }

        // All interior normals should point perfectly upward for a flat surface
        assert_eq!(upward_count, interior_count,
            "All interior normals should point upward for flat surface, only {}/{} did",
            upward_count, interior_count);

        // Clean up temporary file
        let _ = fs::remove_file(temp_path);
    }

    #[test]
    fn test_exported_normals_sloped_surface() {
        use std::fs;

        // Create a 10x10 grid with a slope in the X direction
        let mut array = Array2D::new(10, 10, 0.0);
        for y in 0..10 {
            for x in 0..10 {
                array.set(x, y, x as f32);
            }
        }

        // Export to a temporary GLB file
        let temp_path = "/tmp/test_sloped_normals.glb";
        let result = export_array_to_glb(
            &array,
            1.0,  // scale_x
            1.0,  // scale_y
            1.0,  // scale_z
            temp_path,
            0.1,  // world_uv_scale
            false, // enable_textures
            512,  // texture_resolution
            false, // uv_test
            false, // enable_normal_maps
            1.0,  // normal_strength
            1.0,  // normal_scale
        );

        assert!(result.is_ok(), "Failed to export GLB: {:?}", result.err());

        // Load the GLB file and verify normals
        let data = fs::read(temp_path).expect("Failed to read GLB file");
        let gltf = gltf::Gltf::from_slice(&data).expect("Failed to parse GLB");

        // Get the blob containing binary data
        let blob = gltf.blob.as_ref().expect("No binary blob in GLB");

        // Get the first mesh
        let mesh = gltf.meshes().next().expect("No mesh found in GLB");
        let primitive = mesh.primitives().next().expect("No primitives found");

        // Get the normals accessor
        let normals_accessor = primitive.get(&gltf::Semantic::Normals)
            .expect("No normals found in mesh");

        // Get the buffer view for normals
        let buffer_view = normals_accessor.view().expect("No buffer view for normals");
        let buffer_offset = buffer_view.offset() + normals_accessor.offset();

        // Check normals for the sloped surface (skip edges)
        let width = 10;
        let height = 10;
        let mut interior_count = 0;

        for i in 0..normals_accessor.count() {
            let x = i % width;
            let y = i / width;

            // Skip edge vertices - only check interior vertices
            if x == 0 || y == 0 || x == width - 1 || y == height - 1 {
                continue;
            }

            let offset = buffer_offset + i * 12;
            let nx = f32::from_le_bytes([
                blob[offset],
                blob[offset + 1],
                blob[offset + 2],
                blob[offset + 3],
            ]);
            let ny = f32::from_le_bytes([
                blob[offset + 4],
                blob[offset + 5],
                blob[offset + 6],
                blob[offset + 7],
            ]);
            let nz = f32::from_le_bytes([
                blob[offset + 8],
                blob[offset + 9],
                blob[offset + 10],
                blob[offset + 11],
            ]);

            interior_count += 1;

            // Verify the normal is normalized
            let length = (nx * nx + ny * ny + nz * nz).sqrt();
            assert!((length - 1.0).abs() < 0.01,
                "Normal at ({}, {}) should be normalized, got length {}", x, y, length);

            // For a slope in X direction, nx should be negative, ny should be ~0, nz should be positive
            // The normal should point perpendicular to the slope
            assert!(nx < 0.0, "Normal at ({}, {}) X component should be negative for X-slope, got nx = {}", x, y, nx);
            assert!(ny.abs() < 0.01, "Normal at ({}, {}) Y component should be close to 0 for X-slope, got ny = {}", x, y, ny);
            assert!(nz > 0.0, "Normal at ({}, {}) Z component should be positive (pointing upward), got nz = {}", x, y, nz);
        }

        assert!(interior_count > 0, "Should have checked some interior vertices");

        // Clean up temporary file
        let _ = fs::remove_file(temp_path);
    }
}
