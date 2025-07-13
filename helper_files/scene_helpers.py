from sionna.rt import load_scene, scene

def get_scene_bounds_x(scene):
    """
    Returns the minimum and maximum x coordinates from the scene's bounding box.
    """
    bbox = scene.mi_scene.bbox()
    return bbox.min[0], bbox.max[0]

def get_scene_bounds_y(scene):
    """
    Returns the minimum and maximum y coordinates from the scene's bounding box.
    """
    bbox = scene.mi_scene.bbox()
    return bbox.min[1], bbox.max[1]

def get_scene_bounds_z(scene):
    """
    Returns the minimum and maximum z coordinates from the scene's bounding box.
    """
    bbox = scene.mi_scene.bbox()
    return bbox.min[2], bbox.max[2]

def get_scene_bounds3d(scene):
    """
    Returns the 3D bounds of the scene as a tuple:
    (x_min, x_max, y_min, y_max, z_min, z_max)
    """
    x_min, x_max = get_scene_bounds_x(scene)
    y_min, y_max = get_scene_bounds_y(scene)
    z_min, z_max = get_scene_bounds_z(scene)
    return x_min, x_max, y_min, y_max, z_min, z_max

def get_sionna_scene(scene_name="munich", rt_scene=scene.munich):
    """
    Returns the default Sionna RT scene (munich).
    
    """
    if scene_name=="munich":
        return load_scene(scene.munich)
    elif scene_name=="etoile":
        # Assuming you have a different scene for New York, replace with actual scene object
        return load_scene(scene.etoile)
    elif scene_name=="florence":
        # Assuming you have a different scene for New York, replace with actual scene object
        return load_scene(scene.florence)
    else:
        return load_scene(scene.munich)

def remove_all_transmitters(scene_obj):
    """
    Removes all transmitters from the scene object.
    
    Args:
        scene_obj: The Sionna RT scene object.
    """
    for tx in scene_obj.transmitters:
        scene_obj.remove(tx)

def coordinate_to_grid_indices(coord, x_min, y_min, z_min, cell_size):
    """
    Given a coordinate (x, y, z), return the grid indices (i, j, k) for the grid defined by the minimum bounds and cell size.
    Args:
        coord (tuple or list): (x, y, z) coordinate
        x_min, y_min, z_min (float): minimum bounds of the grid
        cell_size (tuple or list): (cell_size_x, cell_size_y, cell_size_z)
    Returns:
        tuple: (i, j, k) grid indices (as integers)
    """
    x, y, z = coord
    i = int((x - x_min) / cell_size[0])
    j = int((y - y_min) / cell_size[1])
    k = int((z - z_min) / cell_size[2])
    return (i, j, k)

def grid_indices_to_center_coordinate(indices, x_min, y_min, z_min, cell_size):
    """
    Given grid indices (i, j, k), return the coordinates of the center of the cell.
    Args:
        indices (tuple or list): (i, j, k) grid indices
        x_min, y_min, z_min (float): minimum bounds of the grid
        cell_size (tuple or list): (cell_size_x, cell_size_y, cell_size_z)
    Returns:
        tuple: (x, y, z) coordinates of the cell center
    """
    i, j, k = indices
    x = x_min + (i + 0.5) * cell_size[0]
    y = y_min + (j + 0.5) * cell_size[1]
    z = z_min + (k + 0.5) * cell_size[2]
    return (x, y, z)

