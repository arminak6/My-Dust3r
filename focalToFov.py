import math

def focal_length_to_fov(focal_length, sensor_dimension):
    """
    Convert focal length to field of view.
    
    Parameters:
    focal_length (float): The focal length of the lens in mm.
    sensor_dimension (float): The dimension of the camera sensor (width or height) in mm.
    
    Returns:
    float: Field of view in degrees.
    """
    # Calculate the field of view in radians
    fov_rad = 2 * math.atan(sensor_dimension / (2 * focal_length))
    
    # Convert radians to degrees
    fov_deg = math.degrees(fov_rad)
    
    return fov_deg

# Example usage:
focal_length = 375  # mm
sensor_size = 512  # mm

fov = focal_length_to_fov(focal_length, sensor_size)

print(f"FOV: {fov} degrees")
