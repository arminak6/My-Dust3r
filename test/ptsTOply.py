import argparse
import gzip
import numpy as np
import pickle
import torch
from dust3r.utils.device import to_numpy

def apply_pose_transformation(points, pose):
    """
    Applies the camera pose transformation to 3D points.

    Parameters:
    points (np.ndarray): 3D points of shape (N, 3)
    pose (np.ndarray or torch.Tensor): 4x4 camera pose transformation matrix

    Returns:
    np.ndarray: Transformed 3D points of shape (N, 3)
    """
    if isinstance(pose, torch.Tensor):
        pose = pose.detach().cpu().numpy()

    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed_points_homogeneous = points_homogeneous @ pose.T
    transformed_points = transformed_points_homogeneous[:, :3]
    
    return transformed_points

def compute_image_corners(img_shape, focal_length, pose):
    """
    Compute the 3D coordinates of the corners of the 2D image plane using the focal length and camera pose.

    Parameters:
    img_shape (tuple): Shape of the image (height, width)
    focal_length (float): Focal length of the camera
    pose (np.ndarray): 4x4 camera pose transformation matrix

    Returns:
    np.ndarray: 3D coordinates of the image plane corners
    """
    img_height, img_width = img_shape[:2]

    # Image plane in camera space (assuming the image plane is at z = 1.0 in camera space)
    corners_camera_space = np.array([
        [-img_width / 2, -img_height / 2, focal_length],  # Bottom-left
        [img_width / 2, -img_height / 2, focal_length],   # Bottom-right
        [img_width / 2, img_height / 2, focal_length],    # Top-right
        [-img_width / 2, img_height / 2, focal_length]    # Top-left
    ])

    # Transform corners from camera space to world space using the pose
    corners_world_space = apply_pose_transformation(corners_camera_space, pose)

    return corners_world_space

def write_camera_position(f, pose, color=(255, 0, 0)):
    """
    Writes the camera position extracted from the pose matrix to the PLY file.

    Parameters:
    f (file object): The PLY file to write to.
    pose (np.ndarray): 4x4 camera pose transformation matrix.
    color (tuple): RGB color for the camera point.

    Returns:
    None
    """
    camera_position = pose[:3, 3]
    f.write('{} {} {} {} {} {}\n'.format(camera_position[0], camera_position[1], camera_position[2], color[0], color[1], color[2]))

def write_image_corners(f, corners, color=(0, 255, 0)):
    """
    Writes the 3D image plane corners to the PLY file.

    Parameters:
    f (file object): The PLY file to write to.
    corners (np.ndarray): 3D coordinates of the image plane corners.
    color (tuple): RGB color for the image corners.

    Returns:
    None
    """
    for corner in corners:
        f.write('{} {} {} {} {} {}\n'.format(corner[0], corner[1], corner[2], color[0], color[1], color[2]))

def write_frustum_lines(f, camera_position, corners, color=(0, 0, 255)):
    """
    Writes the frustum lines connecting the camera position to the image corners.

    Parameters:
    f (file object): The PLY file to write to.
    camera_position (np.ndarray): 3D position of the camera.
    corners (np.ndarray): 3D coordinates of the image plane corners.
    color (tuple): RGB color for the frustum lines.

    Returns:
    None
    """
    for corner in corners:
        f.write('{} {} {} {} {} {}\n'.format(camera_position[0], camera_position[1], camera_position[2], color[0], color[1], color[2]))
        f.write('{} {} {} {} {} {}\n'.format(corner[0], corner[1], corner[2], color[0], color[1], color[2]))

def main(file1):
    """
    Main function that loads objects from a pickle file, processes point cloud data,
    applies pose transformations, saves 3D points, camera positions, and image corners as a PLY file.

    Parameters:
    file1 (str): Path to the pickle file.

    Returns:
    None
    """
    try:
        with gzip.open(file1, "rb") as file:
            loaded_objects1 = pickle.load(file)

        imgs1 = loaded_objects1[0]  # Images (for color)
        focals1 = loaded_objects1[3]  # Camera focal lengths
        poses1 = loaded_objects1[4]  # Camera poses (4x4 transformation matrices)

        pts3d1 = [pts.tensor.detach().numpy() for pts in loaded_objects1[1]]  # 3D points
        confidence_masks1 = [mask.tensor.detach().numpy() for mask in loaded_objects1[2]]  # Confidence masks

        # Print shapes for debugging
        print(f"Shape of pts3d1: {[arr.shape for arr in pts3d1]}")
        print(f"Shape of imgs1: {[img.shape for img in imgs1]}")

        # Ensure imgs1 contains RGB values and normalize if necessary
        imgs1_normalized = []
        for img in imgs1:
            if img.max() > 1.0:
                img = img / 255.0  # Normalize to [0, 1]
            imgs1_normalized.append(img)

        # Define the header of the PLY file with color information
        header = """ply
        format ascii 1.0
        element vertex {}
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        """

        with open('./eybaba/ammoPretto1_with_cameras_and_frustums.ply', 'w') as f:
            total_points = sum([arr.size // 3 for arr in pts3d1])
            total_camera_positions = len(poses1)
            total_image_corners = total_camera_positions * 4  # Each camera has 4 image corners
            f.write(header.format(total_points + total_camera_positions + total_image_corners))  # Include camera positions and image corners
            
            # Write 3D points and colors
            for idx, (arr, img, pose, focal) in enumerate(zip(pts3d1, imgs1_normalized, poses1, focals1)):
                points = arr.reshape(-1, 3)
                transformed_points = apply_pose_transformation(points, pose)
                colors = (img.reshape(-1, 3) * 255).astype(np.uint8)
                
                if transformed_points.shape[0] != colors.shape[0]:
                    print(f"Warning: Number of points ({transformed_points.shape[0]}) does not match number of colors ({colors.shape[0]}) for frame {idx}")
                
                for point, color in zip(transformed_points, colors):
                    f.write('{} {} {} {} {} {}\n'.format(point[0], point[1], point[2], color[0], color[1], color[2]))

                # Write camera positions (in red)
                write_camera_position(f, pose)

                # Compute and write image corners (in green)
                img_shape = img.shape
                corners_world = compute_image_corners(img_shape, focal, pose)
                write_image_corners(f, corners_world)

                # Write frustum lines (in blue)
                camera_position = pose[:3, 3]
                write_frustum_lines(f, camera_position, corners_world)

    except Exception as e:
        print("An error occurred:", str(e))

# Main function to parse arguments and call the main processing function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process scene data from a pickle file.')
    parser.add_argument('file1', help='path to the pickle file')
    args = parser.parse_args()
    main(args.file1)
