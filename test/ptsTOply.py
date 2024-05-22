import argparse
import gzip
import numpy as np
import pickle
from dust3r.utils.device import to_numpy

def main(file1):
    """
    Main function that loads objects from a pickle file, processes point cloud data,
    and saves it as a PLY file with color information.

    Parameters:
    file1 (str): Path to the pickle file.

    Returns:
    None
    """
    try:
        with gzip.open(file1, "rb") as file:
            loaded_objects1 = pickle.load(file)

        imgs1 = loaded_objects1[0]
        focals1 = loaded_objects1[3]
        poses1 = loaded_objects1[4]
        pts3d1 = to_numpy(loaded_objects1[1])
        confidence_masks1 = to_numpy(loaded_objects1[2])

        # Print shapes for debugging
        print(f"Shape of pts3d1: {[arr.shape for arr in pts3d1]}")
        print(f"Shape of imgs1: {[img.shape for img in imgs1]}")

        # Ensure imgs1 contains RGB values and normalize if necessary
        imgs1_normalized = []
        for img in imgs1:
            if img.max() > 1.0:
                img = img / 255.0
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

        # Write the header to a PLY file
        with open('point_cloud_with_color2.ply', 'w') as f:
            total_points = sum([arr.size // 3 for arr in pts3d1])
            f.write(header.format(total_points))
            
            # Iterate over each array in pts3d1 and corresponding image in imgs1
            for idx, (arr, img) in enumerate(zip(pts3d1, imgs1_normalized)):
                # Flatten the array to get the points
                points = arr.reshape(-1, 3)
                # Flatten the image array to get the colors
                colors = (img.reshape(-1, 3) * 255).astype(np.uint8)
                
                # Check if the number of points matches the number of colors
                if points.shape[0] != colors.shape[0]:
                    print(f"Warning: Number of points ({points.shape[0]}) does not match number of colors ({colors.shape[0]}) for frame {idx}")
                
                # Write each point and its corresponding color to the PLY file
                for point, color in zip(points, colors):
                    f.write('{} {} {} {} {} {}\n'.format(point[0], point[1], point[2], color[0], color[1], color[2]))

    except Exception as e:
        print("An error occurred:", str(e))

# Main function to parse arguments and call the main processing function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process scene data from a pickle file.')
    parser.add_argument('file1', help='path to the pickle file')
    # parse the program arguments
    args = parser.parse_args()

    # call the main function with the program arguments
    main(args.file1)
