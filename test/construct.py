import argparse
import os
import dill
from joblib import dump, load
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
import pickle
import gzip


def main(directory):
    """
    Main function that processes all image files within a directory to generate scene alignment and compute global alignment loss.

    Parameters:
    directory (str): Path to the directory containing image files.

    Returns:
    None
    """
    try:
        if not os.path.isdir(directory):
            print("Invalid directory path:", directory)
            return
        
        # Get all files in the directory
        image_filenames = [os.path.join(directory, filename) for filename in os.listdir(directory)]
        
        # Filter out only image files
        image_filenames = [filename for filename in image_filenames if os.path.isfile(filename) and filename.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(image_filenames) < 1:
            print("No image files found in the directory:", directory)
            return
        
        device = 'cpu'
        batch_size = 1
        schedule = 'cosine'
        lr = 0.01
        niter = 300
        model_name = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
        model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
        
        images = load_images(image_filenames, size=512)
        pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, model, device, batch_size=batch_size)
        scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
        loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

        # Store all needed objects into a list
        objects = [scene.imgs, scene.get_pts3d(), scene.get_masks(), scene.get_focals(), scene.get_im_poses()]
        
        # Save objects using gzip
        with gzip.open("new_method_2.pkl.gz", "wb") as file:
            pickle.dump(objects, file)
        
        # Show result
        scene.show()
    
    except Exception as e:
        print("An error occurred:", str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some images with DUSt3R.')
    parser.add_argument('directory', help='path to directory containing image files')
    args = parser.parse_args()
    main(args.directory)
