import argparse
import dill
from joblib import dump, load
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
import pickle
import gzip


def main(image_filenames):
    """
    Main function that processes a list of image filenames to generate scene alignment and compute global alignment loss.

    Parameters:
    image_filenames (list): A list of image filenames to process.

    Returns:
    None
    """
    try:
        if len(image_filenames) < 1:
            print("At least one image filename is required.")
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



        # store ala needed objects into objects
        objects = [scene.imgs, scene.get_pts3d(),scene.get_masks(),scene.get_focals(),scene.get_im_poses()]
        
        # save objects using gzip
        with gzip.open("new_method.pkl.gz", "wb") as file:
            pickle.dump(objects, file)
        
        # Show result
        scene.show()
    
    except Exception as e:
        print("An error occurred:", str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some images with DUSt3R.')
    parser.add_argument('images', nargs='+', help='image filenames')
    args = parser.parse_args()
    main(args.images)