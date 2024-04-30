import argparse
import gzip
import numpy as np
import os
import pickle
import torch
import trimesh
from scipy.spatial.transform import Rotation
from dust3r.utils.device import to_numpy
# from dust3r.utils.scene_converter import _convert_scene_output_to_glb
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes



def _convert_scene_output_to_glb(outdir, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                                 cam_color=None, as_pointcloud=False, transparent_cams=False, silent=False):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        scene.add_geometry(pct)
    else:
        meshes = [pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]) for i in range(len(imgs))]
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(scene, pose_c2w, camera_edge_color, None if transparent_cams else imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))


    outfile = os.path.join(outdir, 'ssacmilan.glb')
    if not silent:
        print('(exporting 3D scene to', outfile, ')')
    scene.export(file_obj=outfile)
    return outfile

def main(file1, file2):
    """
    Main function that loads objects from two pickle files, combines them, and performs further processing.

    Parameters:
    file1 (str): Path to the first pickle file.
    file2 (str): Path to the second pickle file.

    Returns:
    None
    """
    try:
        with gzip.open(file1, "rb") as file:
            loaded_objects1 = pickle.load(file)

        with gzip.open(file2, "rb") as file:
            loaded_objects2 = pickle.load(file)

        imgs1 = loaded_objects1[0]
        focals1 = loaded_objects1[3]
        poses1 = loaded_objects1[4]
        pts3d1 = to_numpy(loaded_objects1[1])
        confidence_masks1 = to_numpy(loaded_objects1[2])

        imgs2 = loaded_objects2[0]
        focals2 = loaded_objects2[3]
        poses2 = loaded_objects2[4]
        pts3d2 = to_numpy(loaded_objects2[1])
        confidence_masks2 = to_numpy(loaded_objects2[2])

        for i in range(len(imgs2)):
            imgs1.append(imgs2[i])

        focals1 = torch.cat((focals1, focals2))
        poses1 = torch.cat((poses1, poses2))

        for i in range(len(pts3d2)):
            pts3d1.append(pts3d2[i])

        for i in range(len(confidence_masks2)):
            confidence_masks1.append(confidence_masks2[i])

        concate_scene = _convert_scene_output_to_glb("./", imgs1, pts3d1, confidence_masks1, focals1, poses1)

    except Exception as e:
        print("An error occurred:", str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process scene data from pickle files.')
    parser.add_argument('file1', help='path to the first pickle file')
    parser.add_argument('file2', help='path to the second pickle file')
    args = parser.parse_args()
    main(args.file1, args.file2)
