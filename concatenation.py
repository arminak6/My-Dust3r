import dill
import torch
import numpy as np
import os
import trimesh
from scipy.spatial.transform import Rotation
from dust3r.utils.device import to_numpy
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
    outfile = os.path.join(outdir, 'scene_test2.glb')
    if not silent:
        print('(exporting 3D scene to', outfile, ')')
    scene.export(file_obj=outfile)
    return outfile

def main():
    with open('image4_5.pkl', 'rb') as f:
        scene1 = dill.load(f)

    with open('image6_7.pkl', 'rb') as f:
        scene2 = dill.load(f)

    imgs1 = scene1.imgs
    focals1 = scene1.get_focals().cpu()
    poses1 = scene1.get_im_poses().cpu()
    pts3d1 = to_numpy(scene1.get_pts3d())
    confidence_masks1 = to_numpy(scene1.get_masks())
    intrinsics1 = scene1.get_intrinsics()
    known_mask1 = scene1.get_known_focal_mask()
    principal_points1 = scene1.get_principal_points()
    depthmap1 = scene1.get_depthmaps()

    imgs2 = scene2.imgs
    focals2 = scene2.get_focals().cpu()
    poses2 = scene2.get_im_poses().cpu()
    pts3d2 = to_numpy(scene2.get_pts3d())
    confidence_masks2 = to_numpy(scene2.get_masks())
    intrinsics2 = scene2.get_intrinsics()
    known_mask2 = scene2.get_known_focal_mask()
    principal_points2 = scene2.get_principal_points()
    depthmap2 = scene2.get_depthmaps()

    for i in range(len(imgs2)):
        imgs1.append(imgs2[i])

    focals1 = torch.cat((focals1, focals2))
    poses1 = torch.cat((poses1, poses2))

    for i in range(len(pts3d2)):
        pts3d1.append(pts3d2[i])

    for i in range(len(confidence_masks2)):
        confidence_masks1.append(confidence_masks2[i])

    intrinsics1 = torch.cat((intrinsics1, intrinsics2))
    known_mask1 = torch.cat((known_mask1, known_mask2))
    principal_points1 = torch.cat((principal_points1, principal_points2))

    for i in range(len(depthmap2)):
        depthmap1.append(depthmap2[i])

    scene1.imgs = imgs1
    scene1.get_focals = focals1
    scene1.get_im_poses = poses1
    scene1.get_pts3d = pts3d1
    scene1.get_masks = confidence_masks1
    scene1.get_intrinsics = intrinsics1
    scene1.get_known_focal_mask = known_mask1
    scene1.get_principal_points = principal_points1
    scene1.get_depthmaps = depthmap1

    concate_scene = _convert_scene_output_to_glb("./", imgs1, pts3d1, confidence_masks1, focals1, poses1)

if __name__ == "__main__":
    main()
