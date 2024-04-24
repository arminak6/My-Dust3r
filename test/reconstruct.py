import dill
from dust3r.utils.image import find_ration
import torch


# Load the scene object using dill
with open('3images.pkl', 'rb') as f:
    scene = dill.load(f)


ratio = find_ration("DJI_20231115155942_0004_V.JPG")
# print("this is ratio: \n", ratio)
imgs = scene.imgs
focals = scene.get_focals()
poses = scene.get_im_poses()
pts3d = scene.get_pts3d()
confidence_masks = scene.get_masks()
intrinsics = scene.get_intrinsics()



intrinsics = torch.mul(intrinsics, ratio)


print("this is image: \n", imgs)
print("this is focal: \n", focals)
print("this is pose: \n", poses)
print("this is pts3d: \n", pts3d)
print("this is confidence mask: \n", confidence_masks)
print("this is intrinsics: \n", intrinsics)
print("this is scene.get_pw_poses: \n" , scene.get_pw_poses)


# # scene.show()
