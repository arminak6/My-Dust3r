import dill
# from .dust3r.utils.image import find_ration
import torch


# Load the scene object using dill
with open('image4_5.pkl', 'rb') as f:
    scene1 = dill.load(f)

with open('image6_7.pkl', 'rb') as f:
    scene2 = dill.load(f)    


# ratio = find_ration("DJI_20231115155942_0004_V.JPG")
# imgs = scene.imgs
# focals = scene.get_focals()
# poses = scene.get_im_poses()
# pts3d = scene.get_pts3d()
# confidence_masks = scene.get_masks()
# intrinsics = scene.get_intrinsics()
# known_mask = scene.get_known_focal_mask()
# principal_points = scene.get_principal_points()
# depthmap = scene.get_depthmaps()


# intrinsics = torch.mul(intrinsics, ratio)


# print("this is image: \n", imgs)
# print("==============================================================================")
# print("this is focal: \n", focals)
# print("==============================================================================")
# print("this is pose: \n", poses)
# print("==============================================================================")
# print("this is pts3d: \n", pts3d)
# print("==============================================================================")
# print("this is confidence mask: \n", confidence_masks)
# print("==============================================================================")
# print("this is intrinsics: \n", intrinsics)
# print("==============================================================================")
# print("this is scene.get_pw_poses: \n" , scene.get_pw_poses)
# print("==============================================================================")
# print("this is known_mask: \n", known_mask)
# print("==============================================================================")
# print("this is principal_points: \n", principal_points)
# print("==============================================================================")
# print("this is depthmap: \n", depthmap)


# scene.show()





imgs1 = scene1.imgs
focals1 = scene1.get_focals()
poses1 = scene1.get_im_poses()
pts3d1 = scene1.get_pts3d()
confidence_masks1 = scene1.get_masks()
intrinsics1 = scene1.get_intrinsics()
known_mask1 = scene1.get_known_focal_mask()
principal_points1 = scene1.get_principal_points()
depthmap1 = scene1.get_depthmaps()


imgs2 = scene2.imgs
focals2 = scene2.get_focals()
poses2 = scene2.get_im_poses()
pts3d2 = scene2.get_pts3d()
confidence_masks2 = scene2.get_masks()
intrinsics2 = scene2.get_intrinsics()
known_mask2 = scene2.get_known_focal_mask()
principal_points2 = scene2.get_principal_points()
depthmap2 = scene2.get_depthmaps()




# Done images
for i in range(len(imgs2)):
    imgs1.append(imgs2[i])

# Done focal
focals1 = torch.cat((focals1, focals2))

# Done poses
poses1 = torch.cat((poses1, poses2))

# Done pts3d
for i in range(len(pts3d2)):
    pts3d1.append(pts3d2[i])


# Done confidence_masks
for i in range(len(confidence_masks2)):
    confidence_masks1.append(confidence_masks2[i])


# Done intrinsics
intrinsics1 = torch.cat((intrinsics1, intrinsics2))

# Done known_mask
known_mask1 = torch.cat((known_mask1, known_mask2))

# Done principal_points
principal_points1 = torch.cat((principal_points1, principal_points2))


# done_depthmap
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



scene1.show()
