import dill
import torch
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode


# Load outputs from files
with open('agha.pkl', 'rb') as f:
    output1 = dill.load(f)

with open('agha2.pkl', 'rb') as f:
    output2 = dill.load(f)


# Merge image data
merged_img_view1 = torch.cat([output1['view1']['img'], output2['view1']['img']], dim=0)
merged_img_view2 = torch.cat([output1['view2']['img'], output2['view2']['img']], dim=0)

# Merge true shapes
merged_true_shape_view1 = torch.cat([output1['view1']['true_shape'], output2['view1']['true_shape']], dim=0)
merged_true_shape_view2 = torch.cat([output1['view2']['true_shape'], output2['view2']['true_shape']], dim=0)

# Merge indices and instances
merged_idx_view1 = output1['view1']['idx'] + output2['view1']['idx']
merged_instance_view1 = output1['view1']['instance'] + output2['view1']['instance']

merged_idx_view2 = output1['view2']['idx'] + output2['view2']['idx']
merged_instance_view2 = output1['view2']['instance'] + output2['view2']['instance']

# Merge predicted 3D points
merged_pred1_pts3d = torch.cat([output1['pred1']['pts3d'], output2['pred1']['pts3d']], dim=0)
merged_pred2_pts3d_in_other_view = torch.cat([output1['pred2']['pts3d_in_other_view'], output2['pred2']['pts3d_in_other_view']], dim=0)

# Merge confidence values
merged_conf_pred1 = torch.cat([output1['pred1']['conf'], output2['pred1']['conf']], dim=0)
merged_conf_pred2 = torch.cat([output1['pred2']['conf'], output2['pred2']['conf']], dim=0)

# Create the merged output dictionary
output3 = {
    'view1': {
        'img': merged_img_view1,
        'true_shape': merged_true_shape_view1,
        'idx': merged_idx_view1,
        'instance': merged_instance_view1
    },
    'view2': {
        'img': merged_img_view2,
        'true_shape': merged_true_shape_view2,
        'idx': merged_idx_view2,
        'instance': merged_instance_view2
    },
    'pred1': {
        'pts3d': merged_pred1_pts3d,
        'conf': merged_conf_pred1
    },
    'pred2': {
        'pts3d_in_other_view': merged_pred2_pts3d_in_other_view,
        'conf': merged_conf_pred2,
        'loss': None  # Placeholder for loss, to be computed later
    }
}

# Define model and training parameters
model_path = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
device = 'cpu'
batch_size = 1
schedule = 'cosine'
lr = 0.01
niter = 300

# Perform global alignment
scene = global_aligner(output3, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

# Save the scene object
with open('zareE2.pkl', 'wb') as f:
    dill.dump(scene, f)

# Show the scene
scene.show()
