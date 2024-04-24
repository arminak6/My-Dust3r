from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
import dill


image_filenames = [
        'DJI_20231115155942_0004_V.JPG',
        'DJI_20231115155944_0005_V.JPG',
        'DJI_20231115155946_0006_V.JPG'
]

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

with open('3images.pkl', 'wb') as f:
    dill.dump(scene, f)
scene.show()