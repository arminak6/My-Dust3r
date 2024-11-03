1 - Introduction

3D reconstruction is the process of creating a digital 3D model from 2D images, widely used in fields like virtual reality, robotics, and medical imaging. By capturing images from multiple viewpoints, algorithms identify common points across the images to triangulate their positions in 3D space. This generates a point cloud or depth map that represents the shape and structure of the scene. Traditional approaches, such as Structure-from-Motion (SfM) and Multi-View Stereo (MVS), rely on accurate camera calibration and known poses to achieve precise results. However, recent advances in machine learning have enabled more flexible methods that can reconstruct 3D models without detailed camera information, making the process faster and more adaptable to complex environments. These improvements are transforming 3D reconstruction into a key technology across many industries.

2 - Project Overview

Our project aims to reconstruct a detailed 3D model of a cultural heritage site using a minimum of 100 images captured from various viewpoints. The goal is to create an accurate digital representation of the site that can be used for preservation, analysis, or virtual tourism. To achieve this, we chose to use DUSt3R (Dense Unconstrained Stereo 3D Reconstruction), a state-of-the-art method for 3D reconstruction that does not rely on explicit camera calibration or predefined poses.
Before diving into the specifics of our project, it's essential to review the general concepts behind 3D reconstruction. This includes traditional techniques like Structure-from-Motion (SfM) and Multi-View Stereo (MVS), which rely on triangulation and known camera parameters to build 3D models. However, these methods often require precise calibration and well-aligned images to deliver accurate results.
In the following sections, we will explain these general concepts in more detail, outline the workings of the DUSt3R model, and then describe how we applied it to our project. We will also address the challenges we faced and discuss the solutions we implemented to handle large datasets effectively. Finally, we will review the results of our 3D reconstruction process and evaluate its success in capturing the heritage site.

3 - Multi-View Stereo (MVS)

Multi-View Stereo (MVS)[1] is a technique used in 3D reconstruction to create detailed 3D models from multiple 2D images of a scene taken from different viewpoints. It works by identifying corresponding points in these images and using triangulation to estimate the 3D positions of those points. MVS builds on Structure-from-Motion (SfM), which provides the camera poses, to generate a dense reconstruction of the scene. The process typically involves steps like feature matching, depth map estimation, and refinement to create an accurate representation of the surfaces in the scene. MVS is widely used in applications like 3D mapping, virtual reality, and robotics, but it traditionally relies on well-calibrated cameras and sufficient overlap between images to perform effectively.

4 - Structure-from-Motion (SfM)

Structure-from-Motion (SfM)[2] is a method used in 3D reconstruction to estimate the 3D structure of a scene from a sequence of 2D images taken from different viewpoints. SfM simultaneously recovers both the camera positions (extrinsics) and the 3D coordinates of points in the scene by analyzing how objects move between images. This involves detecting and matching key features across images, then using geometric algorithms to calculate the relative motion of the camera and the 3D locations of the matched points. SfM is an essential step in creating sparse 3D reconstructions, and it often serves as a foundation for denser techniques like Multi-View Stereo (MVS). It is used in fields such as computer vision, photogrammetry, and robotics, especially when the camera calibration is unknown.

5 - DUSt3R Method

DUSt3R (Dense Unconstrained Stereo 3D Reconstruction)[3] is a novel approach to 3D reconstruction that operates without requiring camera calibration or pose information. The method begins by taking two RGB images as input, which are processed by a shared Vision Transformer (ViT) encoder. The ViT divides the images into patches and converts them into feature representations. These features are passed through a transformer decoder, which uses both self-attention (within each view) and cross-attention (between views) to correlate the images and align their geometric information.
The network outputs pointmaps, which are 2D grids of 3D points that represent the geometry of the scene. These pointmaps are expressed in the same coordinate system, ensuring consistent 3D reconstruction across views. For multi-view scenarios (with more than two images), DUSt3R applies a global alignment step that aligns pointmaps in a shared 3D space. This step optimizes camera poses and geometry in 3D space, avoiding the traditional bundle adjustment process, which minimizes 2D reprojection errors.
The model is trained with a 3D regression loss, measuring the Euclidean distance between predicted and ground-truth pointmaps. Additionally, confidence maps are predicted to handle uncertain regions in the image, weighting the regression loss and allowing the network to focus on more reliable parts of the scene.
A major advantage of DUSt3R is that it does not rely on explicit camera models, learning geometric relations implicitly through the transformer architecture. This flexibility allows it to handle both monocular and multi-view reconstruction without camera calibration. Its transformer-based design provides end-to-end learning, simplifying the traditional pipeline of feature extraction and triangulation. The global alignment process further streamlines optimization by working directly in 3D space, leading to faster and more robust reconstruction.

6 - Limitation in Our Experiment

One limitation encountered while running the DUSt3R algorithm is its inability to handle large image datasets efficiently, even with 32GB RAM and an RTX 4080 GPU. The algorithm fails with more than 30 images due to memory overflow, restricting its scalability when processing larger image sets. This highlights the need for more powerful hardware or optimization of the algorithm to manage larger inputs effectively.

7 - Our Solution: Handling Large Datasets and File Storage

To address the limitations of running DUSt3R on large datasets, such as GPU memory overflow on an RTX 4050 with 32GB RAM, we developed a strategy involving the processing of smaller subsets of images and merging the resulting partial 3D reconstructions. Each subset was processed individually, generating 3D pointmaps that were then saved and imported into CloudCompare, an open-source software that supports .PLY format for 3D point cloud editing and registration.
In CloudCompare, we applied the Iterative Closest Point (ICP)[4] algorithm to align and merge the partial reconstructions into a complete 3D model. Since DUSt3R provides us with the focal length (in pixels) and the relative camera pose, we further enhanced the accuracy by converting the focal length into Field of View (FOV) using the sensor size. This additional information, combined with the relative pose data, was used to initialize ICP in CloudCompare, allowing us to bypass the memory constraints and achieve a consistent, unified 3D reconstruction.

7 – 1 The Iterative Closest Point (ICP) Algorithm

The Iterative Closest Point (ICP) algorithm is a common method in 3D computer vision for aligning two point clouds by minimizing the distance between corresponding points. It works by iteratively selecting the closest points between two clouds, computing the transformation (rotation and translation) that best aligns them, and applying this transformation. This process is repeated until the alignment converges, meaning the changes between iterations become minimal. ICP is particularly useful for merging partial 3D reconstructions, as it can handle variations in position and orientation between datasets. However, it requires a good initial alignment to ensure that the solution converges accurately.

Another challenge we faced was the large file size of the output generated by DUSt3R. The pointmaps and other related data created massive files, which were difficult to store and manage. To solve this, we implemented two key steps. First, we selectively retained only the essential information required for the next stages of the process, reducing the overall data footprint. Second, we compressed the remaining files by zipping them, which not only saved storage space but also facilitated easier data transfer. After compressing the data, we used these zipped files to reconstruct the entire scene and applied ICP once again to merge the partial reconstructions into a unified model. This approach allowed us to effectively manage both storage and computational resources while maintaining the quality of the final 3D reconstruction.


8 - Final Result

Our final result demonstrates the effectiveness of the DUSt3R model, even with a limited number of images. In this example, we used only 5 images taken from different viewpoints, and despite the small dataset, we achieved a reasonably accurate 3D reconstruction. The model successfully captured the main features of the scene, demonstrating its robustness and ability to perform well even with minimal input data. This result highlights the efficiency of the approach and its potential for larger-scale reconstructions, where the accuracy and detail would only improve with a greater number of images.











Reference
[1] Seitz, S. M., Curless, B., Diebel, J., Scharstein, D., & Szeliski, R. (2006). A Comparison and Evaluation of Multi-View Stereo Reconstruction Algorithms. In Proceedings of the 2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition - Volume 1 (pp. 519–528). https://doi.org/10.1109/CVPR.2006.19

[2] omasi, C., & Kanade, T. (1992). Shape and Motion from Image Streams under Orthography: a Factorization Method. In International Journal of Computer Vision, 9(2), 137–154. https://doi.org/10.1007/BF00129684

[3] Wang, S., Leroy, V., Cabon, Y., Chidlovskii, B., & Revaud, J. (2023). DUSt3R: Geometric 3D Vision Made Easy. Aalto University & Naver Labs Europe. Retrieved from https://dust3r.europe.naverlabs.com 

[4] Besl, P. J., & McKay, N. D. (1992). A Method for Registration of 3-D Shapes. In IEEE Transactions on Pattern Analysis and Machine Intelligence, 14(2), 239–256. https://doi.org/10.1109/34.121791
