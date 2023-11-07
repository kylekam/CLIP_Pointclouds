from decimal import Decimal, ROUND_HALF_UP
import Imath
import skimage.measure
import torch
import tqdm
import trimesh
import matplotlib.pyplot as plt
import numpy as np
import OpenEXR
import open3d as o3d
import open_clip
import pathlib
from PIL import Image
import pyvista as pv

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2

OUTPUT_DIR = "./Reconstruction/"
OUTPUT_FILE = "pcd_with_img_features_256.npz"
CLIP_PATCH_SIZE = (128,128)
EPSILON = 0.001 # 0.1mm resolution for testing occlusion
QUERY = "where are the chairs"

def main():
    print(pathlib.Path.cwd())
    scene_dir = pathlib.Path("./Reconstruction/scene_0000")
    pcd_path = os.path.join(OUTPUT_DIR,"pcd.ply")
    # Load files
    depth_img_files = sorted(scene_dir.glob("depth/*/img0002.exr"))
    rgb_img_files = sorted(scene_dir.glob("rgb/*/img0002.jpg"))
    cam_npz = np.load(scene_dir / "cameras.npz")
    pose = cam_npz.f.pose # camera-to-world
    K = cam_npz.f.K
    clip_model = CLIP(CLIP_PATCH_SIZE)

    imheight, imwidth, _ = cv2.imread(str(rgb_img_files[0])).shape

    ################################################################################
    # Back-project point cloud

    # pcd1, pcd_colors, pcd_views = createPointCloud(depth_img_files, rgb_img_files, pose, K, export=False)

    ################################################################################
    # Project the point cloud back onto the RGB image

    # generatePointCloudWithFeatures(_pcd_path=pcd_path, _pose=pose, _K=K, _rgb_img_files=rgb_img_files,
    #                                _depth_img_files=depth_img_files, _clip_model=clip_model, 
    #                                _output_dir=OUTPUT_DIR, _output_file=OUTPUT_FILE,
    #                                _testing=True, _num_points=100000)

    npzfile = np.load(os.path.join(OUTPUT_DIR,OUTPUT_FILE))
    new_pcd = npzfile['arr_0']
    colors = npzfile['arr_1']
    img_features_l = npzfile['arr_2']

    # Perform a query
    text_feature = clip_model.getTextFeatures(QUERY)
    text_feature /= text_feature.norm(dim=-1, keepdim=True)

    # Calculate
    similarity_scores = np.empty(len(img_features_l))
    for i, img_features in enumerate(tqdm.tqdm(img_features_l, desc="Calculating similarity scores", colour="green")):
        img_features = torch.from_numpy(img_features)
        temp = text_feature.cpu().numpy() @ img_features.cpu().numpy().T
        similarity_scores[i] = temp[0]
    norm_scores = similarity_scores / np.max(similarity_scores)
    heatmap_colors_norm = [plt.cm.turbo(score)[:3] for score in norm_scores]
    heatmap_colors_norm = np.array(heatmap_colors_norm)

    cloud = trimesh.PointCloud(vertices=new_pcd, colors=heatmap_colors_norm)
    
    output_file = QUERY.replace(" ","_") + ".ply"
    cloud.export(os.path.join(OUTPUT_DIR,output_file))
    
    exit()

def generatePointCloudWithFeatures(_pcd_path, _pose, _K, _rgb_img_files,
                                   _depth_img_files, _clip_model, _output_dir,
                                   _output_file = "pcd_with_img_features.npz",
                                   _testing = True, _num_points = 250):
    """
    Generate a point cloud with image features from CLIP.

    BASIC METHOD
    1. Load point clouds
    2. Iterate over a single point in the point cloud
    3. Find out what point of views the point is visible in
    4. Run clip over that point from every visible camera view
    5. Compute element-wise average for image_feature vectors
    6. Backproject the feature vector into that point
    7. Repeat 2-6 for all points in the point cloud

    @param _pcd_path: path to point cloud
    @param _pose: list of camera poses
    @param _K: list of camera intrinsics
    @param _rgb_img_files: list of RGB images
    @param _depth_img_files: list of depth images
    @param _clip_model: CLIP model
    @param _output_dir: output directory
    @param _testing: whether to run in testing mode
    @param _num_points: number of points to sample in testing mode
    """

    imheight, imwidth, _ = cv2.imread(str(_rgb_img_files[0])).shape

    # 1. Load point clouds
    pcd = []
    pcd_o3d = o3d.io.read_point_cloud(_pcd_path)
    pcd.append(np.asarray(pcd_o3d.points))
    pcd = np.concatenate(pcd, axis=0)
    
    # Load all rgb_imgs
    rgb_imgs_PIL = []
    for rgb_img_file in _rgb_img_files:
        rgb_imgs_PIL.append(Image.open(rgb_img_file).convert('RGB'))

    if _testing:
        # Choose X random points in order to reduce computation time
        pcd_sample_idxs = np.random.choice(pcd.shape[0], _num_points, replace=False)
        # pcd_sample_idxs = np.flip(np.append(pcd_sample_idxs,2820054)) # view 9
        # pcd_sample_idxs = np.flip(np.append(pcd_sample_idxs,5891924)) # view 19
        pcd_samples = pcd[pcd_sample_idxs,:]
    else:
        pcd_samples = pcd

    pcd_processed = []
    depth_imgs = []
    # 1.1 Project all points into all camera views first
    for view_idx in tqdm.trange(len(_pose), desc="Projecting points into all camera views", colour="green"):
        pcd_processed.append(getUVDepthCoordinates(_pose[view_idx], _K[view_idx], pcd_samples))
    for depth_img_file in _depth_img_files:
        depth_imgs.append(read_depth_exr_file(str(depth_img_file)))
    pcd_features = {} # {xyz: [features]}
    new_pcd = []
    img_features_l = []
    colors = []
    # 2. Iterate over a single point in the point cloud
    for point_idx in tqdm.trange(len(pcd_samples), desc="Iterating over all points in the point cloud", colour="green"):
        # Test for occlusion
        view_visible = []
        uv_coords = []
        # 3. Find out what point of views the point is visible in
        for view_idx in range(len(_pose)):
            # Get uv coordinates and check if in image
            u, v, depth = pcd_processed[view_idx][:,point_idx]
            u, v, depth = int(u), int(v), depth.item()

            # extra check to see if point is within bounds of image
            if ((u >= CLIP_PATCH_SIZE[0]) and (u < imwidth - CLIP_PATCH_SIZE[0]) and
                (v >= CLIP_PATCH_SIZE[1]) and (v < imheight - CLIP_PATCH_SIZE[1]) and
                (depth >= 0)):
                depth_img = depth_imgs[view_idx]
                ground_truth_depth = depth_img[v,u].item()
                if abs(ground_truth_depth - depth) < EPSILON:
                    view_visible.append(view_idx)
                    uv_coords.append([u,v])
            else:
                continue

        # If the point isn't visible from any POV, skip it.
        if len(view_visible) == 0:
            continue

        # 4. Run clip over that point from every visible camera view
        imgs = [rgb_imgs_PIL[i] for i in view_visible]
        img_features = _clip_model.getImageFeatureForPatch(imgs, uv_coords)
        colors.append(getColorAtPixel(imgs, uv_coords))
        new_pcd.append(pcd_samples[point_idx,:])
        img_features_l.append(img_features)

        # 5. Backproject clip feature into point
        pcd_features[tuple(pcd_samples[point_idx,:].tolist())] = img_features

    # Export point cloud into npz file
    new_pcd = np.stack(new_pcd,axis=0)
    colors = np.stack(colors,axis=0)
    img_features_l = np.stack(img_features_l,axis=0)
    np.savez(os.path.join(_output_dir, _output_file), new_pcd, colors, img_features_l)

class CLIP():
    def __init__(self, _patch_size):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.patch_size = _patch_size
        print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in self.model.parameters()]):,}")

    def getImageFeatureForPatch(self, _images, _uv_coords):
        """
        Get image features for a list of images. Averages the image features for all images
        and returns a single vector.
        @param _images: list of images
        @return: image features
        """      
        # Get correct patch regions
        crop_regions = []
        for u,v in _uv_coords:
            crop_regions.append([u-self.patch_size[0]/2, v-self.patch_size[1]/2, 
                                 u+self.patch_size[0]/2, v+self.patch_size[1]/2])
            
        # Take patches from image and preprocess
        images = [self.preprocess(image.crop(crop_regions[idx])) for idx,image in enumerate(_images)]

        image_inputs = torch.tensor(np.stack(images))
        with torch.no_grad():
            image_features = self.model.encode_image(image_inputs).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
        # TODO: should the norm be done before or after averaging?
        # 5. Compute element-wise average for image_feature vectors
        mean_features = torch.mean(image_features, dim=0)

        return mean_features
    
    def getTextFeatures(self, _text):
        with torch.no_grad():
            text_tokens = open_clip.tokenizer.tokenize([_text])
            return self.model.encode_text(text_tokens).float()

def getColorAtPixel(_imgs, _uv_coords):
    """
    @param _img: RGB image
    @param _uv: uv coordinates
    @return: color at pixel
    """
    color = []
    for idx, img in enumerate(_imgs):
        u, v = _uv_coords[idx]
        if u < 0 or v < 0:
            continue
        else:
            color.append(img.getpixel((u,v)))
    
    return np.mean(color, axis=0).astype(np.uint8)

def round_to_5_decimal_places(number):
    """
    @return rounds floats to 5 decimal places.
    """
    return Decimal(number).quantize(Decimal('1.00000'), rounding=ROUND_HALF_UP)

def getALLUVZCoordinates(_pcd, _point_idx):
    uvz = []
    for points in _pcd[:]:
        uvz.append(points[:,_point_idx])
    
    return uvz

def getUVDepthCoordinates(_pose, _K, _pcd):
    """
    Get uv + depth coordinates for all points in the point cloud.

    @param _pose: camera pose
    @param _K: list of camera intrinsics
    @param _pcd: point cloud
    @param _imheight: image height
    @param _imwidth: image width

    @return: uv coordinates
    """
    # Matrix multiply the point cloud by the inverse of the camera pose to get xyz_cam
    world_to_cam = np.linalg.inv(_pose)
    xyz_cam = (world_to_cam[:3, :3] @ _pcd.T + world_to_cam[:3, 3:4])

    # Matrix multiply the xyz_cam by the camera intrinsics to get uv
    uv = _K @ xyz_cam
    uv_z = uv / uv[2, :]

    # Round all uv values
    uv_z = np.round(uv_z)
    
    # Replace the z row with depth values
    uv_z[2,:] = uv[2,:]
    
    return uv_z

def backprojectTest(pose, depth_img_files, rgb_img_files, K):
    for view_idx in tqdm.trange(len(pose)):
        pcd_colors = []
        pcd = []
        depth_img = read_depth_exr_file(str(depth_img_files[view_idx]))
        rgb_img_path = str(rgb_img_files[view_idx])
        rgb_img = cv2.imread(rgb_img_path)[..., [2, 1, 0]]
        imheight, imwidth, _ = cv2.imread(str(rgb_img_files[0])).shape

        # features = getClipFeatures(rgb_img_path)

        # Read in the point clouds
        # 1, 13, 14
        idxs = [1, 13, 14]
        for idx in idxs:
            pcd_path = os.path.join(os.path.join(OUTPUT_DIR,"point_clouds"),f"pcd_{idx}.ply")
            pcd_o3d = o3d.io.read_point_cloud(pcd_path)
            pcd.append(np.asarray(pcd_o3d.points))
        pcd = np.concatenate(pcd, axis=0)             

        # Matrix multiply the point cloud by the inverse of the camera pose to get xyz_cam
        world_to_cam = np.linalg.inv(pose[view_idx])
        xyz_cam = (world_to_cam[:3, :3] @ pcd.T + world_to_cam[:3, 3:4])

        # Matrix multiply the xyz_cam by the camera intrinsics to get uv
        uv = K[view_idx] @ xyz_cam
        uv_z = uv / uv[2, :]

        # Get colors from RGB image and use to color the point cloud
        # Get binary_encoding of uv coordinates that are within image size
        binary_encoding = np.zeros(uv_z.shape[1])
        binary_encoding[(uv_z[0,:] >= 0) & (uv_z[0,:] < imwidth) & 
                         (uv_z[1,:] >= 0) & (uv_z[1,:] < imheight) &
                         (uv_z[2,:] >= 0)] = 1
             
        binary_encoding = binary_encoding.astype(bool)

        # Use one-hot encoding with pcd. Backproject pcd_colors onto pcd        
        pcd_temp = pcd[binary_encoding,:]
        xyz_cam_temp = xyz_cam[:,binary_encoding]
        uv_temp = uv[:,binary_encoding]
        uv_z_temp = uv_z[:,binary_encoding]

        # Look for pixels with the same xy coordinates and keep only the closest one
        # TODO: sample depth map to test for occlusion
        seen_pixel = {}
        for idx in range(uv_z_temp.shape[1]):
            x_coord = round(uv_z_temp[0,idx])
            y_coord = round(uv_z_temp[1,idx])

            if x_coord == imwidth or y_coord == imheight:
                continue
            # If value in depth map is greater than current depth, then keep 
            if depth_img[y_coord,x_coord] >= uv_temp[2,idx]:
                seen_pixel[x_coord,y_coord] = idx
            # else, point is occluded

            # if (x_coord,y_coord)
            # if (x_coord,y_coord) not in seen_pixel:
            #     seen_pixel[x_coord,y_coord] = (uv_temp[2,idx], idx)
            # else:
            #     # If current depth is smaller that the one in seen_pixel, replace it
            #     if seen_pixel[x_coord,y_coord][0] > uv_temp[2,idx]:
            #         seen_pixel[x_coord,y_coord] = (uv_temp[2,idx], idx)

        # Create pointcloud
        # Fill in gaps with 0 depth values
        final_pcd = []
        missing = 0
        for j in range(imheight):
            for i in range(imwidth):
                if (i,j) not in seen_pixel:
                    final_pcd.append([0,0,0])
                    missing += 1
                else:
                    # final_pcd.append(uv_temp[:,seen_pixel[(i,j)][1]])
                    final_pcd.append(pcd_temp[seen_pixel[(i,j)]])
        
        final_pcd = np.asarray(final_pcd)

        pcd_colors.append(rgb_img.reshape(-1, 3))
        pcd_colors = np.concatenate(pcd_colors, axis=0)

        _ = trimesh.PointCloud(final_pcd, colors=pcd_colors).export(os.path.join(OUTPUT_DIR,"pcd_colors.ply"))

def createPointCloud(_depth_img_files, _rgb_img_files, _pose, _K, export = False):
    """
    Iterate over all the camera views and create a point cloud with color from RGB images.
    @param _depth_img_files: list of depth images
    @param _rgb_img_files: list of RGB images
    @param _pose: list of camera poses
    @param _K: lsit of camera intrinsics
    @return: point cloud and point cloud colors
    """
    imheight, imwidth, _ = cv2.imread(str(_rgb_img_files[0])).shape
    u = np.arange(imwidth)
    v = np.arange(imheight)
    uu, vv = np.meshgrid(u, v)
    uv = np.c_[uu.flatten(), vv.flatten()]

    # Iterate over all camera poses
    pcd = []
    pcd_colors = []
    pcd_views = []

    for view_idx in tqdm.trange(len(_pose)):

        depth_img = read_depth_exr_file(str(_depth_img_files[view_idx]))
        rgb_img = cv2.imread(str(_rgb_img_files[view_idx]))[..., [2, 1, 0]]
        
        # Get pixels with image coordinates?
        pix_vecs = np.linalg.inv(_K[view_idx]) @ np.c_[uv, np.ones((uv.shape[0], 1))].T
        xyz_cam = pix_vecs * depth_img.flatten()

        # Rotate matrix * xyz_cam + translation
        pcd.append((_pose[view_idx, :3, :3] @ xyz_cam + _pose[view_idx, :3, 3:4]).T)
        pcd_views.append(np.full((len(pcd[0]),1), view_idx))
        pcd_colors.append(rgb_img.reshape(-1, 3))

    pcd = np.concatenate(pcd, axis=0)
    pcd_colors = np.concatenate(pcd_colors, axis=0)
    pcd_views = np.concatenate(pcd_views, axis=0)

    if export:
        _ = trimesh.PointCloud(pcd, colors=pcd_colors).export(os.path.join(OUTPUT_DIR,"pcd.ply"))
   
    return pcd, pcd_colors, pcd_views

def read_depth_exr_file(filepath):
    exrfile = OpenEXR.InputFile(filepath)
    raw_bytes = exrfile.channel("B", Imath.PixelType(Imath.PixelType.FLOAT))
    depth_vector = np.frombuffer(raw_bytes, dtype=np.float32)
    height = (
        exrfile.header()["displayWindow"].max.y
        + 1
        - exrfile.header()["displayWindow"].min.y
    )
    width = (
        exrfile.header()["displayWindow"].max.x
        + 1
        - exrfile.header()["displayWindow"].min.x
    )
    depth_map = np.reshape(depth_vector, (height, width)).copy()
    depth_map[depth_map > 65499] = 0
    return depth_map

def points_to_image_torch(xs, ys, ps, sensor_size=(180, 240)):
    xt, yt, pt = torch.from_numpy(xs), torch.from_numpy(ys), torch.from_numpy(ps)
    img = torch.zeros(sensor_size)
    img.index_put_((yt, xt), pt, accumulate=True)
    return img

if __name__ == "__main__":
    main()