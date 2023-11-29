from decimal import Decimal, ROUND_HALF_UP
import Imath
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
import glob
import os
import cv2
from torchvision import transforms
from PointCloudDataset import ScannetPCD, PatchesBlockBuilder
import torch
import gc
import time
import torchvision
from line_profiler import LineProfiler

# TODO: use CUDA_LAUNCH_BLOCKING=1
# TODO: try line_profiler
# TODO: use torch.utils.Dataset for data loading

## DO THESE AFTER MAKING DATALOADING EFFICIENT
# TODO: try better CLIP model?
# TODO: run full pcd through CLIP

# %load_ext line_profiler
# %lprun -f 
def main():
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    OUTPUT_DIR = "./Reconstruction/output/scannet" # Where to save/load files
    NPZ_FILE = "pcd_with_img_features_test_50k.npz" # NPZ with img features to generate/load
    PCD_FILE = "pcd_test_200k.ply" # Point cloud file to generate/load
    CLIP_PATCH_SIZE = (128,128)
    # QUERY = "where are the chairs?"
    # QUERY = "¿dónde están mis zapatos?"
    QUERY = "where are the shoes?"
    NUM_POINTS = 50000
    PERFORMING_QUERY = True
    TESTING = True # enable to run on smaller subset of points
    SYNTHETIC = False
    MAX_EPOCHS = 10
    BATCH_SIZE = 1
    BLOCK_SIZE = 10
    DISABLE_PROGRESS_BAR = True

    if SYNTHETIC:
        EPSILON = 0.001 # 0.001 = 0.1mm resolution for testing occlusion
    else:
        EPSILON = 0.05

    print("Is GPU available: ", torch.cuda.is_available())
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    clip_model = CLIP(CLIP_PATCH_SIZE, device)
    pcd_path = os.path.join(OUTPUT_DIR,PCD_FILE)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if SYNTHETIC:
        depth_img_files, rgb_img_files, pose, K = loadSyntheticData()
    elif TESTING:
        dataset = ScannetPCD(_tdist=0.35,_device=device,_testing=True)
        # depth_img_files, rgb_img_files, pose, K = loadScannetTest(0.1, device)
    else:
        dataset = ScannetPCD(_tdist=0.1)
        # depth_img_files, rgb_img_files, pose, K = loadScannet(0.35)
    

    ################################################################################
    # Back-project point cloud

    # if SYNTHETIC:
    #     createPointCloud(depth_img_files, rgb_img_files, pose, K, OUTPUT_DIR, 
    #                      PCD_FILE, export=True)
    # else:
    #     dataset.createScannnetPointCloud(OUTPUT_DIR, PCD_FILE, export=True)

    ################################################################################
    # Project the point cloud back onto the RGB image

    t0 = time.time()
    patchesBlocks = PatchesBlockBuilder(_pcd_dataset=dataset, _pcd_path=pcd_path, _output_dir=OUTPUT_DIR, 
                                        _output_file=NPZ_FILE, _device=device, _block_size=BLOCK_SIZE, 
                                        _testing=TESTING, _num_points=NUM_POINTS)
    t1 = time.time()
    print("Time to create block of patches:", t1-t0)
    patchesBlocksGenerator = torch.utils.data.DataLoader(patchesBlocks, batch_size=BATCH_SIZE, 
                                                         shuffle=False, num_workers=1)
    
    new_pcd = torch.empty((0,3), dtype=torch.float64)
    img_features_l = torch.empty((0,512), dtype=torch.float32)
    colors_l = torch.empty((0,3), dtype=torch.float32)

    t2 = time.time()
    total_patches = 0
    # Load block of patches
    
    for block in tqdm.tqdm(patchesBlocksGenerator, desc="Generating point cloud features", 
                           colour="green", disable=DISABLE_PROGRESS_BAR):
        block = {key: tensor.to(device) for key, tensor in block.items()}
        patches = clip_model.getPatches(block, dataset)
        total_patches += patches.shape[0]
        colors = getColorAtPixel(dataset, block, device)
        img_features = clip_model.getImageFeatureForPatches(block, patches)
        img_features_l = torch.cat((img_features_l, img_features.to("cpu")),dim=0)
        new_pcd = torch.cat((new_pcd, block["points"][0].cpu()),dim=0)
        colors_l = torch.cat((colors_l, colors.cpu()),dim=0)

    t3 = time.time()
    print("Time to generate point cloud features:", t3-t2)
    print("Total patches:", total_patches)
    print("Patches per second", total_patches/(t3-t2))

    new_pcd = np.stack(new_pcd.cpu(),axis=0)
    colors_l = np.stack(colors_l.cpu(),axis=0)
    img_features_l = np.stack(img_features_l.cpu(),axis=0)
    np.savez(os.path.join(OUTPUT_DIR, NPZ_FILE), new_pcd, colors_l, img_features_l)

    # generatePointCloudWithFeatures(_pcd_path=pcd_path, _pose=pose, _K=K, _rgb_img_files=rgb_img_files,
    #                                _depth_img_files=depth_img_files, _clip_model=clip_model, 
    #                                _output_dir=OUTPUT_DIR, _output_file=NPZ_FILE,
    #                                _testing=TESTING, _scannet=(not SYNTHETIC), _num_points=NUM_POINTS)

    ################################################################################
    # Perform a query

    npzfile = np.load(os.path.join(OUTPUT_DIR,NPZ_FILE))
    new_pcd = npzfile['arr_0']
    colors_l = npzfile['arr_1']
    img_features_l = npzfile['arr_2']

    if PERFORMING_QUERY:
        similarity_scores = clip_model.runQuery(img_features_l, QUERY).cpu()
        heatmap_colors = [plt.cm.viridis(score)[:3] for score in similarity_scores]
        heatmap_colors = np.array(heatmap_colors)
        cloud = trimesh.PointCloud(vertices=new_pcd, colors=heatmap_colors)
        
        output_file = QUERY.replace(" ","_").replace("?","").replace("¿","") + ".ply"
        cloud.export(os.path.join(OUTPUT_DIR,output_file))
    if TESTING:
        cloud = trimesh.PointCloud(vertices=new_pcd, colors=colors_l)
        cloud.export(os.path.join(OUTPUT_DIR,"color_test.ply"))

    # exit()
    
def loadScannet(_tdist = 0.1):
    scene_dir = pathlib.Path("./Reconstruction/scannet_scene")
    rgb_img_files = np.array(sorted(
        glob.glob(os.path.join(scene_dir, "color/*.jpg")),
        key=lambda f: int(os.path.basename(f).split(".")[0]),
    ))
    depth_img_files = np.array(sorted(
        glob.glob(os.path.join(scene_dir, "depth/*.png")),
        key=lambda f: int(os.path.basename(f).split(".")[0]),
    ))
    posefiles = sorted(
        glob.glob(os.path.join(scene_dir, "pose/*.txt")),
        key=lambda f: int(os.path.basename(f).split(".")[0]),
    )
    poses = torch.from_numpy(np.stack([np.loadtxt(f) for f in posefiles], axis=0)).float()

    K_file_depth = os.path.join(scene_dir, "intrinsic/intrinsic_depth.txt")
    K_depth = torch.from_numpy(np.loadtxt(K_file_depth)).float()[:3, :3]

    # Filter out views that are close together
    kf_idx = [0]
    last_kf_pose = poses[0]
    for i in range(1, len(poses)):
        tdist = torch.norm(poses[i, :3, 3] - last_kf_pose[:3, 3])
        if tdist > _tdist:
            kf_idx.append(i)
            last_kf_pose = poses[i]
    kf_idx = np.array(kf_idx)
    
    return depth_img_files[kf_idx], rgb_img_files[kf_idx], poses[kf_idx], K_depth

def loadSyntheticData():
    print(pathlib.Path.cwd())
    scene_dir = pathlib.Path("./Reconstruction/scene_0000")
    # Load files
    depth_img_files = sorted(scene_dir.glob("depth/*/img0002.exr"))
    rgb_img_files = sorted(scene_dir.glob("rgb/*/img0002.jpg"))
    cam_npz = np.load(scene_dir / "cameras.npz")
    pose = cam_npz.f.pose # camera-to-world
    K = cam_npz.f.K

    return depth_img_files, rgb_img_files, pose, K

def generatePointCloudWithFeatures(_pcd_path, _pose, _K, _rgb_img_files,
                                   _depth_img_files, _clip_model, _output_dir,
                                   _output_file = "pcd_with_img_features.npz",
                                   _testing = True, _scannet = False, _num_points = 250):
    """
    Generate a point cloud with image features from CLIP.

    BASIC METHOD
    1. Load point clouds
    2. Iterate over a single point in the point cloud
    3. Find out what point of views the point is visible in
    4. Store all image patches for every point. (maybe preprocessed image patches)
    5. Run clip over every image patch. Free up image patch memory
    
    6. Compute element-wise average for image_feature vectors
    7. Backproject the feature vector into that point
    8. Repeat 2-6 for all points in the point cloud

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
    if _scannet:
        imheight, imwidth, _ = cv2.imread(str(_depth_img_files[0])).shape
    else:
        imheight, imwidth, _ = cv2.imread(str(_rgb_img_files[0])).shape

    device = _clip_model.device
    # 1. Load point clouds
    pcd = []
    pcd_o3d = o3d.io.read_point_cloud(_pcd_path)
    pcd.append(np.asarray(pcd_o3d.points))
    pcd = torch.from_numpy(np.concatenate(pcd, axis=0)).to(device)
    
    # Load all rgb_imgs
    rgb_imgs_PIL = []
    transform = transforms.ToTensor()
    for rgb_img_file in _rgb_img_files:
        img = Image.open(rgb_img_file).convert('RGB')
        rgb_imgs_PIL.append(img.resize((imwidth,imheight)))
    rgb_imgs_PIL = torch.stack([transform(img) for img in rgb_imgs_PIL]).to(device)
    
    if _testing:
        # Choose X random points in order to reduce computation time
        pcd_sample_idxs = torch.randperm(pcd.shape[0])[:_num_points].to(device)
        pcd_samples = pcd[pcd_sample_idxs,:]
    else:
        pcd_samples = pcd

    pcd_processed = []
    depth_imgs = []
    # 1.1 Project all points into all camera views first
    for view_idx in tqdm.trange(len(_pose), desc="Projecting points into all camera views", colour="green"):
        if _scannet:
            pcd_processed.append(getUVDepthCoordinates(_pose[view_idx], _K, pcd_samples))
        else:
            pcd_processed.append(getUVDepthCoordinates(_pose[view_idx], _K[view_idx], pcd_samples))
    for depth_img_file in _depth_img_files:
        if _scannet:
            img = cv2.imread(str(depth_img_file), cv2.IMREAD_ANYDEPTH)
            depth_imgs.append(torch.from_numpy(img.astype(np.float32)) / 1000)
        else:
            depth_imgs.append(read_depth_exr_file(str(depth_img_file)))
    pcd_processed = torch.stack(pcd_processed).to(device)
    depth_imgs = torch.stack(depth_imgs).to(device)

    new_pcd = torch.empty((0,3), dtype=torch.float64).to(device)
    img_features_l = torch.empty((0,512), dtype=torch.float32).to(device)
    colors = torch.empty((0,3), dtype=torch.uint8).to(device)

    ############
    # BATCH TEST
    ############
    
    # imgs = [rgb_imgs_PIL[0], rgb_imgs_PIL[1], rgb_imgs_PIL[2]]
    # uv_coords = torch.tensor([320,240], device=device).unsqueeze(0)
    # uv_coords_stack = torch.cat((uv_coords,uv_coords), dim=0)
    # uv_coords_stack = torch.cat((uv_coords_stack,uv_coords), dim=0)

    batchCLIPTest(1, _clip_model, 200000)
    img_patches = []

    # 2. Iterate over a single point in the point cloud
    for point_idx in tqdm.trange(len(pcd_samples), desc="Iterating over all points in the point cloud", colour="green"):
        # Test for occlusion
        view_visible = torch.tensor([], dtype=torch.int64).to(device)
        uv_coords = torch.empty((0,2), dtype=torch.int64).to(device)
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
                    view_visible = torch.cat((view_visible, torch.tensor([view_idx]).to(device)))
                    valid_uv = torch.tensor([u,v]).to(device)
                    uv_coords = torch.cat((uv_coords, valid_uv.unsqueeze(0)), dim=0)
            else:
                continue

        # If the point isn't visible from any POV, skip it.
        if len(view_visible) == 0:
            continue

        # 4. Run clip over that point from every visible camera view
        imgs = [rgb_imgs_PIL[i] for i in view_visible]
        
        # Get preprocessed image patches
        img_patches = _clip_model.getPreprocessedImagePatches()
        img_features = _clip_model.getImageFeatureForPatch(imgs, uv_coords)
        colors = torch.cat((colors, getColorAtPixel(imgs,uv_coords,_clip_model.device)),dim=0)
        new_pcd = torch.cat((new_pcd, pcd_samples[point_idx,:].unsqueeze(0)),dim=0)
        img_features_l = torch.cat((img_features_l, img_features),dim=0)

    # Export point cloud into npz file
    new_pcd = np.stack(new_pcd.cpu(),axis=0)
    colors = np.stack(colors.cpu(),axis=0)
    img_features_l = np.stack(img_features_l.cpu(),axis=0)
    np.savez(os.path.join(_output_dir, _output_file), new_pcd, colors, img_features_l)

class CLIP():
    def __init__(self, _patch_size, _device):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.model.to(_device)
        self.model.eval()
        self.device = _device
        self.patch_size = torch.tensor(_patch_size).to(_device)
        self.preprocessed_tensor_size = (224,224)
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224,224)),
            torchvision.transforms.CenterCrop((224,224)),
            torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                             std=[0.26862954, 0.26130258, 0.27577711])
        ])

        print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in self.model.parameters()]):,}")

    def preprocessFunctionNoah(self, _patches):
        patches = torch.nn.functional.interpolate(
            _patches, size=(224, 224), mode="bilinear", align_corners=False
        )
        return patches
    
    def preprocessFunction(self, _patches):
        return self.transforms(_patches)
    
    def getPatches(self, _block, _dataset):
        """
        Gets PREPROCESSED patches for a point.
        @param _block: block of views an points
        @param _dataset: dataset containing rgb images
        @return patches: patches corresponding to the point.
        """
        imgs = []
        for idx in _block["views"][0]:
            if idx == -1:
                break
            imgs.append(_dataset.rgb_imgs[idx])

        # Get correct patch regions
        crop_regions = torch.empty((0,4),dtype=torch.int32, device=self.device)
        for u,v in _block["uv_coords"][0]:
            crop_regions = torch.cat((crop_regions, torch.tensor([v-self.patch_size[1]//2,
                                                                  u-self.patch_size[0]//2,
                                                                  self.patch_size[1],
                                                                  self.patch_size[0]], dtype=torch.int32, device=self.device).unsqueeze(0)),
                                                                  dim=0)

        # Take patches from image and preprocess
        patches = torch.empty((0,3,self.preprocessed_tensor_size[0],
                               self.preprocessed_tensor_size[1]),device=self.device)
        for idx, image in enumerate(imgs):
            img_crop = torchvision.transforms.functional.crop(image, crop_regions[idx][0], crop_regions[idx][1],
                                                              crop_regions[idx][2], crop_regions[idx][3])
            
            img = self.preprocessFunction(img_crop).unsqueeze(0)
            # img = self.preprocessFunctionNoah(img_crop.unsqueeze(0))
            patches = torch.cat((patches, img), dim=0)
        return patches
    
    def getImageFeatureForPatches(self, _block, _patches):
        """
        Get image features for a block.
        """
        with torch.no_grad():
            image_features = self.model.encode_image(_patches).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Compute element-wise average for image_feature vectors
        prev_idx = 0
        features = torch.empty((0,512),device=self.device)
        
        for idx in _block["idxs"][0]:
            if idx == -1:
                continue
            temp = image_features[prev_idx:idx]
            temp = torch.mean(temp, dim=0).unsqueeze(0)
            
            features = torch.cat((features, temp), dim=0)
            prev_idx = idx

        return features
            

    def getTextFeatures(self, _text):
        """
        @param _text: list of text queries
        @return text features
        """
        with torch.no_grad():
            text_tokens = open_clip.tokenizer.tokenize(_text).to(self.device)
            text_feature = self.model.encode_text(text_tokens).float()
            text_feature /= text_feature.norm(dim=-1, keepdim=True)
            return text_feature
        
    def runQuery(self, _img_features_l, _query):
        """
        @param _img_features: list of image features
        @param _query: text query
        @return similarity scores
        """
        labels = ["an object", "things", "stuff", "texture", _query]
        labels = [f"a picture of {label}" for label in labels]
        text_feature = self.getTextFeatures(labels)

        # Calculate similarity scores
        similarity_scores = np.empty(len(_img_features_l))
        _img_features_l = torch.from_numpy(_img_features_l).to(self.device)
        temp = 100 * _img_features_l @ text_feature.T
        similarity_scores = temp.softmax(dim=-1)
        similarity_scores = similarity_scores[:, -1] # select the query
        similarity_scores = ((similarity_scores - 0.5) * 2).clamp(0, 1) # relu between 0 and 1
        
        return similarity_scores

    def getImgFeatures(self, _imgs):
        return self.model.encode_image(_imgs).float()

def batchCLIPTest(_batch_size, _clip_model, _sample_size = 200000):
    img = torch.empty((3,224,224), dtype=torch.float32, device=_clip_model.device).unsqueeze(0)
    img_stack = torch.empty((0,3,224,224), dtype=torch.float32, device=_clip_model.device)
    for i in range(_batch_size):
        img_stack = torch.cat((img_stack,img), dim=0)
    with torch.no_grad():
        for point_idx in tqdm.trange(len(_sample_size), desc="Patch test", colour="green"):
                img_featurues = _clip_model.getImgFeatures(img_stack)

def getColorAtPixel(_dataset, _block, _device):
    """
    @param _dataset: dataset containing rgb images and views
    @param _img: RGB image
    @param _uv: uv coordinates
    @return: color at pixel
    """
    colors_l = torch.empty((0,3), dtype=torch.float32, device="cpu")
    prev_idx = 0
    for x, idx in enumerate(_block["idxs"][0]):
        colors = torch.empty((0,3), dtype=torch.float32, device="cpu")
        if idx == -1:
            break
        for i, view in enumerate(_block["views"][0][prev_idx:]):
            if i+prev_idx == idx:
                break
            img = _dataset.rgb_imgs[view]
            u, v = _block["uv_coords"][0][i+prev_idx]
            if u < 0 or v < 0:
                continue
            else:
                colors = torch.cat((colors, img[:,v,u].unsqueeze(0).cpu()), dim=0)
        # clr = torch.mean(colors, dim=0).unsqueeze(0)
        # assert torch.all(clr[0] == _block["colors"][0][x].cpu())
        prev_idx = idx
        colors_l = torch.cat((colors_l, torch.mean(colors, dim=0).unsqueeze(0)), dim=0)

    return colors_l

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
    # world_to_cam = np.linalg.inv(_pose)
    world_to_cam = torch.linalg.inv(_pose)
    xyz_cam = (world_to_cam[:3, :3] @ _pcd.T + world_to_cam[:3, 3:4])

    # Matrix multiply the xyz_cam by the camera intrinsics to get uv
    uv = _K @ xyz_cam
    uv_z = uv / uv[2, :]

    # Round all uv values
    uv_z = torch.round(uv_z)
    # uv_z = np.round(uv_z)
    
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

def loadScannetTest(_tdist=0.1, _device="cpu"):
    scene_dir = pathlib.Path("./Reconstruction/scannet_scene")
    rgb_img_files = np.array(sorted(
        glob.glob(os.path.join(scene_dir, "color/*.jpg")),
        key=lambda f: int(os.path.basename(f).split(".")[0]),
    ))
    depth_img_files = np.array(sorted(
        glob.glob(os.path.join(scene_dir, "depth/*.png")),
        key=lambda f: int(os.path.basename(f).split(".")[0]),
    ))
    posefiles = sorted(
        glob.glob(os.path.join(scene_dir, "pose/*.txt")),
        key=lambda f: int(os.path.basename(f).split(".")[0]),
    )
    poses = torch.from_numpy(np.stack([np.loadtxt(f) for f in posefiles], axis=0)).to(torch.float64).to(_device)

    K_file_depth = os.path.join(scene_dir, "intrinsic/intrinsic_depth.txt")
    K_depth = torch.from_numpy(np.loadtxt(K_file_depth)).to(torch.float64)[:3, :3].to(_device)
    kf_idx = np.array([1,2,3,4,5])
    
    return depth_img_files[kf_idx], rgb_img_files[kf_idx], poses[kf_idx], K_depth

def createScannnetPointCloud(_depth_img_files, _rgb_img_files, _pose,
                             _K_depth, _output_dir, _output_file, export = False):
    """
    Iterate over all the camera views and create a point cloud with color from RGB images.
    @param _depth_img_files: list of depth images
    @param _rgb_img_files: list of RGB images
    @param _pose: list of camera poses
    @param _K: list of camera intrinsics
    @return: point cloud and point cloud colors
    """
    imheight, imwidth, _ = cv2.imread(str(_depth_img_files[0])).shape
    u = np.arange(imwidth)
    v = np.arange(imheight)
    uu, vv = np.meshgrid(u, v)
    uv = np.c_[uu.flatten(), vv.flatten()]

    _K_depth = _K_depth.cpu()
    _pose = _pose.cpu()

    # Iterate over all camera poses
    pcd = []
    pcd_colors = []
    pcd_views = []

    for view_idx in tqdm.trange(len(_pose)):
        depth_img = cv2.imread(str(_depth_img_files[view_idx]), cv2.IMREAD_ANYDEPTH)
        depth_img = torch.from_numpy(depth_img.astype(np.float32)) / 1000
        rgb_img = cv2.imread(str(_rgb_img_files[view_idx]))[..., [2, 1, 0]]

        # Get pixels with image coordinates?
        pix_vecs_depth = np.linalg.inv(_K_depth) @ np.c_[uv, np.ones((uv.shape[0], 1))].T
        depth_array = depth_img.numpy().flatten()
        xyz_cam = pix_vecs_depth * depth_array
        good_idxs = np.where(depth_array > 0)[0] # Filter out points with 0 depth
        xyz_cam = xyz_cam[:,good_idxs]

        # Reshape RGB image according to depth image shape
        rgb_img = cv2.resize(rgb_img, (depth_img.shape[1], depth_img.shape[0]), interpolation=cv2.INTER_AREA)
        rgb_img = rgb_img.reshape(-1, 3)
        rgb_img = rgb_img[good_idxs,:]

        # Rotate matrix * xyz_cam + translation
        pcd.append((_pose[view_idx, :3, :3] @ xyz_cam + _pose[view_idx, :3, 3:4]).T)
        pcd_views.append(np.full((len(pcd[0]),1), view_idx))
        pcd_colors.append(rgb_img.reshape(-1, 3))

    pcd = np.concatenate(pcd, axis=0)
    pcd_colors = np.concatenate(pcd_colors, axis=0)
    pcd_views = np.concatenate(pcd_views, axis=0)

    if export:
        _ = trimesh.PointCloud(pcd, colors=pcd_colors).export(os.path.join(_output_dir, _output_file))
   
    return pcd, pcd_colors, pcd_views

def createPointCloud(_depth_img_files, _rgb_img_files, _pose, _K, _output_dir, _output_file, export = False):
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
        _ = trimesh.PointCloud(pcd, colors=pcd_colors).export(os.path.join(_output_dir,_output_file))
   
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