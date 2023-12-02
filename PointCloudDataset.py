import os
import torch
import torchvision
import numpy as np
import math
import torch
import glob
import pathlib
import tqdm
from PIL import Image
import cv2
import trimesh
import open3d as o3d
import open_clip
import OpenEXR
import Imath

ALOT = 1e6
class SyntheticPCD(torch.utils.data.Dataset):
    def __init__(self, _model, _output_dir, _output_file, _epsilon, _tdist=0.1, 
                 _device="cuda", _testing=False, _export=False,
                 _num_points=50000, _patch_size=(128,128)):
        print("Loading synthetic point cloud dataset...")
        self.imheight = 480
        self.imwidth = 640
        self.patch_size = _patch_size
        self.epsilon = _epsilon
        self.device = _device
        self.model = _model
        self.preprocessed_tensor_size = torch.tensor([224,224], device=self.device)

        scene_dir = pathlib.Path("./Reconstruction/synthetic_scene")
        rgb_img_files = sorted(
            scene_dir.glob("rgb/*/img0002.jpg"),
        )
        depth_img_files = sorted(
            scene_dir.glob("depth/*/img0002.exr"),
        )
        cam_npz = np.load(scene_dir / "cameras.npz")
        poses = torch.from_numpy(cam_npz.f.pose).float()
        K = torch.from_numpy(cam_npz.f.K).float()

        # Filter out views that are close together
        kf_idx = [0]
        last_kf_pose = poses[0]
        for i in range(1, len(poses)):
            tdist = torch.norm(poses[i, :3, 3] - last_kf_pose[:3, 3])
            if tdist > _tdist:
                kf_idx.append(i)
                last_kf_pose = poses[i]
        if _testing and _num_points < 100000:
            kf_idx = np.array([1,2,3,4,5])
        else:
            kf_idx = np.array(kf_idx)

        depth_img_files = [file for i, file in enumerate(depth_img_files) if i in kf_idx]
        rgb_img_files = [file for i, file in enumerate(rgb_img_files) if i in kf_idx]
        self.poses = poses[kf_idx]
        self.K = K[kf_idx]
        self.num_views = len(depth_img_files)
        # Read depth images
        self.depth_imgs = torch.empty((0, 480, 640), dtype=torch.float32, device=_device)
        for depth_img_paths in depth_img_files:
            depth_img = torch.from_numpy(read_depth_exr_file(str(depth_img_paths))).to(_device)
            self.depth_imgs = torch.cat((self.depth_imgs, depth_img.unsqueeze(0)), dim=0)

        # Read RGB images
        transform = torchvision.transforms.ToTensor()
        self.rgb_imgs = torch.empty((0, 3, 480, 640), dtype=torch.uint8, device=_device)
        for i in range(len(rgb_img_files)):
            rgb_img = Image.open(rgb_img_files[i]).convert("RGB")
            rgb_img = rgb_img.resize((640, 480), Image.LANCZOS)
            self.rgb_imgs = torch.cat((self.rgb_imgs, transform(rgb_img).unsqueeze(0).to(_device)), dim=0)

        # Create point cloud
        self.createPointCloud(_output_dir, _output_file, _export)

        # Load point cloud
        pcd = []
        pcd_o3d = o3d.io.read_point_cloud(os.path.join(_output_dir, _output_file))
        pcd.append(np.asarray(pcd_o3d.points))
        self.pcd = torch.from_numpy(np.concatenate(pcd, axis=0).astype(np.float32))

        if _testing:
            # Choose X random points in order to reduce computation time
            pcd_sample_idxs = np.random.choice(self.pcd.shape[0], num_points, replace=False)
            self.pcd = self.pcd[pcd_sample_idxs,:]
        
        pcd_processed = []
        # Project all points into all camera views first
        for view_idx in tqdm.trange(self.num_views, desc="Projecting points into all camera views", colour="green"):
            pcd_processed.append(getUVDepthCoordinates(self.poses[view_idx], self.K[view_idx], self.pcd))
        self.pcd_processed = torch.stack(pcd_processed, dim=0)

    def __len__(self):
        return self.pcd_processed.shape[-1]

    # @profile
    def __getitem__(self, idx):
        '''
        Return info for a single point
        @return the point
        @return color at point
        @return patches for a single point
        '''

        # Get views and uv coordinates for that point
        views, uv_coords, point = self.getViewsForOnePoint(idx)

        if len(views) == 0:
            return torch.tensor([], dtype=torch.int64, device=self.device), torch.tensor([], dtype=torch.float32, device=self.device), torch.tensor([], dtype=torch.float32, device=self.device)
            
        # Get patches and colors for that point
        patches, colors = self.getPatchesAndColors(views, uv_coords)
        
        return point, colors, patches

    def getPatchesAndColors(self, _view_visible, _uv_coords):
        # Get correct patch regions
        crop_regions = torch.empty((0,4),dtype=torch.int32, device=self.device)
        colors = torch.empty((0,3), dtype=torch.float32, device=self.device)
        for u,v in _uv_coords:
            crop_regions = torch.cat((crop_regions, torch.tensor([v-self.patch_size[1]//2,
                                                                  u-self.patch_size[0]//2,
                                                                  self.patch_size[1],
                                                                  self.patch_size[0]], dtype=torch.int32, device=self.device).unsqueeze(0)),
                                                                  dim=0)
            colors = torch.cat((colors, self.getColorAtPixel(_view_visible, _uv_coords)), dim=0)
        
        # Take patches from image and preprocess
        patches = torch.empty((0,3,self.preprocessed_tensor_size[0],
                               self.preprocessed_tensor_size[1]),device=self.device)
        for idx, view in enumerate(_view_visible):
            img_crop = torchvision.transforms.functional.crop(self.rgb_imgs[view], crop_regions[idx][0], crop_regions[idx][1],
                                                              crop_regions[idx][2], crop_regions[idx][3])
            
            img = self.model.preprocessFunction(img_crop).unsqueeze(0)
            patches = torch.cat((patches, img), dim=0)

        return patches, torch.mean(colors, dim=0).unsqueeze(0)
    
    def getColorAtPixel(self, _view_visible, _uv_coords):
        """
        @param _view_visible: views that are un-occluded for that point.
        @param _uv: uv coordinates
        @return: color at pixel
        """
        colors_l = torch.empty((0,3), dtype=torch.float32, device=self.device)
        colors = torch.empty((0,3), dtype=torch.float32, device=self.device)
        for i, view in enumerate(_view_visible):
            img = self.rgb_imgs[view]
            u, v = _uv_coords[0]
            if u < 0 or v < 0:
                continue
            else:
                colors = torch.cat((colors, img[:,v,u].unsqueeze(0)), dim=0)
        colors_l = torch.cat((colors_l, torch.mean(colors, dim=0).unsqueeze(0)), dim=0)

        return colors_l
    
    def getViewsForOnePoint(self, idx):
        """
        @return view_visible: views that are un-occluded for that point.
        @return uv_coords: uv coordinates for that point in the image
        @return points: xyz coordinates for that point in the point cloud
        """
        view_visible_mask = torch.full((self.num_views,), False, dtype=torch.bool).to(self.device)
        view_visible = torch.arange(self.num_views).to(self.device)
        uv_coords = torch.empty((0,2), dtype=torch.int64).to(self.device)
        points = torch.empty((0,3), dtype=torch.float32).to(self.device)
        
        for view_idx in range(self.num_views):
            # Get uv coordinates and check if in image
            u, v, depth = self.pcd_processed[view_idx][:,idx]
            u = int(max(min(u, ALOT), -ALOT))
            v = int(max(min(v, ALOT), -ALOT))
            depth = depth.item()
            # u, v, depth = int(u), int(v), depth.item()

            # extra check to see if point is within bounds of image
            if ((u >= self.patch_size[0]) and (u < self.imwidth - self.patch_size[0]) and
                (v >= self.patch_size[1]) and (v < self.imheight - self.patch_size[1]) and
                (depth >= 0)):
                depth_img = self.depth_imgs[view_idx]
                ground_truth_depth = depth_img[v,u].item()
                if abs(ground_truth_depth - depth) < self.epsilon:
                    view_visible_mask[view_idx] = True
                    # view_visible = torch.cat((view_visible, torch.tensor([view_idx]).to(self.device)))
                    valid_uv = torch.tensor([u,v]).to(self.device)
                    uv_coords = torch.cat((uv_coords, valid_uv.unsqueeze(0)), dim=0)
            else:
                continue

        # If the point isn't visible from any POV, return empty tensors.
        if torch.sum(view_visible_mask).item() == 0:
            return torch.tensor([], dtype=torch.int64, device=self.device), uv_coords, points
        else:
            points = self.pcd[idx,:].unsqueeze(0).to(self.device)

        return view_visible[view_visible_mask], uv_coords, points

    def createPointCloud(self, _output_dir, _output_file, export = False):
        """
        Iterate over all the camera views and create a point cloud with color from RGB images.
        @param _depth_img_files: list of depth images
        @param _rgb_img_files: list of RGB images
        @param _pose: list of camera poses
        @param _K: list of camera intrinsics
        @return: point cloud and point cloud colors
        """
        
        u = np.arange(self.imwidth)
        v = np.arange(self.imheight)
        uu, vv = np.meshgrid(u, v)
        uv = np.c_[uu.flatten(), vv.flatten()]

        # Iterate over all camera poses
        pcd = []
        pcd_colors = []
        pcd_views = []
        transform = torchvision.transforms.ToPILImage()
        for view_idx in tqdm.trange(self.num_views, desc="Creating point cloud", colour="green"):            
            rgb_img = np.array(transform(self.rgb_imgs[view_idx].cpu()))
            depth_array = self.depth_imgs[view_idx].reshape(-1)
            good_pixs = (depth_array > 0).clone().detach()
            good_pixs = np.array(good_pixs.cpu())

            # Get pixels with image coordinates?
            pix_vecs_depth = np.linalg.inv(self.K[view_idx]) @ np.c_[uv, np.ones((uv.shape[0], 1))].T
            depth_array = np.array(self.depth_imgs[view_idx].reshape(-1).cpu())
            xyz_cam = pix_vecs_depth * depth_array
            xyz_cam = xyz_cam[:,good_pixs]
            rgb_img = rgb_img.reshape(-1,3)
            rgb_img = rgb_img[good_pixs,:]

            # Rotate matrix * xyz_cam + translation
            pcd.append((self.poses[view_idx][:3, :3] @ xyz_cam + self.poses[view_idx][:3, 3:4]).T)
            pcd_views.append(np.full((len(pcd[0]),1), view_idx))
            pcd_colors.append(rgb_img)

        pcd = np.concatenate(pcd, axis=0)
        pcd_colors = np.concatenate(pcd_colors, axis=0)
        pcd_views = np.concatenate(pcd_views, axis=0)

        if export:
            _ = trimesh.PointCloud(pcd, colors=pcd_colors).export(os.path.join(_output_dir, _output_file))
    
        return pcd, pcd_colors, pcd_views

class ScannetPCD(torch.utils.data.Dataset):
    def __init__(self, _model, _output_dir, _output_file, _epsilon, _tdist=0.1, 
                _device="cuda", _testing=False, _export=False,
                _num_points=50000, _patch_size=(128,128)):
        # if testing use _tdist=0.35
        print("Loading ScanNet point cloud dataset...")
        self.imheight = 480
        self.imwidth = 640
        self.patch_size = _patch_size
        self.epsilon = _epsilon
        self.device = _device
        self.model = _model
        self.preprocessed_tensor_size = torch.tensor([224,224], device=self.device)
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
        if _testing and _num_points < 100000:
            kf_idx = np.array([1,2,3,4,5])
        else:
            kf_idx = np.array(kf_idx)

        depth_img_files = depth_img_files[kf_idx]
        rgb_img_files = rgb_img_files[kf_idx]
        self.poses = poses[kf_idx]
        self.num_views = len(depth_img_files)

        # Process depth images
        self.depth_imgs = torch.empty((0, 480, 640), dtype=torch.float32, device=_device)
        for i in range(len(depth_img_files)):
            depth_img = cv2.imread(str(depth_img_files[i]), cv2.IMREAD_ANYDEPTH)
            depth_img = torch.from_numpy(depth_img.astype(np.float32)).to(_device) / 1000
            self.depth_imgs = torch.cat((self.depth_imgs, depth_img.unsqueeze(0)), dim=0)
        
        # Process RGB images
        transform = torchvision.transforms.ToTensor()
        self.rgb_imgs = torch.empty((0, 3, 480, 640), dtype=torch.uint8, device=_device)
        for i in range(len(rgb_img_files)):
            rgb_img = Image.open(rgb_img_files[i]).convert("RGB")
            rgb_img = rgb_img.resize((640, 480), Image.LANCZOS)
            self.rgb_imgs = torch.cat((self.rgb_imgs, transform(rgb_img).unsqueeze(0).to(_device)), dim=0)

        self.poses = poses[kf_idx]
        self.K = K_depth

        # Create point cloud
        self.createPointCloud(_output_dir, _output_file, _export)

        # Load point cloud
        pcd = []
        pcd_o3d = o3d.io.read_point_cloud(os.path.join(_output_dir, _output_file))
        pcd.append(np.asarray(pcd_o3d.points))
        self.pcd = torch.from_numpy(np.concatenate(pcd, axis=0).astype(np.float32))

        if _testing:
            # Choose X random points in order to reduce computation time
            pcd_sample_idxs = np.random.choice(self.pcd.shape[0], num_points, replace=False)
            self.pcd = self.pcd[pcd_sample_idxs,:]
        
        pcd_processed = []
        # Project all points into all camera views first
        for view_idx in tqdm.trange(self.num_views, desc="Projecting points into all camera views", colour="green"):
            pcd_processed.append(getUVDepthCoordinates(self.poses[view_idx], self.K, self.pcd))
        self.pcd_processed = torch.stack(pcd_processed, dim=0)

    def __len__(self):
        return self.pcd_processed.shape[-1]

    def __getitem__(self, idx):
        '''
        Return info for a single point
        @return the point
        @return color at point
        @return patches for a single point
        '''

        # Get views and uv coordinates for that point
        views, uv_coords, point = self.getViewsForOnePoint(idx)

        if len(views) == 0:
            return torch.tensor([], dtype=torch.int64, device=self.device), torch.tensor([], dtype=torch.float32, device=self.device), torch.tensor([], dtype=torch.float32, device=self.device)
            
        # Get patches and colors for that point
        patches, colors = self.getPatchesAndColors(views, uv_coords)
        
        return point, colors, patches

    def getPatchesAndColors(self, _view_visible, _uv_coords):
        # Get correct patch regions
        crop_regions = torch.empty((0,4),dtype=torch.int32, device=self.device)
        colors = torch.empty((0,3), dtype=torch.float32, device=self.device)
        for u,v in _uv_coords:
            crop_regions = torch.cat((crop_regions, torch.tensor([v-self.patch_size[1]//2,
                                                                  u-self.patch_size[0]//2,
                                                                  self.patch_size[1],
                                                                  self.patch_size[0]], dtype=torch.int32, device=self.device).unsqueeze(0)),
                                                                  dim=0)
            colors = torch.cat((colors, self.getColorAtPixel(_view_visible, _uv_coords)), dim=0)
        
        # Take patches from image and preprocess
        patches = torch.empty((0,3,self.preprocessed_tensor_size[0],
                               self.preprocessed_tensor_size[1]),device=self.device)
        for idx, view in enumerate(_view_visible):
            img_crop = torchvision.transforms.functional.crop(self.rgb_imgs[view], crop_regions[idx][0], crop_regions[idx][1],
                                                              crop_regions[idx][2], crop_regions[idx][3])
            
            img = self.model.preprocessFunction(img_crop).unsqueeze(0)
            patches = torch.cat((patches, img), dim=0)

        return patches, torch.mean(colors, dim=0).unsqueeze(0)
    
    def getColorAtPixel(self, _view_visible, _uv_coords):
        """
        @param _view_visible: views that are un-occluded for that point.
        @param _uv: uv coordinates
        @return: color at pixel
        """
        colors_l = torch.empty((0,3), dtype=torch.float32, device=self.device)
        colors = torch.empty((0,3), dtype=torch.float32, device=self.device)
        for i, view in enumerate(_view_visible):
            img = self.rgb_imgs[view]
            u, v = _uv_coords[0]
            if u < 0 or v < 0:
                continue
            else:
                colors = torch.cat((colors, img[:,v,u].unsqueeze(0)), dim=0)
        colors_l = torch.cat((colors_l, torch.mean(colors, dim=0).unsqueeze(0)), dim=0)

        return colors_l
    
    def getViewsForOnePoint(self, idx):
        """
        @return view_visible: views that are un-occluded for that point.
        @return uv_coords: uv coordinates for that point in the image
        @return points: xyz coordinates for that point in the point cloud
        """
        view_visible_mask = torch.full((self.num_views,), False, dtype=torch.bool).to(self.device)
        view_visible = torch.arange(self.num_views).to(self.device)
        uv_coords = torch.empty((0,2), dtype=torch.int64).to(self.device)
        points = torch.empty((0,3), dtype=torch.float32).to(self.device)
        
        for view_idx in range(self.num_views):
            # Get uv coordinates and check if in image
            u, v, depth = self.pcd_processed[view_idx][:,idx]
            if u == float('inf'):
                u = -1
            if v == float('inf'):
                v = -1
            u = int(u)
            v = int(v)
            depth = depth.item()
            # u, v, depth = int(u), int(v), depth.item()

            # extra check to see if point is within bounds of image
            if ((u >= self.patch_size[0]) and (u < self.imwidth - self.patch_size[0]) and
                (v >= self.patch_size[1]) and (v < self.imheight - self.patch_size[1]) and
                (depth >= 0)):
                depth_img = self.depth_imgs[view_idx]
                ground_truth_depth = depth_img[v,u].item()
                if abs(ground_truth_depth - depth) < self.epsilon:
                    view_visible_mask[view_idx] = True
                    # view_visible = torch.cat((view_visible, torch.tensor([view_idx]).to(self.device)))
                    valid_uv = torch.tensor([u,v]).to(self.device)
                    uv_coords = torch.cat((uv_coords, valid_uv.unsqueeze(0)), dim=0)
            else:
                continue

        # If the point isn't visible from any POV, return empty tensors.
        if torch.sum(view_visible_mask).item() == 0:
            return torch.tensor([], dtype=torch.int64, device=self.device), uv_coords, points
        else:
            points = self.pcd[idx,:].unsqueeze(0).to(self.device)

        return view_visible[view_visible_mask], uv_coords, points

    def createPointCloud(self, _output_dir, _output_file, export = False):
        """
        Iterate over all the camera views and create a point cloud with color from RGB images.
        @param _depth_img_files: list of depth images
        @param _rgb_img_files: list of RGB images
        @param _pose: list of camera poses
        @param _K: list of camera intrinsics
        @return: point cloud and point cloud colors
        """
        u = np.arange(self.imwidth)
        v = np.arange(self.imheight)
        uu, vv = np.meshgrid(u, v)
        uv = np.c_[uu.flatten(), vv.flatten()]

        # Iterate over all camera poses
        pcd = []
        pcd_colors = []
        pcd_views = []
        transform = torchvision.transforms.ToPILImage()
        for view_idx in tqdm.trange(self.num_views, desc="Creating point cloud", colour="green"):            
            rgb_img = np.array(transform(self.rgb_imgs[view_idx].cpu()))
            depth_array = self.depth_imgs[view_idx].reshape(-1)
            good_pixs = (depth_array > 0).clone().detach()
            good_pixs = np.array(good_pixs.cpu())

            # Get pixels with image coordinates?
            pix_vecs_depth = np.linalg.inv(self.K) @ np.c_[uv, np.ones((uv.shape[0], 1))].T
            depth_array = np.array(self.depth_imgs[view_idx].reshape(-1).cpu())
            xyz_cam = pix_vecs_depth * depth_array
            xyz_cam = xyz_cam[:,good_pixs]
            rgb_img = rgb_img.reshape(-1,3)
            rgb_img = rgb_img[good_pixs,:]

            # Rotate matrix * xyz_cam + translation
            pcd.append((self.poses[view_idx][:3, :3] @ xyz_cam + self.poses[view_idx][:3, 3:4]).T)
            pcd_views.append(np.full((len(pcd[0]),1), view_idx))
            pcd_colors.append(rgb_img)

        pcd = np.concatenate(pcd, axis=0)
        pcd_colors = np.concatenate(pcd_colors, axis=0)
        pcd_views = np.concatenate(pcd_views, axis=0)

        if export:
            _ = trimesh.PointCloud(pcd, colors=pcd_colors).export(os.path.join(_output_dir, _output_file))
    
        return pcd, pcd_colors, pcd_views

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
    world_to_cam = torch.linalg.inv(_pose)
    xyz_cam = (world_to_cam[:3, :3] @ _pcd.T + world_to_cam[:3, 3:4])

    # Matrix multiply the xyz_cam by the camera intrinsics to get uv
    uv = _K @ xyz_cam
    uv_z = uv / uv[2, :]

    # Round all uv values
    uv_z = torch.round(uv_z)
    
    # Replace the z row with depth values
    uv_z[2,:] = uv[2,:]
    
    return uv_z

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