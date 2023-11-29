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

class ScannetPCD(torch.utils.data.Dataset):
    def __init__(self, _tdist=0.1, _device="cuda", _testing=False):
        # if testing use _tdist=0.35
        self.device = _device
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
        if _testing:
            kf_idx = np.array([1,2,3,4,5])
        else:
            kf_idx = np.array(kf_idx)
        depth_img_files = depth_img_files[kf_idx]
        rgb_img_files = rgb_img_files[kf_idx]
        self.poses = poses[kf_idx]

        # Process depth images
        self.depth_imgs = torch.empty((0, 480, 640), dtype=torch.float32, device=_device)
        for i in range(len(depth_img_files)):
            depth_img = cv2.imread(str(depth_img_files[i]), cv2.IMREAD_ANYDEPTH)
            depth_img = torch.from_numpy(depth_img.astype(np.float32)).to(_device) / 1000
            self.depth_imgs = torch.cat((self.depth_imgs, depth_img.unsqueeze(0)), dim=0)
        
        # # Process RGB images
        # transform = torchvision.transforms.ToTensor()
        # self.rgb_imgs = torch.empty((0, 3, 480, 640), dtype=torch.uint8, device=_device)
        # for i in range(len(rgb_img_files)):
        #     rgb_img = Image.open(rgb_img_files[i]).convert("RGB")
        #     rgb_img = rgb_img.resize((640, 480), Image.LANCZOS)
        #     self.rgb_imgs = torch.cat((self.rgb_imgs, transform(rgb_img).unsqueeze(0).to(_device)), dim=0)

        # Process RGB images
        transform = torchvision.transforms.ToTensor()
        self.rgb_imgs = torch.empty((0, 3, 480, 640), dtype=torch.uint8, device=_device)
        for i in range(len(rgb_img_files)):
            rgb_img = Image.open(rgb_img_files[i]).convert("RGB")
            rgb_img = rgb_img.resize((640, 480), Image.LANCZOS)
            self.rgb_imgs = torch.cat((self.rgb_imgs, transform(rgb_img).unsqueeze(0).to(_device)), dim=0)


        self.poses = poses[kf_idx]
        self.K = K_depth

    def __len__(self):
        return len(self.depth_imgs)
    
    def __getitem__(self, idx):
        pose = self.poses[idx]
        depth_img = self.depth_imgs[idx]
        rgb_img = self.rgb_imgs[idx]

        depth_array = depth_img.reshape(-1)
        good_pixs = (depth_array > 0).clone().detach()
        
        return depth_img, rgb_img, pose, self.K, good_pixs

    def createScannnetPointCloud(self, _output_dir, _output_file, export = False):
        """
        Iterate over all the camera views and create a point cloud with color from RGB images.
        @param _depth_img_files: list of depth images
        @param _rgb_img_files: list of RGB images
        @param _pose: list of camera poses
        @param _K: list of camera intrinsics
        @return: point cloud and point cloud colors
        """
        depth_img, _, _, _, _ = self[0]

        imheight, imwidth = depth_img.shape
        u = np.arange(imwidth)
        v = np.arange(imheight)
        uu, vv = np.meshgrid(u, v)
        uv = np.c_[uu.flatten(), vv.flatten()]

        # Iterate over all camera poses
        pcd = []
        pcd_colors = []
        pcd_views = []
        transform = torchvision.transforms.ToPILImage()
        for view_idx in tqdm.trange(len(self)):
            depth_img, rgb_img, pose, K, good_pixs = self[view_idx]
            rgb_img = np.array(transform(rgb_img.cpu()))
            good_pixs = np.array(good_pixs.cpu())

            # Get pixels with image coordinates?
            pix_vecs_depth = np.linalg.inv(K) @ np.c_[uv, np.ones((uv.shape[0], 1))].T
            depth_array = np.array(depth_img.reshape(-1).cpu())
            xyz_cam = pix_vecs_depth * depth_array
            xyz_cam = xyz_cam[:,good_pixs]
            rgb_img = rgb_img.reshape(-1,3)
            rgb_img = rgb_img[good_pixs,:]

            # Rotate matrix * xyz_cam + translation
            pcd.append((pose[:3, :3] @ xyz_cam + pose[:3, 3:4]).T)
            pcd_views.append(np.full((len(pcd[0]),1), view_idx))
            pcd_colors.append(rgb_img)

        pcd = np.concatenate(pcd, axis=0)
        pcd_colors = np.concatenate(pcd_colors, axis=0)
        pcd_views = np.concatenate(pcd_views, axis=0)

        if export:
            _ = trimesh.PointCloud(pcd, colors=pcd_colors).export(os.path.join(_output_dir, _output_file))
    
        return pcd, pcd_colors, pcd_views

class PatchesBlockBuilder(torch.utils.data.Dataset):
    """
    This class will give blocks of views corresponding to each point
    in the point cloud.
    """
    def __init__(self, _pcd_dataset, _pcd_path, _output_dir, _output_file, 
                 _device, _block_size,_patch_size=(128,128), 
                 _testing=True, _num_points=2000):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.pcd_dataset = _pcd_dataset
        self.pcd_path = _pcd_path
        self.output_file = _output_file
        # self.device = _device
        self.device = "cpu"
        self.patch_size = _patch_size
        self.testing = _testing
        self.num_points = _num_points
        
        self.block_size = _block_size
        self.curr_pcd_idx = 0
        self.num_views = len(self.pcd_dataset.depth_imgs)

        depth_img = self.pcd_dataset[0][0]
        self.imheight, self.imwidth = depth_img.shape
        self.preprocessed_tensor_size = torch.tensor([224,224], device=self.device)
        
        pcd = []
        pcd_o3d = o3d.io.read_point_cloud(_pcd_path)
        pcd.append(np.asarray(pcd_o3d.points))
        self.pcd = torch.from_numpy(np.concatenate(pcd, axis=0)).to(torch.float32).to(self.device)
        
        if self.testing:
            # Choose X random points in order to reduce computation time
            pcd_idxs = torch.randperm(self.pcd.shape[0])[:_num_points]
            # temp = torch.tensor([2000,2800,3000])
            # pcd_idxs = torch.cat((temp,pcd_idxs), dim=0)
            self.pcd = self.pcd[pcd_idxs,:]
            self.epsilon = 0.05
        else:
            self.epsilon = 0.001 # 0.001 = 0.1mm resolution for testing occlusion

        # Read in point cloud, project through each view, and append to list
        self.pcd_processed = []
        for i in range(len(self.pcd_dataset.depth_imgs)):
            _, _, pose, K, _ = self.pcd_dataset[i]
            pcd_processed = self.getUVDepthCoordinates(pose, K, self.pcd)
            # TODO: move <=0 depth check here?
            # pcd_keep = (pcd_processed[2,:] > 0).clone().detach()
            # self.pcd_processed.append(pcd_processed[:,pcd_keep])
            self.pcd_processed.append(pcd_processed)
        self.pcd_processed = torch.stack(self.pcd_processed).to(self.device)
        
        self.blocks = self.buildBlocks()

    def __len__(self):
        return len(self.blocks)
    
    def __getitem__(self, idx):
        """
        Block of patches.
        """
        return self.blocks[idx]

    def buildBlocks(self):
        """
        Builds all blocks.
        """
        blocks = []
        num_points = self.pcd_processed.shape[2]
        with tqdm.tqdm(total=num_points) as pbar:
            while(self.curr_pcd_idx < num_points):
                # blocks = torch.cat((blocks, self.buildOneBlock()), dim=0)
                start_idx = self.curr_pcd_idx
                blocks.append(self.buildOneBlock())
                end_idx = self.curr_pcd_idx
                pbar.update(end_idx-start_idx)
            pbar.close()
        # Remove last block if it is empty
        if blocks[-1]["idxs"][0] == -1:
            blocks = blocks[:-1]
        return blocks

    def buildOneBlock(self):
        """
        Builds one block of _block_size. Pads with -1.
        @return dict "views": tensor of views correspond to that point
                     "idxs": tensor of idxs corresponding to where that point ends
                     "uv_coords": tensor of uv coordinates respective to that view
                     "colors": tensor of colors corresponding to that point
        """
        idxs = torch.tensor([], dtype=torch.int64, device=self.device)
        views_l = torch.tensor([], dtype=torch.int64, device=self.device)
        uv_coords_l = torch.empty((0,2), dtype=torch.int64, device=self.device)
        # patches_l = torch.empty((0,3,self.preprocessed_tensor_size[0],self.preprocessed_tensor_size[1]), dtype=torch.int64, device=self.device)
        points_l = torch.empty((0,3), dtype=torch.float32, device=self.device)
        colors_l = torch.empty((0,3), dtype=torch.float32, device=self.device)
        prev_idx = 0
        while(len(views_l)<self.block_size):
            # if no more points, pad with empty
            if self.curr_pcd_idx >= self.pcd_processed.shape[2]:
                num_pads = self.block_size - len(views_l)
                idxs = torch.cat((idxs, torch.tensor([-1],device=self.device)), dim=0)
                while(num_pads != 0):
                    views_l = torch.cat((views_l, torch.tensor([-1],device=self.device)), dim=0)
                    num_pads -= 1
                continue

            views, uv_coords, points = self.getViewsForOnePoint(self.curr_pcd_idx)
            # if no views, skip
            if views.shape[0] == 0:
                self.curr_pcd_idx += 1
                continue
            # if too many views, pad with empty
            if len(views_l) + len(views) > self.block_size:
                num_pads = self.block_size - len(views_l)
                idxs = torch.cat((idxs, torch.tensor([-1],device=self.device)), dim=0)
                while(num_pads != 0):
                    # TODO: can i remove the views padding?
                    views_l = torch.cat((views_l, torch.tensor([-1],device=self.device)), dim=0)
                    num_pads -= 1
            else:
                prev_idx += len(views)
                idxs = torch.cat((idxs, torch.tensor([prev_idx],device=self.device)), dim=0)
                self.curr_pcd_idx += 1
                views_l = torch.cat((views_l, views), dim=0)
                uv_coords_l = torch.cat((uv_coords_l, uv_coords), dim=0)
                # patches_l = torch.cat((patches_l, self.getPatches(views,uv_coords)), dim=0)
                points_l = torch.cat((points_l, points), dim=0)
                colors_l = torch.cat((colors_l, self.getColorAtPixel(views,uv_coords)), dim=0)
        block = {}
        block["views"] = views_l # may have to do views instead of patches
        block["idxs"] = idxs
        # block["patches"] = patches_l
        block["uv_coords"] = uv_coords_l
        block["points"] = points_l
        block["colors"] = colors_l
        
        return block
    
    def getViewsForOnePoint(self, idx):
        """
        @return view_visible: views that are un-occluded for that point.
        @return uv_coords: uv coordinates for that point in the image
        @return points: xyz coordinates for that point in the point cloud
        """
        view_visible_mask = torch.full((self.num_views,), False, dtype=torch.bool, device=self.device)
        view_visible = torch.arange(self.num_views).to(self.device)
        uv_coords = torch.empty((0,2), dtype=torch.int64).to(self.device)
        points = torch.empty((0,3), dtype=torch.float32).to(self.device)
        
        for view_idx in range(self.num_views):
            # Get uv coordinates and check if in image
            u, v, depth = self.pcd_processed[view_idx][:,idx]
            u, v, depth = int(u), int(v), depth.item()

            # extra check to see if point is within bounds of image
            if ((u >= self.patch_size[0]) and (u < self.imwidth - self.patch_size[0]) and
                (v >= self.patch_size[1]) and (v < self.imheight - self.patch_size[1]) and
                (depth >= 0)):
                depth_img = self.pcd_dataset.depth_imgs[view_idx]
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
            points = self.pcd[idx,:].unsqueeze(0)

        return view_visible[view_visible_mask], uv_coords, points
        
    def getPatches(self, _view_visible, _uv_coords):
        """
        Gets PREPROCESSED patches for a point.
        @param _view_visible: views that are un-occluded for that point.
        @param _uv_coords: uv coordinates for that point in the image.
        @return patches: patches corresponding to the point.
        """
        imgs = []
        for idx in _view_visible:
            imgs.append(self.pcd_dataset.rgb_imgs[idx])

        # Get correct patch regions
        crop_regions = torch.empty((0,4),dtype=torch.int32, device=self.device)
        for u,v in _uv_coords:
            crop_regions = torch.cat((crop_regions, torch.tensor([u-self.patch_size[0]//2, 
                                                                  v-self.patch_size[1]//2,
                                                                  self.patch_size[1],
                                                                  self.patch_size[0]], dtype=torch.int32, device=self.device).unsqueeze(0)),
                                                                  dim=0)

        # Take patches from image and preprocess
        toPILImage = torchvision.transforms.ToPILImage()
        patches = torch.empty((0,3,self.preprocessed_tensor_size[0],
                               self.preprocessed_tensor_size[1]),device=self.device)
        for idx, image in enumerate(imgs):
            img_crop = torchvision.transforms.functional.crop(image, crop_regions[idx][0], crop_regions[idx][1],
                                                              crop_regions[idx][2], crop_regions[idx][3])
            # self.preprocess <--optimize?
            # img = torch.empty((3,224,224), dtype=torch.float32, device=self.device).unsqueeze(0)
            img = self.preprocess(toPILImage(img_crop)).to(self.device).unsqueeze(0)
            patches = torch.cat((patches, img), dim=0)
        
        return patches

    def getColorAtPixel(self, _views, _uv_coords):
        """
        @param _img: RGB image
        @param _uv: uv coordinates
        @return: color at pixel
        """
        color = torch.empty((0,3), dtype=torch.float32, device=self.device)
        for idx, view in enumerate(_views):
            img = self.pcd_dataset.rgb_imgs[view]
            u, v = _uv_coords[idx]
            if u < 0 or v < 0:
                continue
            else:
                color = torch.cat((color, img[:,v,u].unsqueeze(0).cpu()), dim=0)
                
        # for idx, img in enumerate(_imgs):
        #     u, v = _uv_coords[idx]
        #     if u < 0 or v < 0:
        #         continue
        #     else:
        #         color = torch.cat((color, img[:,v,u].unsqueeze(0)), dim=0)
        #         # color.append(img[:,u,v])
        #         # color.append(img.getpixel((u,v)))
        
        mean = torch.mean(color, dim=0)
        return mean.unsqueeze(0)
        # return np.mean(color, axis=0).astype(np.uint8)

    def getUVDepthCoordinates(self, _pose, _K, _pcd):
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
