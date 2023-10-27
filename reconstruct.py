import pathlib

import matplotlib.pyplot as plt
import numpy as np
import skimage.measure
import tqdm
import trimesh
import Imath
import OpenEXR
import torch
import open3d as o3d
import open_clip

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2

OUTPUT_DIR = "./Reconstruction/"

def main():
    print(pathlib.Path.cwd())
    scene_dir = pathlib.Path("./Reconstruction/scene_0000")

    # Load files
    depth_img_files = sorted(scene_dir.glob("depth/*/img0002.exr"))
    rgb_img_files = sorted(scene_dir.glob("rgb/*/img0002.jpg"))
    cam_npz = np.load(scene_dir / "cameras.npz")
    pose = cam_npz.f.pose # camera-to-world
    K = cam_npz.f.K

    imheight, imwidth, _ = cv2.imread(str(rgb_img_files[0])).shape

    ################################################################################
    # Back-project point cloud

    # pcd, pcd_colors = createPointCloud(depth_img_files, rgb_img_files, pose, K, export=False)

    ################################################################################
    # Project the point cloud back onto the RGB image

    ## PASS 1
    # 1. Load point clouds
    # 2. Test for occlusion and get points from one camera view
    # 3. Run clip over a point in the camera view
    # 4. Backproject clip feature into point
    # 5. Repeat 3-4 for all points in the camera view

    ## PASS 2
    # 1. Load point clouds
    # 2. Iterate over a single point in the point cloud
    # 3. Find out what point of views the point is visible in
    # 4. Run clip over that point from every visible camera view
    # 5. Backproject clip feature into point
    # 6. Average clip features for that point
    # 7. Repeat 2-6 for all points in the point cloud

    for view_idx in tqdm.trange(len(pose)):
        pcd_colors = []
        pcd = []
        depth_img = read_depth_exr_file(str(depth_img_files[view_idx]))
        rgb_img = cv2.imread(str(rgb_img_files[view_idx]))[..., [2, 1, 0]]

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

    # TODO: run clip over each pixel that is projected into each view



    exit()

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

    for view_idx in tqdm.trange(len(_pose)):

        depth_img = read_depth_exr_file(str(_depth_img_files[view_idx]))
        rgb_img = cv2.imread(str(_rgb_img_files[view_idx]))[..., [2, 1, 0]]
        
        # Get pixels with image coordinates?
        pix_vecs = np.linalg.inv(_K[view_idx]) @ np.c_[uv, np.ones((uv.shape[0], 1))].T
        xyz_cam = pix_vecs * depth_img.flatten()

        # Rotate matrix * xyz_cam + translation
        pcd.append((_pose[view_idx, :3, :3] @ xyz_cam + _pose[view_idx, :3, 3:4]).T)
        pcd_colors.append(rgb_img.reshape(-1, 3))

    pcd = np.concatenate(pcd, axis=0)
    pcd_colors = np.concatenate(pcd_colors, axis=0)

    if export:
        _ = trimesh.PointCloud(pcd, colors=pcd_colors).export(os.path.join(OUTPUT_DIR,"pcd.ply"))
   
    return pcd, pcd_colors


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