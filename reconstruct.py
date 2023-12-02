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
from PointCloudDataset import ScannetPCD, SyntheticPCD
import torch
import gc
import time
import torchvision
from line_profiler import LineProfiler
from multiprocessing import Process, Queue

prof = LineProfiler()

## DO THESE AFTER MAKING DATALOADING EFFICIENT
# TODO: try better CLIP model?

# @profile
def main():
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    OUTPUT_DIR = "./Reconstruction/output/synthetic" # Where to save/load files
    NPZ_FILE = "pcd_with_colors.npz" # NPZ with img features to generate/load
    NPZ_FEATURES_FILE = "img_features.npz" # NPZ with img features to generate/load
    PCD_FILE = "pcd.ply" # Point cloud file to generate/load
    CLIP_PATCH_SIZE = (128,128)
    # QUERY = "where are the chairs?"
    QUERY = "what can I sit on?"
    # QUERY = "¿dónde están mis zapatos?"
    # QUERY = "bed"
    PERFORMING_QUERY = True
    TESTING = True # enable to run on smaller subset of points
    SYNTHETIC = True
    BATCH_SIZE = 80 # 128 is pretty good
    NUM_WORKERS = 16 # 16 is bad
    DISABLE_PROGRESS_BAR = False
    GPU_THROUGHPUT_SIZE = 200
    DATA_BUFFER_SIZE = 512 # 256 is pretty good

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
        dataset = SyntheticPCD(_model=clip_model, _output_dir=OUTPUT_DIR, _output_file=PCD_FILE,
                               _epsilon=EPSILON, _tdist=0.1, _device="cpu", _testing=TESTING,
                               _export=True, _patch_size=(128,128))
    else:
        dataset = ScannetPCD(_model=clip_model, _output_dir=OUTPUT_DIR, _output_file=PCD_FILE,
                             _epsilon=EPSILON, _tdist=0.35, _device="cpu", _testing=TESTING,
                             _export=True, _patch_size=(128,128))
    
    qq = Queue()
    readerProc = startReaderProc(qq, clip_model, _save_path=os.path.join(OUTPUT_DIR, NPZ_FEATURES_FILE))

    ################################################################################
    # Back-project point cloud

    dataset.createPointCloud(OUTPUT_DIR, PCD_FILE, export=True)
    
    ################################################################################
    # Project the point cloud back onto the RGB image

    my_collator = MyCollator(device="cpu", gpu_throughput_size=GPU_THROUGHPUT_SIZE)
    pointsGenerator = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                                  shuffle=False, num_workers=NUM_WORKERS,
                                                  collate_fn=my_collator)
    good_points_count = 0
    patch_count = 0
    time_spent_waiting = 0.0
    new_pcd = torch.empty((0,3), dtype=torch.float64)
    colors_l = torch.empty((0,3), dtype=torch.float32)
    leave = False
    t1 = time.time()
    for point_l, color_l, patches_l, stop_idx_l, gpu_batch_idx_l in tqdm.tqdm(pointsGenerator, desc="Iterating over points",
                                           colour="green", disable=DISABLE_PROGRESS_BAR):
        if point_l.shape[0] != 0:
            good_points_count += 1
            patch_count += patches_l.shape[0]

            new_pcd = torch.cat((new_pcd, point_l.cpu()),dim=0)
            colors_l = torch.cat((colors_l, color_l.cpu()),dim=0)
            # wait if buffer is full
            while qq.qsize() + patches_l.shape[0] >= DATA_BUFFER_SIZE:
                # leave = True
                # break
                time.sleep(0.1)
                time_spent_waiting += 0.1
            qq.put((patches_l, stop_idx_l, gpu_batch_idx_l))
        if leave:
            break
    qq.put((torch.tensor([-1]), torch.tensor([-1])))
    print("Waiting for reader to finish")
    readerProc.join()
    t2 = time.time()
    
    print("Time spent waiting: ", time_spent_waiting)
    print("Successful points: ", good_points_count)
    print("Time to process points: ", t2-t1)
    print("Points per second: ", good_points_count / (t2-t1))
    print("Patches per second: ", patch_count / (t2-t1))

    new_pcd = np.stack(new_pcd.cpu(),axis=0)
    colors_l = np.stack(colors_l.cpu(),axis=0)
    np.savez(os.path.join(OUTPUT_DIR, NPZ_FILE), new_pcd, colors_l)

    # ################################################################################
    # Perform a query

    npzfile = np.load(os.path.join(OUTPUT_DIR,NPZ_FILE))
    npz_feature_file = np.load(os.path.join(OUTPUT_DIR,NPZ_FEATURES_FILE))
    new_pcd = npzfile['arr_0']
    colors_l = npzfile['arr_1']
    img_features_l = npz_feature_file['arr_0']

    if PERFORMING_QUERY:
        print("Performing query")
        similarity_scores = clip_model.runQuery(img_features_l, QUERY).cpu()
        heatmap_colors = [plt.cm.viridis(score)[:3] for score in similarity_scores]
        heatmap_colors = np.array(heatmap_colors)
        cloud = trimesh.PointCloud(vertices=new_pcd, colors=heatmap_colors)
        
        output_file = QUERY.replace(" ","_").replace("?","").replace("¿","") + ".ply"
        cloud.export(os.path.join(OUTPUT_DIR,output_file))
    if TESTING:
        cloud = trimesh.PointCloud(vertices=new_pcd, colors=colors_l)
        cloud.export(os.path.join(OUTPUT_DIR,"color_test.ply"))


def reader_proc(_queue, _model, _save_path):
    # Run CLIP over patches received from queue
    # output the features to CPU on list
    img_features_l = torch.empty((0,512), dtype=torch.float32)
    while True:
        data = _queue.get()
        if data[1][0] == -1:
            break
        patches = data[0].to(_model.device)
        prev_gpu_idx = 0
        prev_stop_idx = 0
        for gpu_batch_idx in data[2]:
            indices = (data[1] == gpu_batch_idx).nonzero(as_tuple=True)
            first_occurrence_index = indices[0][0].item() + 1
            features = _model.getImageFeatureForPatches(patches[prev_gpu_idx:gpu_batch_idx], data[1][prev_stop_idx:first_occurrence_index])
            img_features_l = torch.cat((img_features_l, features.cpu()), dim=0)
            prev_gpu_idx = gpu_batch_idx
            prev_stop_idx = first_occurrence_index
        
    img_features_l = np.stack(img_features_l.cpu(),axis=0)
    np.savez(_save_path, img_features_l)

def startReaderProc(_qq, _model, _save_path):
    readerP = Process(target=reader_proc, args=(_qq,_model,_save_path,))
    readerP.daemon = True
    readerP.start()
    return readerP

class MyCollator(object):
    def __init__(self, **params):
        self.device = params["device"]
        self.gpu_throughput_size = params["gpu_throughput_size"]

    # @profile
    def __call__(self, batch):
        point_l = torch.empty((0,3), dtype=torch.int64, device=self.device)
        color_l = torch.empty((0,3), dtype=torch.float32, device=self.device)
        patches_l = torch.empty((0,3,224,224), dtype=torch.float32, device=self.device)
        stop_idx_l = torch.empty((0,1), dtype=torch.int64, device=self.device)
        gpu_batch_idx_l = torch.empty((0,1), dtype=torch.int64, device=self.device)
        last_gpu_batch_idx = torch.tensor([0], device=self.device)
        prev_stop_idx = torch.tensor([0], device=self.device)
        for point, color, patches in batch:
            point_l = torch.cat((point_l, point), dim=0)
            color_l = torch.cat((color_l, color), dim=0)
            patches_l = torch.cat((patches_l, patches), dim=0)
            prev_stop_idx += patches.shape[0]
            if stop_idx_l.shape[0] == 0:
                stop_idx_l = torch.cat((stop_idx_l, torch.tensor([[prev_stop_idx]], dtype=torch.int64, device=self.device)), dim=0)
            elif stop_idx_l[-1] != prev_stop_idx:
                stop_idx_l = torch.cat((stop_idx_l, torch.tensor([[prev_stop_idx]], dtype=torch.int64, device=self.device)), dim=0)
            if stop_idx_l[-1] >= self.gpu_throughput_size + last_gpu_batch_idx:
                last_gpu_batch_idx = stop_idx_l[-2].unsqueeze(0)
                gpu_batch_idx_l = torch.cat((gpu_batch_idx_l, last_gpu_batch_idx), dim=0)
        # Add last index to gpu_batch_idx_l
        if stop_idx_l.shape[0] != 0 and gpu_batch_idx_l.shape[0] != 0:
            if gpu_batch_idx_l[-1] != stop_idx_l[-1]:
                gpu_batch_idx_l = torch.cat((gpu_batch_idx_l, stop_idx_l[-1].unsqueeze(0)), dim=0)
        elif gpu_batch_idx_l.shape[0] == 0:
            gpu_batch_idx_l = torch.cat((gpu_batch_idx_l, stop_idx_l[-1].unsqueeze(0)), dim=0)
        return point_l, color_l, patches_l, stop_idx_l, gpu_batch_idx_l
    
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
                                                              crop_regions[idx][2], crop_regions[idx][3], antialias=None)
            
            img = self.preprocessFunction(img_crop).unsqueeze(0)
            patches = torch.cat((patches, img), dim=0)
        return patches
    
    def getImageFeatureForPatches(self, _patches, _stop_idx_l):
        """
        Get image features for a block.
        @param _data: tuple containing patches and stop_idx
        """
        with torch.no_grad():
            image_features = self.model.encode_image(_patches).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Compute element-wise average for image_feature vectors
        prev_idx = 0
        features = torch.empty((0,512),device=self.device)
        
        for idx in _stop_idx_l:
            if idx == prev_idx or idx == 0:
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

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()