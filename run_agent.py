import gradio as gr
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import PchipInterpolator
import supervision as sv
from supervision.geometry.core import Position
import torchvision
import copy
import math
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from einops import repeat
from packaging import version
from accelerate.utils import set_seed
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPVisionModelWithProjection
from diffusers import AutoencoderKLTemporalDecoder
from models.unet_spatio_temporal_condition_controlnet import UNetSpatioTemporalConditionControlNetModel
from pipeline.pipeline import FlowControlNetPipeline
from models.svdxt_featureflow_forward_controlnet_s2d_fixcmp_norefine import FlowControlNet, CMP_demo
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from segment_anything import sam_model_registry, SamPredictor

from agent.description import agent_prompt, agent_descriptions, agent_object
from agent.camera import agent_camera_motion
from agent.trajectory import agent_single_trajectory_grounding
from agent.llm import OpenAIModel
from utils.flow_viz import flow_to_image
from utils.utils import ensure_dirname, arr2pil
import time

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors

action_last = {}

def visualize_camera_poses(extrinsics, save_path, hw_ratio=9/16, base_xval=0.25, zval=0.5):
    fig = plt.figure(figsize=(18, 7))
    ax = fig.add_subplot(projection='3d')
    ax.set_aspect("auto")

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([1, -1])

    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')

    def swap_yz_axis(extrinsic):
        swap_matrix = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0]
        ])
        rotation = extrinsic[:3, :3]
        translation = extrinsic[:3, 3]
        new_rotation = swap_matrix @ rotation
        new_translation = swap_matrix @ translation
        new_extrinsic = np.eye(4)
        new_extrinsic[:3, :3] = new_rotation
        new_extrinsic[:3, 3] = new_translation
        return new_extrinsic

    def extrinsic2pyramid(extrinsic, color_map='red'):
        vertex_std = np.array([[0, 0, 0, 1],
                               [base_xval, -base_xval * hw_ratio, zval, 1],
                               [base_xval, base_xval * hw_ratio, zval, 1],
                               [-base_xval, base_xval * hw_ratio, zval, 1],
                               [-base_xval, -base_xval * hw_ratio, zval, 1]])
        vertex_transformed = vertex_std @ extrinsic.T
        meshes = [[vertex_transformed[0, :-1], vertex_transformed[1][:-1], vertex_transformed[2, :-1]],
                  [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
                  [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
                  [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
                  [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]]]
        color = color_map if isinstance(color_map, str) else plt.cm.rainbow(color_map)
        ax.add_collection3d(Poly3DCollection(meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=0.35))

    num_frames = extrinsics.shape[0]
    for frame_idx in range(num_frames):
        extrinsic = swap_yz_axis(extrinsics[frame_idx])
        color = frame_idx / num_frames
        extrinsic2pyramid(extrinsic, color_map=color)

    cmap = plt.cm.rainbow
    norm = mcolors.Normalize(vmin=0, vmax=num_frames)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical', label='Frame Number')

    plt.title('Extrinsic Parameters')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {save_path}.")

output_dir_video = "./gradio/videos"
output_dir_frame = "./gradio/frames"

ensure_dirname(output_dir_video)
ensure_dirname(output_dir_frame)

def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image, _ = transform(image_pil, None)
    return image_pil, image

def get_grounding_output(model, image, caption, box_threshold, text_threshold, device="cpu"):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."
    model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]
    boxes = outputs["pred_boxes"][0]

    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]
    boxes_filt = boxes[filt_mask]

    confidences = []
    for logit in logits_filt:
        confidences.append(logit.max().item())

    confidences = np.array(confidences).astype(np.float32)
    sorted_index = np.argsort(confidences)[::-1].tolist()
    confidences = confidences[sorted_index]
    boxes_filt = boxes_filt[sorted_index]
    class_id = np.arange(len(sorted_index))

    return boxes_filt, class_id, confidences

def rotation_matrix(rx, ry, rz):
    rx, ry, rz = np.deg2rad(rx), np.deg2rad(ry), np.deg2rad(rz)
    
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]])
    
    R_y = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])
    
    R_z = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])
    
    R = R_z @ R_y @ R_x
    return R

def generate_camera_extrinsics(camera_params):
    extrinsics = []
    
    for params in camera_params:
        x_t, y_t, z_t, x_r, y_r, z_r = params
        
        R = rotation_matrix(x_r, y_r, z_r)

        T = np.array([x_t, y_t, z_t])
        
        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, :3] = R
        extrinsic_matrix[:3, 3] = T
        
        extrinsics.append(extrinsic_matrix)
    
    return np.array(extrinsics)

def init_models(pretrained_model_name_or_path, resume_from_checkpoint, weight_dtype, device='cuda', enable_xformers_memory_efficient_attention=False, allow_tf32=False):
    print('start loading models...')
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(pretrained_model_name_or_path, subfolder="image_encoder", revision=None, variant="fp16")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(pretrained_model_name_or_path, subfolder="vae", revision=None, variant="fp16")
    unet = UNetSpatioTemporalConditionControlNetModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet", low_cpu_mem_usage=True, variant="fp16",)
    controlnet = FlowControlNet.from_pretrained(resume_from_checkpoint)
    cmp = CMP_demo('./models/cmp/experiments/semiauto_annot/resnet50_vip+mpii_liteflow/config.yaml', 42000)
    metric3d = torch.hub.load("./models/Metric3D", "metric3d_vit_small", source='local', map_location='cpu')
    metric3d.load_state_dict(torch.load('ckpts/metric_depth_vit_small_800k.pth')['model_state_dict'], strict=False)
    ground_dino_args = SLConfig.fromfile("./models/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    ground_dino_args.device = device
    ground_dino_args.bert_base_uncased_path = "/data/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594"
    ground_dino = build_model(ground_dino_args)
    ground_dino_checkpoint = torch.load("ckpts/groundingdino_swint_ogc.pth", map_location=device)
    ground_dino.load_state_dict(clean_state_dict(ground_dino_checkpoint["model"]), strict=False)
    ground_dino.eval()
    sam = sam_model_registry["vit_h"](checkpoint="ckpts/sam_vit_h_4b8939.pth")
    
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(False)
    cmp.requires_grad_(False)
    metric3d.requires_grad_(False)
    ground_dino.requires_grad_(False)
    sam.requires_grad_(False)

    image_encoder.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    controlnet.to(device, dtype=weight_dtype)
    cmp.to(device)
    metric3d.to(device)
    ground_dino.to(device)
    sam.to(device)
    
    sampredictor = SamPredictor(sam)

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                print(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly")

    if allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    
    pipeline = FlowControlNetPipeline.from_pretrained(
        pretrained_model_name_or_path,
        unet=unet,
        controlnet=controlnet,
        image_encoder=image_encoder,
        vae=vae,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(device)
    
    mllm = OpenAIModel(base_url="",
                    api_key="",
                    model="",
                    temperature=0.6,
                    max_tokens=1600)

    print('models loaded.')

    return pipeline, cmp, metric3d, mllm, ground_dino, sampredictor


class Drag:
    def __init__(self, device, height, width, model_length):
        self.device = device

        svd_ckpt = "ckpts/stable-video-diffusion-img2vid-xt-1-1"
        mofa_ckpt = "ckpts/controlnet"

        self.device = 'cuda:0'
        self.weight_dtype = torch.float16

        self.pipeline, self.cmp, self.metric3d, self.llm, self.ground_dino, self.sam_predictor = init_models(svd_ckpt, mofa_ckpt, weight_dtype=self.weight_dtype, device=self.device)
        
        self.first_frames_transform = transforms.Compose([transforms.ToTensor()])

        self.height = height
        self.width = width
        self.model_length = model_length
        self.FOV = 60
    
    def interpolate_trajectory(self, points):
        x = [point[0] for point in points]
        y = [point[1] for point in points]

        t = np.linspace(0, 1, len(points))

        fx = PchipInterpolator(t, x)
        fy = PchipInterpolator(t, y)

        new_t = np.linspace(0, 1, self.model_length)

        new_x = fx(new_t)
        new_y = fy(new_t)
        new_points = list(zip(new_x, new_y))

        return new_points
    
    def interpolate_camera(self, points):
        x_t = [point[0] for point in points]
        y_t = [point[1] for point in points]
        z_t = [point[2] for point in points]
        x_r = [point[3] for point in points]
        y_r = [point[4] for point in points]
        z_r = [point[5] for point in points]

        t = np.linspace(0, 1, len(points))

        fxt = PchipInterpolator(t, x_t)
        fyt = PchipInterpolator(t, y_t)
        fzt = PchipInterpolator(t, z_t)
        fxr = PchipInterpolator(t, x_r)
        fyr = PchipInterpolator(t, y_r)
        fzr = PchipInterpolator(t, z_r)

        new_t = np.linspace(0, 1, self.model_length)

        new_x_t = fxt(new_t)
        new_y_t = fyt(new_t)
        new_z_t = fzt(new_t)
        new_x_r = fxr(new_t)
        new_y_r = fyr(new_t)
        new_z_r = fzr(new_t)
        new_points = list(zip(new_x_t, new_y_t, new_z_t, new_x_r, new_y_r, new_z_r))

        return new_points  
    
    def get_sparseflow_and_mask_forward(self, resized_all_points, n_steps, H, W, is_backward_flow=False):

        K = resized_all_points.shape[0]

        starts = resized_all_points[:, 0]

        interpolated_ends = resized_all_points[:, 1:]

        s_flow = np.zeros((K, n_steps, H, W, 2))
        mask = np.zeros((K, n_steps, H, W))

        for k in range(K):
            for i in range(n_steps):
                start, end = starts[k], interpolated_ends[k][i]
                flow = np.int64(end - start) * (-1 if is_backward_flow is True else 1)
                s_flow[k][i][int(start[1]), int(start[0])] = flow
                mask[k][i][int(start[1]), int(start[0])] = 1

        s_flow = np.sum(s_flow, axis=0)
        mask = np.sum(mask, axis=0)

        return s_flow, mask
    
    def get_cmp_flow(self, frames, sparse_optical_flow, mask):
        b, t, c, h, w = frames.shape
        frames = frames.flatten(0, 1)
        sparse_optical_flow = sparse_optical_flow.flatten(0, 1)
        mask = mask.flatten(0, 1)

        cmp_flow_384, _ = self.cmp_run(frames, sparse_optical_flow, mask)
        
        if self.height != 384 or self.width != 384:
            scales = [self.height / 384, self.width / 384]
            controlnet_flow = F.interpolate(cmp_flow_384, (self.height, self.width), mode='nearest').reshape(b, t, 2, h, w)
            controlnet_flow[:, :, 0] *= scales[1]
            controlnet_flow[:, :, 1] *= scales[0]
        
        return controlnet_flow

    def cmp_run(self, image, sparse, mask):
        inference_size = [384, 384]
        
        image_384 = F.interpolate(image, size=inference_size, mode="bilinear", align_corners=True)
        image_384 = image_384 * 2 - 1
        cmp_output = self.cmp.model.model(image_384, torch.cat([sparse, mask], dim=1))
        flow = self.cmp.fuser.convert_flow(cmp_output)
        if flow.shape[2] != image_384.shape[2]:
            flow = F.interpolate(flow, size=image_384.shape[2:4], mode="bilinear", align_corners=True)

        return flow, cmp_output

    def get_integral_flow(self, pred_depth, object_flow, intrinsic, extrinsic):
        B, _, H, W = pred_depth.shape
        T = object_flow.shape[1]
        
        pred_depth = pred_depth.unsqueeze(1).repeat(1, T, 1, 1, 1)
        
        intrinsic_1, intrinsic_rest = intrinsic[:, :1], intrinsic[:, 1:]
        extrinsic_1, extrinsic_rest = extrinsic[:, :1], extrinsic[:, 1:]
        j, i = torch.meshgrid(
                torch.linspace(0, H - 1, H),
                torch.linspace(0, W - 1, W),
                indexing='ij'
            )

        u = i.reshape([1, 1, H * W, 1]).expand([B, T, H * W, 1]).to(pred_depth)
        v = j.reshape([1, 1, H * W, 1]).expand([B, T, H * W, 1]).to(pred_depth)
        uv = torch.cat((u, v), dim=-1)
        uv_offset = uv + object_flow.reshape(B, T, 2, -1).permute(0, 1, 3, 2)
        
        uv_norm = uv_offset.clone()
        uv_norm[..., 0] = uv_norm[..., 0] / (W - 1)
        uv_norm[..., 1] = uv_norm[..., 1] / (H - 1)
        uv_norm = uv_norm * 2 - 1
        sample_grid = uv_norm.view(-1, H*W, 1, 2)
        depth_reshaped = pred_depth.view(-1, 1, H, W)
        depth_sampled = F.grid_sample(depth_reshaped, sample_grid, mode='bilinear', padding_mode="border", align_corners=True)
        depth_sampled = depth_sampled.view(B, T, H*W, 1)
        
        uv1 = torch.cat((uv_offset, torch.ones_like(u)), dim=-1)
        xy1 = torch.matmul(intrinsic_1.inverse().unsqueeze(2), uv1.unsqueeze(-1)).squeeze(-1)
        xyz_c = xy1 * depth_sampled

        xyz1_c = torch.cat([xyz_c, torch.ones_like(xyz_c[...,:1])], dim=-1)
        xyz1_w = torch.matmul(extrinsic_1.inverse().unsqueeze(2), xyz1_c.unsqueeze(-1))
        xyz_c_rest= torch.matmul(extrinsic_rest.unsqueeze(2), xyz1_w).squeeze(-1)[...,:-1]
        
        xy1_rest = xyz_c_rest / xyz_c_rest[...,-1:]
        uv_rest = torch.matmul(intrinsic_rest.unsqueeze(2), xy1_rest.unsqueeze(-1)).squeeze(-1)[...,:-1]
        
        flows = (uv_rest - uv).reshape(B, -1, H, W, 2).permute(0, 1, 4, 2, 3)
        
        return flows

    @torch.no_grad()
    def cal_monocular_depth(self, video_frame, focal):
        input_size = (616, 1064)

        B, _, H, W = video_frame.shape

        rgb = video_frame * 255 
        mean = torch.tensor([123.675, 116.28, 103.53]).to(video_frame)[None, :, None, None]
        std = torch.tensor([58.395, 57.12, 57.375]).to(video_frame)[None, :, None, None]
        rgb = torch.div((rgb - mean), std)
        
        scale = min(input_size[0] / rgb.shape[2], input_size[1] / rgb.shape[3])
        rgb = F.interpolate(rgb, scale_factor=scale, mode='bilinear')
        resize_focal = focal * scale
        
        h, w = rgb.shape[2:]
        pad_h = input_size[0] - h
        pad_w = input_size[1] - w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
        rgb = F.pad(rgb, (pad_w_half, pad_w_half, pad_h_half, pad_h_half), mode='constant', value=0)
        
        pred_depth, confidence, output_dict = self.metric3d.inference({'input': rgb})
        
        pred_depth = pred_depth.permute(2, 3, 0, 1)
        pred_depth = pred_depth[pad_info[0]:pred_depth.shape[0]-pad_info[1], pad_info[2] : pred_depth.shape[1] - pad_info[3]].permute(2, 3, 0, 1)
        pred_depth = F.interpolate(pred_depth, size=(H, W), mode='bilinear')
        
        canonical_to_real_scale = resize_focal[:, None, None, None] / 1000.0
        pred_depth = pred_depth * canonical_to_real_scale
        pred_depth = torch.clamp(pred_depth, 0, 300)

        return pred_depth

    @torch.no_grad()
    def get_object_segment(self, image_path, output_path, text_prompt, box_threshold=0.3, text_threshold=0.25, num_samples_max=64):
        image_pil, image = load_image(image_path)

        boxes_filt, class_ids, confidences = get_grounding_output(self.ground_dino, image, text_prompt, box_threshold, text_threshold, device=self.device)
        if len(boxes_filt) == 0:
            return None, None, None
           
        image_cv = cv2.imread(image_path)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        self.sam_predictor.set_image(image_cv)

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.tensor([W, H, W, H], device=self.device)
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        detections = sv.Detections(xyxy=boxes_filt.cpu().numpy(), class_id=class_ids, confidence=confidences)

        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_filt, image_cv.shape[:2]).to(self.device)

        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.device),
            multimask_output=False,
        )
        
        masks = masks.squeeze(1).cpu().numpy()
        detections.mask = masks
        
        center_point_list = []
        sample_point_list = []
        for mask_idx in range(len(masks)):
            mask_single = masks[mask_idx]
            all_point_mask = np.argwhere(mask_single)
            all_point_mask = all_point_mask[..., ::-1]

            num_valid_points = all_point_mask.shape[0]
            area = H * W
            bbox_area = detections.area[mask_idx]
            num_samples = int(num_samples_max * bbox_area / area)

            if num_valid_points >= num_samples:
                sampled_indices = np.random.choice(num_valid_points, num_samples, replace=False)
            else:
                sampled_indices = np.random.choice(num_valid_points, num_samples, replace=True)
            
            sample_point = all_point_mask[sampled_indices]
            central_point = np.mean(all_point_mask, axis=0, keepdims=True)
            sample_point = np.concatenate([central_point, sample_point], axis=0)
            sample_point[:, 0]
            sample_point[:, 1]
            
            center_point_list.append(central_point[0].tolist())
            sample_point_list.append(sample_point)
        
        lables = [str(idx) for idx in class_ids.tolist()]
        
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        label_annotator = sv.LabelAnnotator(text_scale=2, text_thickness=2, text_position=Position.TOP_LEFT)
        annotated_image = box_annotator.annotate(scene=image_cv, detections=detections)
        annotated_image = mask_annotator.annotate(scene=annotated_image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=lables)
        segment_path = os.path.join(output_path, "segment.png")
        cv2.imwrite(segment_path, annotated_image[..., ::-1])
        
        return segment_path, center_point_list, sample_point_list
        
    def agent_analyse(self, input_name, text):
        global action_last
        
        image_path = os.path.join(output_dir_frame, input_name)
        
        object_movement_text, camera_motion_text = agent_prompt(image_path, text, self.llm)
        object_movement_text = agent_descriptions(image_path, object_movement_text, self.llm)
        
        trajectory_list =[]
        for description in object_movement_text:
            output_path = os.path.join(output_dir_frame, input_name.split('.')[0], description)
            os.makedirs(output_path, exist_ok=True)
            object_name = agent_object(image_path, description, self.llm)[0]
            segment_path, center_point, sample_point = self.get_object_segment(image_path, output_path, object_name)
            if segment_path is None: continue
            trajectory, start_point_act, trajectory_act = agent_single_trajectory_grounding(image_path, segment_path, output_path, description, center_point, sample_point, self.llm)
            trajectory_list.extend(trajectory)
            act_last_temp = {}
            act_last_temp['segment_path'] = segment_path
            act_last_temp['center_point'] = center_point
            act_last_temp['sample_point'] = sample_point
            act_last_temp['start_point_act'] = start_point_act
            act_last_temp['trajectory_act'] = trajectory_act
            action_last[description] = act_last_temp
                   
        all_points_list = []
        all_points = [tuple([tuple([int(e1[0]*384), int(e1[1]*384)]) for e1 in e]) for e in trajectory_list]
        for tnum in range(len(all_points)):
            all_points_list.append(self.interpolate_trajectory(all_points[tnum]))
        all_points = np.array(all_points_list)
        all_points = np.int64(all_points)
        all_points = np.clip(all_points, a_min=0, a_max=384-1)
        
        camera_motion, camera_act = agent_camera_motion(image_path, camera_motion_text, self.llm)
        camera_motion_list = self.interpolate_camera(camera_motion)
        act_last_temp = {}
        act_last_temp['camera_motion_text'] = camera_motion_text
        act_last_temp['camera_act'] = camera_act
        action_last['camera_info'] = act_last_temp
        
        cw2 = generate_camera_extrinsics(camera_motion_list)
        w2c = np.linalg.inv(cw2)
        
        visualize_camera_poses(cw2[::4], os.path.join(output_dir_frame, input_name.split('.')[0], 'camera_vis.png'))
        
        return all_points, w2c

    def agent_analyse_rethink(self, input_name, video_path):
        global action_last
        action_temp = {}
        
        image_path = os.path.join(output_dir_frame, input_name)
        
        trajectory_list =[]
        for description, action_last_dict in action_last.items():
            if 'camera_info' in description:
                continue
            output_path = os.path.join(output_dir_frame, input_name.split('.')[0], description)
            os.makedirs(output_path, exist_ok=True)
            segment_path, center_point, sample_point = action_last_dict['segment_path'], action_last_dict['center_point'], action_last_dict['sample_point']
            start_point_act_last, trajectory_act_last = action_last_dict['start_point_act'], action_last_dict['trajectory_act']
            trajectory, start_point_act, trajectory_act = agent_single_trajectory_grounding(image_path, segment_path, output_path, description, center_point, sample_point, self.llm, \
                rethink=True, video_path=video_path, start_point_act_last=start_point_act_last, trajectory_act_last=trajectory_act_last)
            trajectory_list.extend(trajectory)
            act_last_temp = {}
            act_last_temp['segment_path'] = segment_path
            act_last_temp['center_point'] = center_point
            act_last_temp['sample_point'] = sample_point
            act_last_temp['start_point_act'] = start_point_act
            act_last_temp['trajectory_act'] = trajectory_act
            action_temp[description] = act_last_temp
            
        all_points_list = []
        all_points = [tuple([tuple([int(e1[0]*384), int(e1[1]*384)]) for e1 in e]) for e in trajectory_list]
        for tnum in range(len(all_points)):
            all_points_list.append(self.interpolate_trajectory(all_points[tnum]))
        all_points = np.array(all_points_list)
        all_points = np.int64(all_points)
        all_points = np.clip(all_points, a_min=0, a_max=384-1)
        
        camera_info = action_last['camera_info']
        camera_motion_text = camera_info['camera_motion_text']
        camera_act_last = camera_info['camera_act']
        camera_motion, camera_act = agent_camera_motion(image_path, camera_motion_text, self.llm, \
            rethink=True, video_path=video_path, camera_act_last=camera_act_last)
        camera_motion_list = self.interpolate_camera(camera_motion)
        act_last_temp = {}
        act_last_temp['camera_motion_text'] = camera_motion_text
        act_last_temp['camera_motion'] = camera_motion
        act_last_temp['camera_act'] = camera_act
        action_temp['camera_info'] = act_last_temp
        
        cw2 = generate_camera_extrinsics(camera_motion_list)
        w2c = np.linalg.inv(cw2)
        
        action_last = action_temp
        
        visualize_camera_poses(cw2[::4], os.path.join(output_dir_frame, input_name.split('.')[0], 'camera_vis_rethink.png'))
        
        return all_points, w2c

    @torch.no_grad()
    def forward_sample(self, image_pil, input_first_frame, sparse_optical_flow, sparse_optical_flow_mask, K, w2c, ctrl_scale=1.):
        seed = 42
        set_seed(seed)
        
        controlnet_flow = self.get_cmp_flow(input_first_frame, sparse_optical_flow, sparse_optical_flow_mask)
        pred_depth = self.cal_monocular_depth(input_first_frame[:, 0, :, :, :], K[:, 0, 0, 0])
        
        trans_scale = pred_depth.min() * 0.15
        w2c[:, :, :3, 3] = w2c[:, :, :3, 3] * trans_scale
        w2c[:, :, 3, 3] = w2c[:, :, 3, 3] * 0.2
        
        controlnet_flow_camera = self.get_integral_flow(pred_depth, controlnet_flow, K, w2c)

        outputs = self.pipeline(
            image_pil, 
            image_pil,
            controlnet_flow_camera, 
            height=self.height,
            width=self.width,
            num_frames=self.model_length,
            decode_chunk_size=8,
            motion_bucket_id=127,
            fps=7,
            noise_aug_strength=0.02,
            controlnet_cond_scale=ctrl_scale,
        )
        
        return outputs.frames, controlnet_flow_camera 

    def run(self, input_path, input_path_agent, first_frame, text, ctrl_scale=0.6):
        input_name_agent = os.path.basename(input_path_agent)
        id = os.path.basename(input_path).split('.')[0]
        
        image_pil = copy.deepcopy(first_frame).convert('RGB')
        first_frames = repeat(self.first_frames_transform(image_pil), 'c h w -> b t c h w', b=1, t=self.model_length-1)
        
        all_points, w2c = self.agent_analyse(input_name_agent, text)
        
        w2c = torch.tensor(w2c)[None].float()
          
        if all_points.shape[0] != 0:
            sparse_optical_flow, sparse_optical_flow_mask = self.get_sparseflow_and_mask_forward(all_points, self.model_length - 1, 384, 384)
        else:
            sparse_optical_flow, sparse_optical_flow_mask = np.zeros((self.model_length - 1, 384, 384, 2)), np.zeros((self.model_length - 1, 384, 384))

        sparse_optical_flow = repeat(torch.tensor(sparse_optical_flow).float(), 't h w c -> b t c h w', b=1)
        sparse_optical_flow_mask = repeat(torch.tensor(sparse_optical_flow_mask).float(), 't h w -> b t c h w', b=1, c=sparse_optical_flow.shape[2])
        
        W, H = image_pil.size
        pixel = min(H, W)
        K = np.eye(3)
        K[0, 0] = (pixel/2.0)/math.tan(math.radians(self.FOV/2.0))
        K[1, 1] = (pixel/2.0)/math.tan(math.radians(self.FOV/2.0))
        K[0, 2] = W // 2
        K[1, 2] = H // 2
        K = torch.tensor(K)[None, None].repeat(1, self.model_length, 1, 1).float()
        
        frames, controlnet_flows = self.forward_sample(image_pil, first_frames.to(self.device), sparse_optical_flow.to(self.device), sparse_optical_flow_mask.to(self.device), K.to(self.device), w2c.to(self.device), ctrl_scale)
        
        frame = frames[0]
        controlnet_flow = controlnet_flows[0]
        
        for i in range(self.model_length):
            img = frame[i]
            frame[i] = np.array(img)

        viz_flows_sparse = []
        for i in range(self.model_length - 1):
            temp_flow = controlnet_flow[i].permute(1, 2, 0)
            viz_flows_sparse.append(flow_to_image(temp_flow))
        viz_flows_sparse = [np.uint8(np.ones_like(viz_flows_sparse[-1]) * 255)] + viz_flows_sparse
        viz_flows_sparse = np.stack(viz_flows_sparse)
        
        video_path = os.path.join(output_dir_video, f"{id}.mp4")
        condition_path = os.path.join(output_dir_video, f"{id}-flow.mp4")
        torchvision.io.write_video(video_path, frame, fps=8, video_codec='h264', options={'crf': '10'})
        torchvision.io.write_video(condition_path, viz_flows_sparse, fps=8, video_codec='h264', options={'crf': '10'})
                   
        return video_path, condition_path
    
    def rethink(self, input_path, input_path_agent, first_frame, text, ctrl_scale=0.6):
        input_name_agent = os.path.basename(input_path_agent)
        id = os.path.basename(input_path).split('.')[0]
        
        video_path = os.path.join(output_dir_video, f"{id}.mp4")
        condition_path = os.path.join(output_dir_video, f"{id}-flow.mp4")
        
        image_pil = copy.deepcopy(first_frame).convert('RGB')
        first_frames = repeat(self.first_frames_transform(image_pil), 'c h w -> b t c h w', b=1, t=self.model_length-1)
        
        all_points, w2c = self.agent_analyse_rethink(input_name_agent, video_path)
        
        w2c = torch.tensor(w2c)[None].float()
          
        if all_points.shape[0] != 0:
            sparse_optical_flow, sparse_optical_flow_mask = self.get_sparseflow_and_mask_forward(all_points, self.model_length - 1, 384, 384)
        else:
            sparse_optical_flow, sparse_optical_flow_mask = np.zeros((self.model_length - 1, 384, 384, 2)), np.zeros((self.model_length - 1, 384, 384))

        sparse_optical_flow = repeat(torch.tensor(sparse_optical_flow).float(), 't h w c -> b t c h w', b=1)
        sparse_optical_flow_mask = repeat(torch.tensor(sparse_optical_flow_mask).float(), 't h w -> b t c h w', b=1, c=sparse_optical_flow.shape[2])
        
        W, H = image_pil.size
        pixel = min(H, W)
        K = np.eye(3)
        K[0, 0] = (pixel/2.0)/math.tan(math.radians(self.FOV/2.0))
        K[1, 1] = (pixel/2.0)/math.tan(math.radians(self.FOV/2.0))
        K[0, 2] = W // 2
        K[1, 2] = H // 2
        K = torch.tensor(K)[None, None].repeat(1, self.model_length, 1, 1).float()
        
        frames, controlnet_flows = self.forward_sample(image_pil, first_frames.to(self.device), sparse_optical_flow.to(self.device), sparse_optical_flow_mask.to(self.device), K.to(self.device), w2c.to(self.device), ctrl_scale)
        
        frame = frames[0]
        controlnet_flow = controlnet_flows[0]
        
        for i in range(self.model_length):
            img = frame[i]
            frame[i] = np.array(img)

        viz_flows_sparse = []
        for i in range(self.model_length - 1):
            temp_flow = controlnet_flow[i].permute(1, 2, 0)
            viz_flows_sparse.append(flow_to_image(temp_flow))
        viz_flows_sparse = [np.uint8(np.ones_like(viz_flows_sparse[-1]) * 255)] + viz_flows_sparse
        viz_flows_sparse = np.stack(viz_flows_sparse)
        
        torchvision.io.write_video(video_path, frame, fps=8, video_codec='h264', options={'crf': '10'})
        torchvision.io.write_video(condition_path, viz_flows_sparse, fps=8, video_codec='h264', options={'crf': '10'})
                   
        return video_path, condition_path

with gr.Blocks() as demo:
    
    target_size = 512
    agent_size = 1600
    DragNUWA_net = Drag("cuda:0", target_size, target_size, 25)
    input_name = gr.State()
    input_name_agent = gr.State()
    input_image_pillow = gr.State()
    
    def preprocess_image(image_arry):
        image_pil = arr2pil(image_arry)
        image_pil_agent = arr2pil(image_arry)
        raw_w, raw_h = image_pil.size

        max_edge = min(raw_w, raw_h)
        max_edge_agent = min(raw_w, raw_h)
        resize_ratio = target_size / max_edge
        resize_ratio_agent = agent_size / max_edge_agent

        image_pil = image_pil.resize((round(raw_w * resize_ratio), round(raw_h * resize_ratio)), Image.BILINEAR)
        image_pil_agent = image_pil_agent.resize((round(raw_w * resize_ratio_agent), round(raw_h * resize_ratio_agent)), Image.BILINEAR)

        new_w, new_h = image_pil.size
        crop_w = new_w - (new_w % 64)
        crop_h = new_h - (new_h % 64)
        new_w_agent, new_h_agent = image_pil_agent.size
        crop_w_agent = new_w_agent - (new_w_agent % 64)
        crop_h_agent = new_h_agent - (new_h_agent % 64)

        image_pil = transforms.CenterCrop((crop_h, crop_w))(image_pil.convert('RGB'))
        image_pil_agent = transforms.CenterCrop((crop_h_agent, crop_w_agent))(image_pil_agent.convert('RGB'))

        DragNUWA_net.width = crop_w
        DragNUWA_net.height = crop_h

        id = str(time.time()).split('.')[0]
        id_agent = id + '_agent'

        first_frame_path = os.path.join(output_dir_frame, f"{id}.png")
        first_frame_path_agent = os.path.join(output_dir_frame, f"{id_agent}.png")
        image_pil.save(first_frame_path)
        image_pil_agent.save(first_frame_path_agent)
        return f"{id}.png", f"{id}_agent.png", image_pil, image_pil
    
    with gr.Row():
        with gr.Column(scale=2):
            text = gr.Textbox(label="Add Text Here", visible=True)
            run_button = gr.Button(value="Run")
            rethink_button = gr.Button(value="Rethink")
                        
        with gr.Column(scale=5):
            input_image = gr.Image(label="Add Image Here", interactive=True)
            
    with gr.Row():
        with gr.Column(scale=6):
            output_video_mp4 = gr.Video(label="Output Video mp4", interactive=False)
        with gr.Column(scale=6):
            output_flow_mp4 = gr.Video(label="Output Flow mp4", interactive=False)
    
    input_image.upload(preprocess_image, input_image, [input_name, input_name_agent, input_image, input_image_pillow])
    run_button.click(DragNUWA_net.run, [input_name, input_name_agent, input_image_pillow, text], [output_video_mp4, output_flow_mp4])
    rethink_button.click(DragNUWA_net.rethink, [input_name, input_name_agent, input_image_pillow, text], [output_video_mp4, output_flow_mp4])
    
    demo.launch(server_name="127.0.0.1", debug=True, server_port=1234)