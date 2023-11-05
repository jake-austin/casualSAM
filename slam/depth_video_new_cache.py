# %%
from time import time
from abc import ABC, abstractmethod
from PIL import Image

import torch
import numpy as np
import os
from os.path import join, basename
from glob import glob
from slam.losses import OccMask
from util.util_colormap import heatmap_to_pseudo_color
from third_party.raft import load_RAFT, get_input_padder
from util.util_3dvideo import save_ply, depth_to_points_Rt, Video_3D_Webpage
from networks.MiDaS import MidasNet, MidasNet_featopt
from configs import midas_pretrain_path, davis_path
from networks.goem_opt import _so3_exp_map, CameraPoseDeltaCollection, DepthScaleShiftCollection, DepthBasedWarping, CameraIntrinsics, get_relative_transform
from skimage.transform import resize as imresize
from tqdm import tqdm
from slam.util_misc import Timer, ResultCache, KeyFrameBuffer
from slam import util_sintel_io
from torch.nn.functional import interpolate
from slam.models.uncertainty import HRNetUncertaintyModel, MidasUnceratintyModel
from slam.models.cache import FeatListCache
import torch.nn.functional as F
import torchvision
import cv2

def get_video_dataset(opt):
    if opt.dataset_name == 'davis':
        return DavisVideoDataset(opt)
    elif opt.dataset_name == 'sintel':
        return SintelVideoDataset(opt)
    elif opt.dataset_name == 'tum':
        return TUMVideoDataset(opt)
    elif opt.dataset_name == 'tum_static':
        return StaticTUMVideoDataset(opt)
    elif opt.dataset_name == 'davis_local':
        return DavisLocalVideoDataset(opt)
    elif opt.dataset_name == 'local':
        return LocalVideoDataset(opt)
    else:
        raise NotImplementedError(
            f'dataset name only support davis and sintel. Got {opt.dataset_name} instead')



class Resize(object):
    """Resize sample to given size (width, height).
    """

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_AREA,
    ):
        """Init.

        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        self.__width = width
        self.__height = height

        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self.__resize_method} not implemented"
                )

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=self.__width
            )
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=self.__width
            )
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, sample):
        width, height = self.get_size(
            sample["image"].shape[1], sample["image"].shape[0]
        )

        # resize sample
        sample["image"] = cv2.resize(
            sample["image"],
            (width, height),
            interpolation=self.__image_interpolation_method,
        )

        if self.__resize_target:
            if "disparity" in sample:
                sample["disparity"] = cv2.resize(
                    sample["disparity"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

            if "depth" in sample:
                sample["depth"] = cv2.resize(
                    sample["depth"], (width, height), interpolation=cv2.INTER_NEAREST
                )

            sample["mask"] = cv2.resize(
                sample["mask"].astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            sample["mask"] = sample["mask"].astype(bool)

        return sample


class NormalizeImage(object):
    """Normlize image by given mean and std.
    """

    def __init__(self, mean, std):
        self.__mean = mean
        self.__std = std

    def __call__(self, sample):
        sample["image"] = (sample["image"] - self.__mean) / self.__std

        return sample

class PrepareForNet(object):
    """Prepare sample for usage as network input.
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        image = np.transpose(sample["image"], (2, 0, 1))
        sample["image"] = np.ascontiguousarray(image).astype(np.float32)

        if "mask" in sample:
            sample["mask"] = sample["mask"].astype(np.float32)
            sample["mask"] = np.ascontiguousarray(sample["mask"])

        if "disparity" in sample:
            disparity = sample["disparity"].astype(np.float32)
            sample["disparity"] = np.ascontiguousarray(disparity)

        if "depth" in sample:
            depth = sample["depth"].astype(np.float32)
            sample["depth"] = np.ascontiguousarray(depth)

        return sample

class ZoeDepth(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True)
        self.dummy_param = torch.nn.Parameter(torch.zeros([1]))
    def forward(self, x):
        return (1/self.net.infer(x)) + torch.clip(self.dummy_param, 0, 1e-5)
    def parameters(self):
        return [self.dummy_param]


class MidasV3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.hub.load("isl-org/MiDaS", "DPT_BEiT_L_512")
        self.net.eval()
        self.transforms = torchvision.transforms.Compose(
            [
                Resize(
                    512,
                    512,
                    resize_target=None,
                    keep_aspect_ratio=False,
                    ensure_multiple_of=32,
                    resize_method="minimal",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                PrepareForNet(),
            ]
        )

        self.dummy_param = torch.nn.Parameter(torch.zeros([1]))
    def forward(self, x):
        print(x.mean())
        imgs = []
        for i in range(len(x)):
            img = self.transforms({"image": x[i].permute(1,2,0).cpu().numpy()})['image']
            print(img.shape)
            img = torch.from_numpy(img).unsqueeze(0)
            imgs.append(img)

        transformed = torch.concatenate(imgs, dim=0).to(x.device)
        print(transformed.shape)
        prediction = self.net(transformed).unsqueeze(1)
        print(prediction.shape)
        prediction = torch.nn.functional.interpolate(
            prediction,
            size=x.shape[2:],
            mode="bicubic",
            align_corners=False,
        )
        print(prediction.shape)
        out = (prediction) + torch.clip(self.dummy_param, 0, 1e-5)
        print(out.mean())
        return out
    def parameters(self):
        return [self.dummy_param]


class RawUncertaintyOpt(torch.nn.Module):
    def __init__(self, number_of_frames, shape, output_channel=1, constant_uncertainty=False) -> None:
        """
        Saves a per-frame uncertainty map
        """
        super().__init__()
        self.constant_uncertainty = constant_uncertainty
        self.output_channel = output_channel
        self.maps = torch.nn.Parameter(.5 * torch.ones(number_of_frames, output_channel, *shape))
        self.elu = torch.nn.ELU(True)

        if output_channel > 1:
            self.scale = torch.nn.Parameter(torch.ones([1])*-2)
    def forward(self, indices):
        B, C, H, W = self.maps.shape
        if self.constant_uncertainty:
            return (torch.ones([B, self.output_channel, H, W]).to(indices.device)[indices])*0.5
        else:
            uncertainty = self.elu(self.maps[indices])+1
            if self.output_channel > 1:
                uncertainty_2 = uncertainty * \
                    F.softplus(self.scale, beta=2)
                uncertainty = torch.cat([uncertainty, uncertainty_2], dim=1)
            return uncertainty

class DepthVideoDataset(ABC):
    def __init__(self) -> None:
        pass
    # fuctions related to initialization

    def initialize(self, opt):
        self.opt = opt
        self.get_paths(opt)
        if hasattr(opt, 'eval_only'):
            if opt.eval_only:
                self.initialize_eval()
                return
        self.read_images()
        self.read_intrinsics()
        if opt.has_gt:
            self.read_gt_info()
        self.init_networks()
        self.init_cache()
        self.depth_based_warping = DepthBasedWarping()

    def get_output_shape(self, multiple=64, max_width=None):  # TODO: fix
        if max_width is None:
            W_max = self.opt.max_width  # use the read in width
        else:
            W_max = max_width
        H_original, W_original = self.original_shape
        aspc = W_original/H_original
        W_output = min(W_original, W_max)
        W_output = (W_output//multiple)*multiple
        H_output = ((W_output/aspc)//multiple)*multiple
        return int(H_output), int(W_output)

    def load_checkpoint(self, path):
        self.depth_net.load_state_dict(torch.load(join(path, 'depth_net.pth')))
        if os.path.exists(join(path, 'uncertainty_net.pth')):
            self.uncertainty_net.load_state_dict(
                torch.load(join(path, 'uncertainty_net.pth')))
        self.poses.load_state_dict(torch.load(join(path, 'poses.pth')))
        self.scale_and_shift.load_state_dict(
            torch.load(join(path, 'scale_and_shift.pth')))
        if hasattr(self, 'camera_intrinsics'):
            self.camera_intrinsics.load_state_dict(
                torch.load(join(path, 'camera_intrinsics.pth')))
        self.depth_net.eval()

    @abstractmethod
    def get_paths(self, opt):
        raise NotImplementedError

    def init_cache(self):
        self.flow_cache = ResultCache(
            max_cache_length=self.opt.cache_length, enable_cpu=False)
        self.flow_mask_cache = ResultCache(
            max_cache_length=self.opt.cache_length, enable_cpu=False)
        self.depth_cache = ResultCache(max_cache_length=self.opt.cache_length)
        self.mask_cache = ResultCache(max_cache_length=self.opt.cache_length)
        self.depthfeat_cache = FeatListCache(
            number_of_frames=self.number_of_frames, list_size=4)
        self.uncertainfeat_cache = FeatListCache(
            number_of_frames=self.number_of_frames, list_size=4)

    def clear_flow_cache(self):
        del self.flow_cache
        del self.flow_mask_cache

    @abstractmethod
    def read_images(self):
        raise NotImplementedError

    @abstractmethod
    def read_intrinsics(self):
        raise NotImplementedError

    @abstractmethod
    def read_gt_info(self):
        raise NotImplementedError

    def initialize_depth_scale(self):
        for i in range(self.number_of_frames):
            self.scale_and_shift.set_scale(i, self.opt.depth_scale_init)

    def init_networks(self):
        # pose, depth_scale, depth_net and flow_net
        self.poses = CameraPoseDeltaCollection(self.number_of_frames)
        self.scale_and_shift = DepthScaleShiftCollection(
            self.number_of_frames, use_inverse=self.opt.use_inverse_scale, grid_size=self.opt.scale_grid_size)
        self.scale_and_shift.set_outputshape(self.opt_shape)
        if self.opt.depth_net == 'midas':
            self.depth_net = MidasNet(path=midas_pretrain_path,
                                      normalize_input=True)

        elif self.opt.depth_net == 'midas_feat_opt':
            self.depth_net = MidasNet_featopt(path=midas_pretrain_path,
                                              normalize_input=True)
            self.depth_net.add_refine_branch()
        elif self.opt.depth_net == 'zoedepth':
            self.depth_net = ZoeDepth()
        elif self.opt.depth_net == 'midasv3':
            self.depth_net = MidasV3()
        else:
            raise NotImplementedError
        self.depth_net = self.depth_net.eval()
        if self.opt.use_uncertainty:
            # self.uncertainty_net = HRNetUncertaintyModel(
            #    output_shape=self.opt_shape)  # .eval()
            if self.opt.depth_net != "zoedepth" and self.opt.depth_net != "midasv3" and not self.opt.learn_raw_uncertainty:
                self.uncertainty_net = MidasUnceratintyModel(
                    output_channel=self.opt.uncertainty_channels, constant_uncertainty=self.opt.use_constant_uncertainty)
            else:
                self.uncertainty_net = RawUncertaintyOpt(output_channel=self.opt.uncertainty_channels, constant_uncertainty=self.opt.use_constant_uncertainty, number_of_frames=self.number_of_frames, shape=self.opt_shape)
        self.flow_net = load_RAFT()
        self.flow_net = self.flow_net.eval()
        self.get_valid_flow_mask = OccMask(th=1.0)
        self.camera_intrinsics = CameraIntrinsics(
            init_focal_length=self.opt.init_focal_length)
        self.camera_intrinsics.register_shape(
            self.original_shape, self.opt_shape)

    def cuda(self, cache_data=False):
        self.to('cuda', cache_data)

    def to(self, device, cache_data=False):
        self.poses = self.poses.to(device)
        self.scale_and_shift = self.scale_and_shift.to(device)
        self.depth_net = self.depth_net.to(device)
        self.flow_net = self.flow_net.to(device)
        self.depth_based_warping = self.depth_based_warping.to(device)
        self.camera_intrinsics = self.camera_intrinsics.to(device)
        self.uncertainty_net = self.uncertainty_net.to(device)
        self.device = device

        if hasattr(self, 'K'):
            self.K = self.K.to(device)
            self.inv_K = torch.linalg.inv(self.K)

        self.device = device
        if cache_data:
            self.images = self.images.to(device)
            self.masks = self.masks.to(device)
            self.images_orig = self.images_orig.to(device)

    def reload_depth_net(self):
        if self.opt.depth_net != 'zoedepth' and self.opt.depth_net != 'midasv3':
            self.depth_net = MidasNet(path=midas_pretrain_path,
                                    normalize_input=True)
        elif self.opt.depth_net == 'midasv3':
            self.depth_net = MidasV3()
        else:
            self.depth_net = ZoeDepth()
        self.depth_net.to(self.device)
        self.depth_net = self.depth_net.eval()
        self.reset_optimizer()

    def init_depth_optimizer(self):
        if self.opt.depth_net == 'midas':
            optimization_layers = self.opt.optimization_layers
            for p in self.depth_net.parameters():
                p.requires_grad = False
            for l in optimization_layers:
                for p in getattr(self.depth_net.scratch, l).parameters():
                    p.requires_grad = True
            optimization_param = []
            for l in optimization_layers:
                optimization_param += list(getattr(self.depth_net.scratch,
                                                   l).parameters())
            self.depth_optimizer = torch.optim.Adam(
                optimization_param, lr=self.opt.depth_lr)
        elif self.opt.depth_net == 'midas_feat_opt':
            for p in self.depth_net.parameters():
                p.requires_grad = False
            for p in self.depth_net.refine_branch.parameters():
                p.requires_grad = True
            self.depth_optimizer = torch.optim.Adam(
                self.depth_net.refine_branch.parameters(),
                lr=self.opt.depth_lr)
        elif self.opt.depth_net == 'zoedepth' or self.opt.depth_net == 'midasv3':
            self.depth_optimizer = torch.optim.Adam(
                self.depth_net.parameters(),
                lr=self.opt.depth_lr)
        else:
            raise NotImplementedError

    def init_pose_optimizer(self):
        rotation_params, translation_params = self.poses.get_rotation_and_translation_params()
        self.rotation_optimizer = torch.optim.Adam(
            rotation_params, lr=self.opt.rotation_lr, weight_decay=0, amsgrad=True)
        self.translation_optimizer = torch.optim.Adam(
            translation_params, lr=self.opt.translation_lr, weight_decay=0, amsgrad=True)

    def init_scale_optimizer(self):
        self.scale_and_shift_optimizer = torch.optim.Adam(
            self.scale_and_shift.parameters(), lr=self.opt.scale_lr, weight_decay=0, amsgrad=True)

    def init_uncertainty_optimizer(self):
        for p in self.uncertainty_net.parameters():
            p.requires_grad = False
        if self.opt.use_uncertainty:
            if self.opt.depth_net != 'zoedepth' and self.opt.depth_net != 'midasv3' and not self.opt.learn_raw_uncertainty:
                for p in self.uncertainty_net.uncertainty_decoder.parameters():
                    p.requires_grad = True
                self.uncertainty_optimizer = torch.optim.Adam(
                    self.uncertainty_net.uncertainty_decoder.parameters(), lr=self.opt.uncertainty_lr)
            elif self.opt.depth_net == 'zoedepth' or self.opt.depth_net == 'midasv3' or self.opt.learn_raw_uncertainty:
                for p in self.uncertainty_net.parameters():
                    p.requires_grad = True
                self.uncertainty_optimizer = torch.optim.Adam(
                    self.uncertainty_net.parameters(), lr=self.opt.uncertainty_lr)
            else:
                raise NotImplementedError

    def init_intrinsics_optimizer(self):
        self.intrinsics_optimizer = torch.optim.Adam(
            self.camera_intrinsics.parameters(), lr=self.opt.intrinsics_lr, weight_decay=0, amsgrad=True)

    def init_optimizer(self):
        self.active_optimizers = []
        self.init_depth_optimizer()
        self.init_pose_optimizer()
        self.init_scale_optimizer()
        self.init_intrinsics_optimizer()
        self.init_uncertainty_optimizer()

    def reset_optimizer(self):
        self.active_optimizers = []
        self.all_optmizers = []
        del self.depth_optimizer
        del self.rotation_optimizer
        del self.translation_optimizer
        del self.scale_and_shift_optimizer
        del self.intrinsics_optimizer
        self.init_depth_optimizer()
        self.init_pose_optimizer()
        self.init_scale_optimizer()
        self.init_intrinsics_optimizer()
        self.init_uncertainty_optimizer()

    def init_keyframe_buffer(self, buffer_size=5):
        self.keyframe_buffer = []
        self.keyframe_buffer_size = buffer_size

    def store_initial_depth(self, batch_size=10):
        with torch.no_grad():
            # import pdb
            # pdb.set_trace()
            self.initial_depth = []
            frame_range = list(range(self.number_of_frames))
            chuncks = [frame_range[i:i+batch_size]
                       for i in range(0, len(frame_range), batch_size)]
            for chunk in chuncks:
                imgs = self.images[chunk, ...]
                if self.opt.depth_net != 'zoedepth' and self.opt.depth_net != 'midasv3':
                    feats = self.depth_net.forward_backbone(imgs)
                    self.depthfeat_cache.add_feat(feats)
                    depths = self.depth_net.forward_refine(feats)
                elif self.opt.depth_net == 'zoedepth' or self.opt.depth_net == 'midasv3':
                    depths = self.depth_net(imgs)
                    #print(depths.shape, len(depths))
                    #self.depthfeat_cache.add_feat(torch.empty(len(depths), device=depths.device))
                else:
                    raise NotImplementedError
                self.initial_depth.append(depths)
            self.initial_depth = torch.cat(self.initial_depth, dim=0)
            if self.opt.depth_net != 'zoedepth' and self.opt.depth_net != 'midasv3':
                self.depthfeat_cache.convert_to_tensor()

    def store_initial_uncertainty_feat(self, batch_size=10):
        if self.opt.depth_net != 'zoedepth' and self.opt.depth_net != 'midasv3' and not self.opt.learn_raw_uncertainty:
            with torch.no_grad():
                frame_range = list(range(self.number_of_frames))
                chuncks = [frame_range[i:i+batch_size]
                        for i in range(0, len(frame_range), batch_size)]
                for chunk in chuncks:
                    imgs = self.images[chunk, ...]
                    feats = self.uncertainty_net.get_uncertainty_feat(imgs)
                    self.uncertainfeat_cache.add_feat(feats)
                self.uncertainfeat_cache.convert_to_tensor()

    def store_relative_pose(self, frame_list=None):
        with torch.no_grad():
            if frame_list is None:
                frame_list = list(range(self.number_of_frames))
            R, t = self.poses(frame_list)
            relative_R, relative_t = get_relative_transform(
                R[:-1, ...], t[:-1, ...], R[1:, ...], t[1:, ...])
            self.relative_R = relative_R.detach().clone()
            self.relative_t = relative_t.detach().clone()

    def get_current_relative_pose(self, frame_list=None):
        if frame_list is None:
            frame_list = list(range(self.number_of_frames))
        R, t = self.poses(frame_list)
        relative_R, relative_t = get_relative_transform(
            R[:-1, ...], t[:-1, ...], R[1:, ...], t[1:, ...])
        return relative_R, relative_t

    def predict_depth(self, frame_index_list, no_grad=False):
        depths = []
        frame_list_cuda = torch.LongTensor(frame_index_list).to(self.device)
        if self.opt.depth_net != 'zoedepth' and self.opt.depth_net != 'midasv3':
            depth_feat = self.depthfeat_cache[frame_list_cuda]
            depth_out = self.depth_net.forward_refine(depth_feat)
        elif self.opt.depth_net == 'zoedepth' or self.opt.depth_net == 'midasv3':
            depth_out = self.initial_depth[frame_list_cuda]
        else:
            raise NotImplementedError
        del frame_list_cuda
        return depth_out

    def predict_uncertainty(self, frame_index_list):
        img_index = torch.LongTensor(frame_index_list).to(device=self.device)
        if self.opt.depth_net != 'zoedepth' and self.opt.depth_net != 'midasv3' and not self.opt.learn_raw_uncertainty:
            feat = self.uncertainfeat_cache[img_index]
            return self.uncertainty_net(feat)
        elif self.opt.depth_net == 'zoedepth' or self.opt.depth_net == 'midasv3' or self.opt.learn_raw_uncertainty:
            return self.uncertainty_net(img_index)


    def predict_depth_with_uncertainty(self, frame_index_list, no_depth_grad=False):
        if self.opt.depth_net != 'zoedepth' and self.opt.depth_net != 'midasv3':
            if type(frame_index_list) is not torch.Tensor:
                index_list_cuda = torch.LongTensor(
                    frame_index_list).to(self.device)
                depth = self.depth_net.forward_refine(
                    self.depthfeat_cache[index_list_cuda])
                if not self.opt.learn_raw_uncertainty:
                    uncetrainty = self.uncertainty_net(
                        self.uncertainfeat_cache[index_list_cuda])
                else:
                    uncetrainty = self.uncertainty_net(index_list_cuda)
            else:
                depth = self.depth_net.forward_refine(
                    self.depthfeat_cache[frame_index_list])
                if not self.opt.learn_raw_uncertainty:
                    uncetrainty = self.uncertainty_net(
                        self.uncertainfeat_cache[frame_index_list])
                else:
                    uncetrainty = self.uncertainty_net(frame_index_list)
            return depth, uncetrainty
        elif self.opt.depth_net == 'zoedepth' or self.opt.depth_net == 'midasv3':
            if type(frame_index_list) is not torch.Tensor:
                index_list_cuda = torch.LongTensor(
                    frame_index_list).to(self.device)
                depth = self.initial_depth[index_list_cuda]
                uncetrainty = self.uncertainty_net(index_list_cuda)
            else:
                depth = self.initial_depth[frame_index_list]
                uncetrainty = self.uncertainty_net(frame_index_list)
            return depth, uncetrainty

    # def predict_depth_with_cache(self, frame_index_list):
    #     # this function does not pass gradients to depth prediction networks.
    #     out = []
    #     for i in frame_index_list:
    #         v = self.depth_cache.get_item(i)
    #         if v is None:
    #             with torch.no_grad():
    #                 v = self.depth_net(
    #                     self.images[i:i+1, ...].to(self.device), inverse_depth=True)
    #             self.depth_cache.add_item(i, v)
    #         out.append(v)
    #     return torch.cat(out, dim=0)

    def predict_flow_with_cache(self, frame_pair):
        if not hasattr(self, 'flow_resize_scale'):
            output_H, output_W = self.output_shape
            opt_H, opt_W = self.opt_shape
            self.flow_resize_scale = [opt_H/output_H, opt_W/output_W]
        scale_H, scale_W = self.flow_resize_scale

        i, j = frame_pair
        v = self.flow_cache.get_item((i, j))
        if v is None:
            with torch.no_grad():
                v = self.flow_net(image1=self.images_orig[i:i+1, ...].to(
                    self.device)*255, image2=self.images_orig[j:j+1, ...].to(self.device)*255, iters=20, test_mode=True)[1]
                v = interpolate(v, size=self.opt_shape,
                                mode='bilinear', align_corners=True)
                v[:, 0, ...] *= scale_W  # x
                v[:, 1, ...] *= scale_H  # y
            self.flow_cache.add_item((i, j), v)
        return v

    def get_flow_mask_with_cache(self, frame_pair):
        i, j = frame_pair
        v_i_j = self.flow_mask_cache.get_item((i, j))
        v_j_i = self.flow_mask_cache.get_item((j, i))
        if v_i_j is None or v_j_i is None:
            flow_i_j = self.predict_flow_with_cache((i, j))
            flow_j_i = self.predict_flow_with_cache((j, i))
            # print(flow_j_i.shape)
            valid_mask_i = self.get_valid_flow_mask(flow_i_j, flow_j_i)
            valid_mask_j = self.get_valid_flow_mask(flow_j_i, flow_i_j)
            self.flow_mask_cache.add_item((i, j), valid_mask_i)
            self.flow_mask_cache.add_item((j, i), valid_mask_j)
            v_i_j = valid_mask_i
            v_j_i = valid_mask_j
        return v_i_j, v_j_i

    def get_mask_with_cache(self, i):
        return self.masks[i:i+1, ...].to(self.device)

    def freeze_frame(self, frame_index, scale_only=False, pose_only=False):
        if scale_only:
            pos_param = getattr(self.poses, f'delta_rotation_{frame_index}')
            pos_param.requires_grad = False
            trans_param = getattr(
                self.poses, f'delta_translation_{frame_index}')
            trans_param.requires_grad = False
        if pose_only:
            scale_param = getattr(self.scale_and_shift, f'scale_{frame_index}')
        else:
            pos_param = getattr(self.poses, f'delta_rotation_{frame_index}')
            pos_param.requires_grad = False
            trans_param = getattr(
                self.poses, f'delta_translation_{frame_index}')
            trans_param.requires_grad = False
            scale_param = getattr(self.scale_and_shift, f'scale_{frame_index}')

    def unfreeze_frame(self, frame_index, opt_scale=False):
        pos_param = getattr(self.poses, f'delta_rotation_{frame_index}')
        pos_param.requires_grad = True
        trans_param = getattr(
            self.poses, f'delta_translation_{frame_index}')
        trans_param.requires_grad = True
        if opt_scale:
            scale_param = getattr(self.scale_and_shift, f'scale_{frame_index}')
            scale_param.requires_grad = True


# %%


class SintelVideoDataset(DepthVideoDataset):
    def __init__(self, opt):
        super().__init__()
        self.get_paths(opt)
        self.initialize(opt)

    def get_paths(self, opt):
        track_name = opt.track_name
        self.paths = {}
        sintel_cam_path, sintel_depth_path, sintel_flow_path, sintel_seg_path, sintel_img_path, number_of_frames = util_sintel_io.get_path(
            track_name=track_name)
        self.paths['cam'] = sintel_cam_path
        self.paths['depth'] = sintel_depth_path
        self.paths['seg'] = sintel_seg_path
        self.paths['img'] = sintel_img_path
        self.number_of_frames = number_of_frames

    def initialize_eval(self):
        i = 1
        img = util_sintel_io.img_read(
            join(self.paths['img'], f'frame_{i:04d}.png'))
        if not hasattr(self, 'original_shape'):
            self.original_shape = img.shape[:2]
            H_output, W_output = self.get_output_shape(
                multiple=64, max_width=self.opt.max_width)
            self.output_shape = (H_output, W_output)
        self.read_intrinsics()
        self.read_gt_info()

    def read_images(self):
        self.images = []
        self.images_orig = []
        self.masks = []
        if self.opt.frame_cap is not None:
            self.number_of_frames = min(
                self.opt.frame_cap, self.number_of_frames)
        for i in range(1, self.number_of_frames+1, self.opt.image_sequence_stride):
            img = util_sintel_io.img_read(
                join(self.paths['img'], f'frame_{i:04d}.png'))
            if not hasattr(self, 'original_shape'):
                self.original_shape = img.shape[:2]
                H_output, W_output = self.get_output_shape(
                    multiple=64, max_width=self.opt.max_width)
                self.output_shape = (H_output, W_output)
                H_opt, W_opt = self.get_output_shape(
                    multiple=32, max_width=self.opt.max_width_opt)
                self.opt_shape = (H_opt, W_opt)
            H_output, W_output = self.output_shape
            H_opt, W_opt = self.opt_shape
            img = imresize(img, (H_opt, W_opt),
                           preserve_range=True).astype(float)
            img_orig = imresize(img, (H_output, W_output),
                                preserve_range=True).astype(float)
            self.images.append(img)
            self.images_orig.append(img_orig)
        for i in range(1, self.number_of_frames+1):
            if self.opt.not_load_mask:
                seg = np.ones([H_opt, W_opt], dtype=float)
            else:
                seg = util_sintel_io.seg_read(
                    join(self.paths['seg'], f'frame_{i:04d}.png'))
                seg = 1-seg
                seg = imresize(seg, (H_opt, W_opt),
                               preserve_range=True).astype(float)
                seg = np.where(seg > 0.99, 1, 0)
            self.masks.append(seg[None, ...])
        self.images = np.stack(self.images, axis=0)
        self.images = torch.from_numpy(self.images).permute(
            [0, 3, 1, 2]).float().pin_memory()
        self.images_orig = np.stack(self.images_orig, axis=0)
        self.images_orig = torch.from_numpy(self.images_orig).permute(
            [0, 3, 1, 2]).float().pin_memory()
        self.masks = np.stack(self.masks, axis=0)
        self.masks = torch.from_numpy(self.masks).float().pin_memory()
        self.number_of_frames = len(self.images)

    def read_intrinsics(self):
        print('reading intrinsics....')
        cam = util_sintel_io.cam_read(
            join(self.paths['cam'], f'frame_0001.cam'))
        K, R, t = util_sintel_io.process_sintel_cam(cam)
        self.K = K
        H_opt, W_opt = self.opt_shape
        H, W = self.original_shape
        self.K[0, :] *= W_opt/W
        self.K[1, :] *= H_opt/H
        print('intrinsics loaded:')
        print(self.K)
        self.K = torch.from_numpy(self.K[None, ...]).float().pin_memory()

    def read_gt_info(self):
        print('reading gt info')
        self.R_gt = []
        self.t_gt = []
        self.depth_gt = []
        H_output, W_output = self.output_shape
        for i in range(1, self.number_of_frames+1):
            depth = util_sintel_io.depth_read(
                join(self.paths['depth'], f'frame_{i:04d}.dpt'))
            depth = imresize(depth, (H_output, W_output),
                             preserve_range=True).astype(float)
            self.depth_gt.append(depth)
        for i in range(1, self.number_of_frames+1):
            cam = util_sintel_io.cam_read(
                join(self.paths['cam'], f'frame_{i:04d}.cam'))
            K, R, t = util_sintel_io.process_sintel_cam(cam)
            self.R_gt.append(R)
            self.t_gt.append(t)

        self.R_gt = np.stack(self.R_gt, axis=0)
        self.t_gt = np.stack(self.t_gt, axis=0)
        self.depth_gt = np.stack(self.depth_gt, axis=0)
        self.R_gt = torch.from_numpy(self.R_gt).float().pin_memory()
        self.t_gt = torch.from_numpy(self.t_gt).float().pin_memory()
        self.depth_gt = torch.from_numpy(
            self.depth_gt[:, None, ...]).float().pin_memory()
        print('done')


class DavisLocalVideoDataset(DepthVideoDataset):
    def __init__(self, opt):
        super().__init__()
        self.get_paths(opt)
        self.initialize(opt)

    # def initialize(self, opt):
    #    super().initialize(opt)

    def get_paths(self, opt):
        track_name = opt.track_name
        data_list_root = davis_path + '/JPEGImages/Full-Resolution'
        mask_root = davis_path + '/Annotations/Full-Resolution'
        image_path = join(data_list_root, f'{track_name}')
        self.paths = {'image_path': image_path,
                      'mask_path': join(mask_root, f'{track_name}')}

    def read_images(self):
        print('reading images...')
        img_path = self.paths['image_path']
        image_paths = sorted(glob(join(img_path, '*.jpg')))
        if len(image_paths) == 0:
            image_paths = sorted(glob(join(img_path, '*.png')))
        image_paths = image_paths[::self.opt.image_sequence_stride]
        image_paths = image_paths[:self.opt.frame_cap]
        self.images = []
        self.images_orig = []
        for img_path in tqdm(image_paths):
            img_raw = np.asarray(Image.open(
                img_path).convert('RGB')).astype(float)/255
            if not hasattr(self, 'original_shape'):
                self.original_shape = img_raw.shape[:2]
                H_output, W_output = self.get_output_shape(
                    multiple=64, max_width=self.opt.max_width)
                self.output_shape = (H_output, W_output)
                H_opt, W_opt = self.get_output_shape(
                    multiple=32, max_width=self.opt.max_width_opt)
                self.opt_shape = (H_opt, W_opt)
            H_output, W_output = self.output_shape
            H_opt, W_opt = self.opt_shape
            img = imresize(img_raw, (H_opt, W_opt),
                           preserve_range=True).astype(float)
            img_orig = imresize(img_raw, (H_output, W_output),
                                preserve_range=True).astype(float)
            self.images.append(img.copy())
            self.images_orig.append(img_orig.copy())
        self.number_of_frames = len(self.images)
        self.images = np.stack(self.images, axis=0)
        self.images = torch.from_numpy(self.images).permute(
            [0, 3, 1, 2]).float().pin_memory()
        self.images_orig = np.stack(self.images_orig, axis=0)
        self.images_orig = torch.from_numpy(self.images_orig).permute(
            [0, 3, 1, 2]).float().pin_memory()
        self.masks = torch.ones(
            [self.number_of_frames, 1, H_opt, W_opt]).float().pin_memory()

        print('done.')

    def read_intrinsics(self):
        pass

    def read_gt_info(self):
        if self.opt.not_load_mask:
            return
        else:
            print('reading gt info...')
            self.masks = []
            mask_paths = sorted(glob(join(self.paths['mask_path'], '*.png')))
            mask_paths = mask_paths[::self.opt.image_sequence_stride]
            for mask_path in mask_paths:
                mask = np.asarray(Image.open(
                    mask_path).convert('L')).astype(float)/255
                mask = imresize(mask, self.opt_shape,
                                preserve_range=True).astype(float)
                mask = np.where(mask > 0.001, 1, 0)
                mask = 1-mask
                mask = torch.from_numpy(mask[None, ...]).float().pin_memory()
                self.masks.append(mask)
            self.masks = np.stack(self.masks, axis=0)
            self.masks = torch.from_numpy(self.masks).float().pin_memory()
            print('done.')
        # print('reading gt info')
        # depth_and_pose_path = self.paths['depth_and_pose_paths']
        # self.R_gt = []
        # self.t_gt = []
        # self.depth_gt = []
        # H, W = self.output_shape
        # for x in range(self.number_of_frames):
        #     np_file = np.load(depth_and_pose_path[x], allow_pickle=True)
        #     depth_gt = np_file['depth']+1e-6
        #     depth_gt = imresize(depth_gt, (H, W), preserve_range=True)
        #     extrinsics = np.linalg.inv(np_file['w2c']).astype(float)
        #     self.R_gt.append(extrinsics[:3, :3])
        #     self.t_gt.append(extrinsics[:3, 3:4])
        #     self.depth_gt.append(depth_gt)
        #     del np_file
        # self.R_gt = np.stack(self.R_gt, axis=0)
        # self.t_gt = np.stack(self.t_gt, axis=0)
        # self.depth_gt = np.stack(self.depth_gt, axis=0)
        # self.R_gt = torch.from_numpy(self.R_gt).float().pin_memory()
        # self.t_gt = torch.from_numpy(self.t_gt).float().pin_memory()
        # self.depth_gt = torch.from_numpy(
        #     self.depth_gt[:, None, ...]).float().pin_memory()
        # print('done')


class DavisVideoDataset(DepthVideoDataset):
    def __init__(self, opt):
        super().__init__()
        self.get_paths(opt)
        self.initialize(opt)

    # def initialize(self, opt):
    #    super().initialize(opt)

    def get_paths(self, opt):
        track_name = opt.track_name
        data_list_root = davis_path + '/JPEGImages/Full-Resolution'
        mask_root = davis_path + '/Annotations/Full-Resolution'
        image_path = join(data_list_root, f'{track_name}')
        self.paths = {'image_path': image_path,
                      'mask_path': join(mask_root, f'{track_name}')}

    def read_images(self):
        print('reading images...')
        img_path = self.paths['image_path']
        image_paths = sorted(glob(join(img_path, '*.jpg')))
        if len(image_paths) == 0:
            image_paths = sorted(glob(join(img_path, '*.png')))
        image_paths = image_paths[::self.opt.image_sequence_stride]
        image_paths = image_paths[:self.opt.frame_cap]
        self.images = []
        self.images_orig = []
        for img_path in tqdm(image_paths):
            img_raw = np.asarray(Image.open(
                img_path).convert('RGB')).astype(float)/255
            if not hasattr(self, 'original_shape'):
                self.original_shape = img_raw.shape[:2]
                H_output, W_output = self.get_output_shape(
                    multiple=64, max_width=self.opt.max_width)
                self.output_shape = (H_output, W_output)
                H_opt, W_opt = self.get_output_shape(
                    multiple=32, max_width=self.opt.max_width_opt)
                self.opt_shape = (H_opt, W_opt)
            H_output, W_output = self.output_shape
            H_opt, W_opt = self.opt_shape
            img = imresize(img_raw, (H_opt, W_opt),
                           preserve_range=True).astype(float)
            img_orig = imresize(img_raw, (H_output, W_output),
                                preserve_range=True).astype(float)
            self.images.append(img.copy())
            self.images_orig.append(img_orig.copy())
        self.number_of_frames = len(self.images)
        self.images = np.stack(self.images, axis=0)
        self.images = torch.from_numpy(self.images).permute(
            [0, 3, 1, 2]).float().pin_memory()
        self.images_orig = np.stack(self.images_orig, axis=0)
        self.images_orig = torch.from_numpy(self.images_orig).permute(
            [0, 3, 1, 2]).float().pin_memory()
        self.masks = torch.ones(
            [self.number_of_frames, 1, H_opt, W_opt]).float().pin_memory()

        print('done.')

    def read_intrinsics(self):
        pass

    def read_gt_info(self):
        if self.opt.not_load_mask:
            return
        else:
            print('reading gt info...')
            self.masks = []
            mask_paths = sorted(glob(join(self.paths['mask_path'], '*.png')))
            mask_paths = mask_paths[::self.opt.image_sequence_stride]
            for mask_path in mask_paths:
                mask = np.asarray(Image.open(
                    mask_path).convert('L')).astype(float)/255
                mask = imresize(mask, self.opt_shape,
                                preserve_range=True).astype(float)
                mask = np.where(mask > 0.001, 1, 0)
                mask = 1-mask
                mask = torch.from_numpy(mask[None, ...]).float().pin_memory()
                self.masks.append(mask)
            self.masks = np.stack(self.masks, axis=0)
            self.masks = torch.from_numpy(self.masks).float().pin_memory()
            print('done.')
        # print('reading gt info')
        # depth_and_pose_path = self.paths['depth_and_pose_paths']
        # self.R_gt = []
        # self.t_gt = []
        # self.depth_gt = []
        # H, W = self.output_shape
        # for x in range(self.number_of_frames):
        #     np_file = np.load(depth_and_pose_path[x], allow_pickle=True)
        #     depth_gt = np_file['depth']+1e-6
        #     depth_gt = imresize(depth_gt, (H, W), preserve_range=True)
        #     extrinsics = np.linalg.inv(np_file['w2c']).astype(float)
        #     self.R_gt.append(extrinsics[:3, :3])
        #     self.t_gt.append(extrinsics[:3, 3:4])
        #     self.depth_gt.append(depth_gt)
        #     del np_file
        # self.R_gt = np.stack(self.R_gt, axis=0)
        # self.t_gt = np.stack(self.t_gt, axis=0)
        # self.depth_gt = np.stack(self.depth_gt, axis=0)
        # self.R_gt = torch.from_numpy(self.R_gt).float().pin_memory()
        # self.t_gt = torch.from_numpy(self.t_gt).float().pin_memory()
        # self.depth_gt = torch.from_numpy(
        #     self.depth_gt[:, None, ...]).float().pin_memory()
        # print('done')


class LocalVideoDataset(DavisVideoDataset):

    def get_paths(self, opt):
        assert opt.localdata_path is not None
        self.paths = {'image_path': opt.localdata_path,
                      'mask_path': None}


class TUMVideoDataset(DepthVideoDataset):
    def __init__(self, opt):
        super().__init__()
        self.get_paths(opt)
        self.initialize(opt)

    def get_paths(self, opt):
        track_name = opt.track_name
        data_list_root = "/data/vision/billf/scratch/ztzhang/data/layered-video/TUM/raw_data"
        image_path = join(
            data_list_root, f'{track_name}', f'rgb_{opt.image_sequence_stride}', 'image_float_tensor.npy')
        self.paths = {'image_path': image_path}

    def read_images(self):
        print('reading images...')
        img_path = self.paths['image_path']
        self.images_raw = np.load(img_path).astype(float)/255
        if self.opt.frame_cap is not None:
            self.images_raw = self.images_raw[:self.opt.frame_cap, ...]
        if self.opt.track_name == 'soup_can':
            self.images_raw = np.transpose(self.images_raw, [0, 2, 1, 3])
        self.images = []
        self.images_orig = []
        for image_id in tqdm(range(self.images_raw.shape[0])):
            img_raw = self.images_raw[image_id]
            if not hasattr(self, 'original_shape'):
                self.original_shape = img_raw.shape[:2]
                H_output, W_output = self.get_output_shape(
                    multiple=64, max_width=self.opt.max_width)
                self.output_shape = (H_output, W_output)
                H_opt, W_opt = self.get_output_shape(
                    multiple=32, max_width=self.opt.max_width_opt)
                self.opt_shape = (H_opt, W_opt)
            H_output, W_output = self.output_shape
            H_opt, W_opt = self.opt_shape
            img = imresize(img_raw, (H_opt, W_opt),
                           preserve_range=True).astype(float)
            img_orig = imresize(img_raw, (H_output, W_output),
                                preserve_range=True).astype(float)
            self.images.append(img.copy())
            self.images_orig.append(img_orig.copy())
        self.number_of_frames = len(self.images)
        self.images = np.stack(self.images, axis=0)
        self.images = torch.from_numpy(self.images).permute(
            [0, 3, 1, 2]).float().pin_memory()
        self.images_orig = np.stack(self.images_orig, axis=0)
        self.images_orig = torch.from_numpy(self.images_orig).permute(
            [0, 3, 1, 2]).float().pin_memory()
        self.masks = torch.ones(
            [self.number_of_frames, 1, H_opt, W_opt]).float().pin_memory()
        print('done.')

    def read_intrinsics(self):
        print('reading intrinsics....')
        self.K = np.eye(3)
        fx, fy, cx, cy = [535.4, 539.2, 320.1, 247.6]
        self.K[0, 0] = fx
        self.K[1, 1] = fy
        self.K[0, 2] = cx
        self.K[1, 2] = cy

        # self.K = np.load(self.paths['depth_and_pose_paths'][0])['K']
        H_original, W_original = self.original_shape
        H, W = self.opt_shape
        self.K[0, :] /= W_original/W
        self.K[1, :] /= H_original/H
        print('intrinsics loaded:')
        print(self.K)
        self.K = torch.from_numpy(self.K[None, ...]).float().pin_memory()

    def read_gt_info(self):
        pass


class StaticTUMVideoDataset(DepthVideoDataset):
    def __init__(self, opt):
        super().__init__()
        self.get_paths(opt)
        self.initialize(opt)

    def get_paths(self, opt):
        track_name = opt.track_name
        data_list_root = "/data/vision/billf/scratch/ztzhang/data/layered-video/TUM_static/raw_data"
        image_path = join(
            data_list_root, f'{track_name}', f'rgb_{opt.image_sequence_stride}', 'image_float_tensor.npy')
        self.paths = {'image_path': image_path}

    def read_images(self):
        print('reading images...')
        img_path = self.paths['image_path']
        self.images_raw = np.load(img_path).astype(float)/255
        # need to crop out the border
        self.images_raw = self.images_raw[:, 16:-16, 32:-32, :]
        if self.opt.frame_cap is not None:
            self.images_raw = self.images_raw[:self.opt.frame_cap, ...]
        self.images = []
        self.images_orig = []
        for image_id in tqdm(range(self.images_raw.shape[0])):
            img_raw = self.images_raw[image_id]
            if not hasattr(self, 'original_shape'):
                self.original_shape = img_raw.shape[:2]
                H_output, W_output = self.get_output_shape(
                    multiple=64, max_width=self.opt.max_width)
                self.output_shape = (H_output, W_output)
                H_opt, W_opt = self.get_output_shape(
                    multiple=32, max_width=self.opt.max_width_opt)
                self.opt_shape = (H_opt, W_opt)
            H_output, W_output = self.output_shape
            H_opt, W_opt = self.opt_shape
            img = imresize(img_raw, (H_opt, W_opt),
                           preserve_range=True).astype(float)
            img_orig = imresize(img_raw, (H_output, W_output),
                                preserve_range=True).astype(float)
            self.images.append(img.copy())
            self.images_orig.append(img_orig.copy())
        self.number_of_frames = len(self.images)
        self.images = np.stack(self.images, axis=0)
        self.images = torch.from_numpy(self.images).permute(
            [0, 3, 1, 2]).float().pin_memory()
        self.images_orig = np.stack(self.images_orig, axis=0)
        self.images_orig = torch.from_numpy(self.images_orig).permute(
            [0, 3, 1, 2]).float().pin_memory()
        self.masks = torch.ones(
            [self.number_of_frames, 1, H_opt, W_opt]).float().pin_memory()
        print('done.')

    def read_intrinsics(self):
        print('reading intrinsics....')
        self.K = np.eye(3)
        fx, fy, cx, cy = [517.3, 516.5, 318.6, 255.3]
        self.K[0, 0] = fx
        self.K[1, 1] = fy
        self.K[0, 2] = cx-32
        self.K[1, 2] = cy-16
        H_original, W_original = self.original_shape
        H, W = self.opt_shape
        self.K[0, :] /= W_original/W
        self.K[1, :] /= H_original/H
        print('intrinsics loaded:')
        print(self.K)
        self.K = torch.from_numpy(self.K[None, ...]).float().pin_memory()

    def read_gt_info(self):
        pass

# %%
