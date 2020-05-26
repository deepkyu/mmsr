import os
import os.path as osp
import torch
import torchvision
import argparse
import yaml
import tqdm
import glob

import utils.util as util
import data.util as data_util
import models.archs.EDVR_arch as EDVR_arch


class EDVRWrapper:
    def __init__(self, **conf):
        self.CUDA_VISIBLE_DEVICES = 0
        self.device = 'cpu' if self.CUDA_VISIBLE_DEVICES is None else 'cuda'
        self.ckpt_path = '../experiments/pretrained_models/EDVR_latest.pth'
        self.padding = 'new_info'
        self.batch = 4
        self.split_H = 540
        self.split_W = 960
        self.network_conf = {
            'nf': 64,
            'nframes': 5,
            'groups': 8,
            'front_RBs': 5,
            'back_RBs': 10,
            'predeblur': False,
            'HR_in': False,
            'w_TSA': True,
        }
        for k, v in conf.items():
            setattr(self, k, v)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.CUDA_VISIBLE_DEVICES)
        with torch.no_grad():
            self.model = EDVR_arch.EDVR(**self.network_conf)

            # set up the models
            self.model.load_state_dict(torch.load(self.ckpt_path), strict=True)
            self.model.eval()
            self.model = self.model.to(self.device)

    def __call__(self, input_path, output_path):
        with torch.no_grad():
            # read LQ images
            imgs_LQ, _, info = torchvision.io.read_video(input_path)  # imgs_LQ: Tensor[T,H,W,C]
            imgs_LQ = imgs_LQ.permute(0, 3, 1, 2).contiguous() / 255.0  # imgs_LQ: Tensor[T,C,H,W]
            max_idx = imgs_LQ.shape[0]
            input_batch = list()
            output_tensor = list()

            # process each image
            for img_idx in tqdm.tqdm(range(max_idx)):
                select_idx = data_util.index_generation(img_idx, max_idx, self.network_conf['nframes'],
                                                        padding=self.padding)
                # imgs_in: Tensor[nframes,C,H,W]
                imgs_in = imgs_LQ.index_select(0, torch.LongTensor(select_idx))
                input_batch.append(imgs_in)
                if (img_idx + 1) % self.batch == 0 and len(input_batch) != 0:
                    input_tensor = torch.stack(input_batch, dim=0)  # Tensor[B,nframes,C,H,W]
                    output = self.single_inference(input_tensor)
                    output_tensor.append(output)
                    input_batch = list()
                    del input_tensor
            if len(input_batch) != 0:
                input_tensor = torch.stack(input_batch, dim=0)
                output = self.single_inference(input_tensor)
                output_tensor.append(output)
                del input_tensor
            del input_batch
            del imgs_LQ

            output_tensor = torch.cat(output_tensor, dim=0)  # output_tensor: Tensor[T,C,H,W]

            # write video
            output = output_tensor.permute(0, 2, 3, 1)  # output: Tensor[T,H,W,C]
            torchvision.io.write_video(output_path, output, fps=info['video_fps'])

    def single_inference(self, input_tensor):
        h_outputs = list()
        h_tensors = input_tensor.split(self.split_H, dim=-2)
        for h_tensor in h_tensors:
            w_outputs = list()
            split_tensors = h_tensor.split(self.split_W, dim=-1)
            for split_tensor in split_tensors:
                output = util.single_forward(self.model, split_tensor.to(self.device))  # output: Tensor[B,1,C,H,W]
                output = output.squeeze(1).float().to('cpu').clamp_(0, 1)  # clamp / output: Tensor[B,C,H,W]
                output = (output * 255.0).round().type(torch.uint8)
                w_outputs.append(output)
            h_outputs.append(torch.cat(w_outputs, dim=-1))
        result = torch.cat(h_outputs, dim=-2)
        return result


def get_output_path(input_path, output_path=None):
    if output_path is None:
        dir_, base = osp.split(input_path)
        filename, ext = osp.splitext(base)
        output_path = os.path.join(dir_, '{}_output{}'.format(filename, ext))
    else:
        output_path = output_path
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for config.")
    parser.add_argument('-i', '--input', type=str, required=True,
                        help="path to input video file")
    parser.add_argument('-o', '--output', type=str, default=None,
                        help="path to output video file")
    parser.add_argument('-I', '--input_list_path', type=str, default=None,
                        help="path to input videos (Unix style pathname pattern expansion)")
    args = parser.parse_args()
    with open(args.config) as f:
        conf = yaml.load(f, Loader=yaml.Loader)
    wrapper = EDVRWrapper(**conf)
    if args.input_list_path is None:
        output_path = get_output_path(args.input, args.output)
        wrapper(args.input, output_path)
    else:
        input_path_list = glob.glob(args.input_list_path, recursive=True)
        for input_path in input_path_list:
            output_path = get_output_path(input_path)
            wrapper(input_path, output_path)


if __name__ == '__main__':
    main()
