import os
import os.path as osp
import torch
import torchvision
import argparse
import yaml

import data.util as data_util
import models.archs.EDVR_arch as EDVR_arch


class EDVRWrapper():
    def __init__(self, **conf):
        self.CUDA_VISIBLE_DEVICES = 0
        self.device = 'cpu' if self.CUDA_VISIBLE_DEVICES is None else 'cuda'
        self.mode = 'blur_bicubic'
        self.ckpt_path = '../experiments/pretrained_models/EDVR_latest.pth'
        self.padding = 'new_info'
        self.save_file = True
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

    def __call__(self, input_path, output_path):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.CUDA_VISIBLE_DEVICES)

        model = EDVR_arch.EDVR(**self.network_conf)

        # set up the models
        model.load_state_dict(torch.load(self.ckpt_path), strict=True)
        model.eval()
        model = model.to(self.device)

        # read LQ images
        imgs_LQ, _, info = torchvision.io.read_video(input_path)  # imgs_LQ: Tensor[T,H,W,C]
        imgs_LQ = imgs_LQ.transpose(1, 3).transpose(2, 3)  # imgs_LQ: Tensor[T,C,H,W]
        max_idx = imgs_LQ.shape[0]
        input_tensor = []

        # process each image
        for img_idx in range(max_idx):
            select_idx = data_util.index_generation(img_idx, max_idx, self.network_conf['nframes'], padding=self.padding)
            imgs_in = imgs_LQ.index_select(0, torch.LongTensor(select_idx))
            input_tensor.append(imgs_in)
        input_tensor = torch.stack(input_tensor, dim=0).to(self.device)  # input_tensor: Tensor[T,nframes,C,H,W]
        output = model(input_tensor) * 255.0  # output: Tensor[T,1,C,H,W]

        # write video
        output = output.unsqueeze().type(torch.uint8).transpose(1, 3).transpose(2, 3)  # output: Tensor[T,W,H,C]
        torchvision.io.write_video(output_path, output, fps=info['video_fps'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for config.")
    parser.add_argument('-i', '--input', type=str, required=True,
                        help="path to input video file")
    parser.add_argument('-o', '--output', type=str, default=None,
                        help="path to output video file")
    args = parser.parse_args()
    with open(args.config) as f:
        conf = yaml.load(f, Loader=yaml.Loader)
    wrapper = EDVRWrapper(**conf)
    if args.output is None:
        dir_, base = osp.split(args.input)
        filename, ext = osp.splitext(base)
        output_path = os.path.join(dir_, '{}_output{}'.format(filename, ext))
    else:
        output_path = args.output
    wrapper(args.input, output_path)


if __name__ == '__main__':
    main()