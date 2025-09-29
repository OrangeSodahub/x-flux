import argparse
from PIL import Image
import os
import torch

import numpy as np
from src.flux.xflux_pipeline_depth import XFluxSampler
from src.flux.util import load_ae, load_clip, load_flow_model2, load_controlnet, load_t5


def create_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt", type=str, required=True,
        help="The input text prompt"
    )
    parser.add_argument(
        "--neg_prompt", type=str, default="",
        help="The input text negative prompt"
    )
    parser.add_argument(
        "--img_prompt", type=str, default=None,
        help="Path to input image prompt"
    )
    parser.add_argument(
        "--neg_img_prompt", type=str, default=None,
        help="Path to input negative image prompt"
    )
    parser.add_argument(
        "--ip_scale", type=float, default=1.0,
        help="Strength of input image prompt"
    )
    parser.add_argument(
        "--neg_ip_scale", type=float, default=1.0,
        help="Strength of negative input image prompt"
    )
    parser.add_argument(
        "--local_path", type=str, default=None,
        help="Local path to the model checkpoint (Controlnet)"
    )
    parser.add_argument(
        "--repo_id", type=str, default=None,
        help="A HuggingFace repo id to download model (Controlnet)"
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="A filename to download from HuggingFace"
    )
    parser.add_argument(
        "--ip_repo_id", type=str, default=None,
        help="A HuggingFace repo id to download model (IP-Adapter)"
    )
    parser.add_argument(
        "--ip_name", type=str, default=None,
        help="A IP-Adapter filename to download from HuggingFace"
    )
    parser.add_argument(
        "--ip_local_path", type=str, default=None,
        help="Local path to the model checkpoint (IP-Adapter)"
    )
    parser.add_argument(
        "--lora_repo_id", type=str, default=None,
        help="A HuggingFace repo id to download model (LoRA)"
    )
    parser.add_argument(
        "--lora_name", type=str, default=None,
        help="A LoRA filename to download from HuggingFace"
    )
    parser.add_argument(
        "--lora_local_path", type=str, default=None,
        help="Local path to the model checkpoint (Controlnet)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (e.g. cpu, cuda:0, cuda:1, etc.)"
    )
    parser.add_argument(
        "--offload", action='store_true', help="Offload model to CPU when not in use"
    )
    parser.add_argument(
        "--use_ip", action='store_true', help="Load IP model"
    )
    parser.add_argument(
        "--use_lora", action='store_true', help="Load Lora model"
    )
    parser.add_argument(
        "--use_controlnet", action='store_true', help="Load Controlnet model"
    )
    parser.add_argument(
        "--num_images_per_prompt", type=int, default=1,
        help="The number of images to generate per prompt"
    )
    parser.add_argument(
        "--depth", type=str, default=None, help="Path to image"
    )
    parser.add_argument(
        "--lora_weight", type=float, default=0.9, help="Lora model strength (from 0 to 1.0)"
    )
    parser.add_argument(
        "--control_weight", type=float, default=0.8, help="Controlnet model strength (from 0 to 1.0)"
    )
    parser.add_argument(
        "--control_type", type=str, default="canny",
        choices=("canny", "openpose", "depth", "zoe", "hed", "hough", "tile"),
        help="Name of controlnet condition, example: canny"
    )
    parser.add_argument(
        "--model_type", type=str, default="flux-dev",
        choices=("flux-dev", "flux-dev-fp8", "flux-schnell"),
        help="Model type to use (flux-dev, flux-dev-fp8, flux-schnell)"
    )
    parser.add_argument(
        "--width", type=int, default=1024, help="The width for generated image"
    )
    parser.add_argument(
        "--height", type=int, default=1024, help="The height for generated image"
    )
    parser.add_argument(
        "--num_steps", type=int, default=25, help="The num_steps for diffusion process"
    )
    parser.add_argument(
        "--guidance", type=float, default=4, help="The guidance for diffusion process"
    )
    parser.add_argument(
        "--seed", type=int, default=123456789, help="A seed for reproducible inference"
    )
    parser.add_argument(
        "--true_gs", type=float, default=3.5, help="true guidance"
    )
    parser.add_argument(
        "--timestep_to_start_cfg", type=int, default=5, help="timestep to start true guidance"
    )
    parser.add_argument(
        "--save_path", type=str, default='results', help="Path to save"
    )
    return parser


def save_depth(depth, save_path):
    import cv2
    def apply_depth_colormap(gray, minmax=None, cmap=cv2.COLORMAP_JET):
        """
        Input:
            gray: gray image, tensor/numpy, (H, W)
        Output:
            depth: (3, H, W), tensor
        """
        if type(gray) is not np.ndarray:
            gray = gray.detach().cpu().numpy().astype(np.float32)
        gray = gray.squeeze()
        assert len(gray.shape) == 2, f'{gray.shape}'
        x = np.nan_to_num(gray)  # change nan to 0
        if minmax is None:
            mi = np.min(x)  # get minimum positive value
            ma = np.max(x)
        else:
            mi, ma = minmax
        x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
        # TODO
        x = 1 - x  # reverse the colormap
        x = (255 * x).astype(np.uint8)
        x = Image.fromarray(cv2.applyColorMap(x, cmap))
        # x = T.ToTensor()(x)  # (3, H, W)
        return x

    depth_cond = apply_depth_colormap(depth)
    depth_cond.save(save_path)


def main(args):

    seed = args.seed

    def get_models(name: str, device, offload: bool, is_schnell: bool):
        t5 = load_t5(device, max_length=256 if is_schnell else 512)
        clip = load_clip(device)
        model = load_flow_model2(name, device="cpu").to(device)
        vae = load_ae(name, device="cpu" if offload else device).to(device)
        return model, vae, t5, clip

    # load components from 'flux-dev'
    dit, vae, t5, clip = get_models(name=args.model_type, device=args.device, offload=False, is_schnell=False)
    print(f'Loaded dit, vae, t5, clip models from {args.model_type}.')
    controlnet = None
    if args.use_controlnet:
        controlnet = load_controlnet(name=args.model_type, device=args.device)
        checkpoint = torch.load(args.local_path, weights_only=True)
        controlnet.load_state_dict(checkpoint, strict=True)
        print(f'Loaded controlnet checkpoint from {args.local_path}.')

    dit.requires_grad_(False)
    vae.requires_grad_(False)
    t5.requires_grad_(False)
    clip.requires_grad_(False)
    if controlnet:
        controlnet.requires_grad_(False)

    # build pipeline
    xflux_pipeline = XFluxSampler(clip, t5, vae, dit, controlnet=controlnet, device=args.device)
    print(f'Loaded pipeline ...')

    # depth condition
    infer_depth = np.load(args.depth)
    infer_prompt = 'A tabletop with a grid tablecloth.'
    infer_sample = {
        'val_depth': infer_depth,
        'val_prompt': infer_prompt,
    }

    os.makedirs(args.save_path, exist_ok=True)
    with torch.no_grad():

        save_depth(infer_depth, os.path.join(args.save_path, 'depth.png'))

        for i in range(args.num_images_per_prompt):

            image = xflux_pipeline.infer_data(infer_sample, seed=seed, dtype=torch.float32)

            # save results
            ind = len(os.listdir(args.save_path))
            Image.fromarray(image).save(os.path.join(args.save_path, f"result_{ind}.png"))

            seed = seed + 1


if __name__ == "__main__":
    args = create_argparser().parse_args()
    main(args)
