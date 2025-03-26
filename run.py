import argparse, os
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
import copy

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

import torchvision.transforms as transforms
import torch.nn.functional as F
import time
from tqdm import tqdm
import cv2

feat_maps = []

def feat_merge(opt, input_feats, ref_feats, refL_feats, start_step=0, refinement=False):
    feat_maps = [{'config': {
                'gamma1':opt.gamma1,
                'gamma2':opt.gamma2,
                'beta1':opt.beta1,
                'beta2':opt.beta2,
                'timestep':_,
                'pre_softmax':opt.pre_softmax,
                }} for _ in range(50)]

    for i in range(len(feat_maps)):
        if i < (50 - start_step):
            continue
        input_feat = input_feats[i]
        ref_feat = ref_feats[i]
        refL_feat = refL_feats[i]
        ori_keys = ref_feat.keys()

        for ori_key in ori_keys:
            if ori_key[-1] == 'q':
                feat_maps[i][ori_key+"_input"] = input_feat[ori_key]
            if ori_key[-1] == 'v':
                feat_maps[i][ori_key+"_ref"] = ref_feat[ori_key]
            if ori_key[-1] == 'k':
                feat_maps[i][ori_key+"_input"] = input_feat[ori_key]
                feat_maps[i][ori_key+"_refL"] = refL_feat[ori_key]
                feat_maps[i][ori_key+"_ref"] = ref_feat[ori_key]
    return feat_maps


def load_img(path, device):
    image = Image.open(path).convert("RGB")
    imageL = image.convert("L").convert("RGB")
    x, y = image.size
    print(f"Loaded input image of size ({x}, {y}) from {path}")
    h = w = 512
    images = []
    for image_ in [image, imageL]:
        image_ = image_.resize((w, h), resample=Image.Resampling.LANCZOS)
        image_ = np.array(image_).astype(np.uint8)
        image_ = image_.astype(np.float32) / 255.0
        image_ = image_[None].transpose(0, 3, 1, 2)
        image_ = torch.from_numpy(image_).to(device)
        images.append(2.*image_ - 1.)
    return images[0], images[1]

def adain(input_feat, ref_feat):
    input_mean = input_feat.mean(dim=[0, 2, 3],keepdim=True)
    input_std = input_feat.std(dim=[0, 2, 3],keepdim=True)
    ref_mean = ref_feat.mean(dim=[0, 2, 3],keepdim=True)
    ref_std = ref_feat.std(dim=[0, 2, 3],keepdim=True)
    output = ((input_feat-input_mean)/input_std)*ref_std + ref_mean
    return output

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def init_featmaps(opt):
    global feat_maps
    feat_maps = [{'config': {
                'gamma1':opt.gamma1,
                'gamma2':opt.gamma2,
                'beta1':opt.beta1,
                'beta2':opt.beta2,
                'pre_softmax':opt.pre_softmax,
                }} for _ in range(50)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default = './dataset/main/input')
    parser.add_argument('--ref', default = './dataset/main/reference')
    parser.add_argument('--Tmax', type=int, default=50, help='DDIM eta')
    parser.add_argument('--T', type=int, default=5, help='DDIM eta')
    parser.add_argument('--ddim_eta', type=float, default=0.0, help='DDIM eta')
    parser.add_argument('--H', type=int, default=512, help='image height, in pixel space')
    parser.add_argument('--W', type=int, default=512, help='image width, in pixel space')
    parser.add_argument('--C', type=int, default=4, help='latent channels')
    parser.add_argument('--f', type=int, default=8, help='downsampling factor')
    parser.add_argument('--gamma1', type=float, default=0.5, help='balance between g2g attention and c2c attention (shallower)')
    parser.add_argument('--gamma2', type=float, default=1, help='balance between g2g attention and c2c attention (deeper)')
    parser.add_argument('--beta1', type=float, default=0.5, help='strength of self-attention injection (shallower)')
    parser.add_argument('--beta2', type=float, default=0, help='strength of self-attention injection (deeper)')
    parser.add_argument('--w', type=float, default=10, help='strength of classifier-free colorization guidance')
    parser.add_argument('--N', type=int, default=3, help='number of repetitions')
    parser.add_argument("--attn_layer_encoder", type=str, default='100', help='injection attention feature layers')
    parser.add_argument("--attn_layer_middle", type=str, default='100', help='injection attention feature layers')
    parser.add_argument("--attn_layer_decoder", type=str, default='6,7,8,9,10,11', help='injection attention feature layers')
    parser.add_argument('--model_config', type=str, default='models/ldm/stable-diffusion-v1/v1-inference.yaml', help='model config')
    parser.add_argument('--ckpt', type=str, default='models/ldm/stable-diffusion-v1/model.ckpt', help='model checkpoint')
    parser.add_argument('--precision', type=str, default='autocast', help='choices: ["full", "autocast"]')
    parser.add_argument('--output_path', type=str, default='output')
    parser.add_argument("--without_init_adain", action='store_true')
    parser.add_argument("--store_on_cpu", action='store_true')
    parser.add_argument("--visualization_mode", action='store_true')
    parser.add_argument("--pre_softmax", action='store_true')
    opt = parser.parse_args()

    seed_everything(22)
    output_path = opt.output_path
    os.makedirs(output_path, exist_ok=True)

    model_config = OmegaConf.load(f"{opt.model_config}")
    model = load_model_from_config(model_config, f"{opt.ckpt}")

    if opt.attn_layer_encoder == "":
        self_attn_output_block_indices_encoder = []
    else:
        self_attn_output_block_indices_encoder = list(map(int, opt.attn_layer_encoder.split(',')))
    if opt.attn_layer_middle == "":
        self_attn_output_block_indices_middle = []
    else:
        self_attn_output_block_indices_middle = list(map(int, opt.attn_layer_middle.split(',')))
    self_attn_output_block_indices_decoder = list(map(int, opt.attn_layer_decoder.split(',')))
    ddim_inversion_steps = opt.Tmax
    save_feature_timesteps = ddim_steps = opt.Tmax

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    unet_model = model.model.diffusion_model
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)
    time_range = np.flip(sampler.ddim_timesteps)
    idx_time_dict = {}
    time_idx_dict = {}
    for i, t in enumerate(time_range):
        idx_time_dict[t] = i
        time_idx_dict[i] = t

    seed = torch.initial_seed()
    opt.seed = seed

    init_featmaps(opt)
    global feat_maps

    def ddim_sampler_callback(pred_x0, xt, i):
        save_feature_maps_callback(i)
        save_feature_map(xt, 'z_enc', i)

    def save_feature_maps(blocks, i, feature_type="input_block"):
        block_idx = 0
        for block_idx, block in enumerate(blocks):
            if len(block) > 1 and "SpatialTransformer" in str(type(block[1])):
                if feature_type == "input_block":
                    self_attn_output_block_indices = self_attn_output_block_indices_encoder
                elif feature_type == "middle_block":
                    self_attn_output_block_indices = self_attn_output_block_indices_middle
                elif feature_type == "output_block":
                    self_attn_output_block_indices = self_attn_output_block_indices_decoder
                if block_idx in self_attn_output_block_indices:
                    # self-attn
                    q = block[1].transformer_blocks[0].attn1.q
                    k = block[1].transformer_blocks[0].attn1.k
                    v = block[1].transformer_blocks[0].attn1.v
                    save_feature_map(q, f"{feature_type}_{block_idx}_self_attn_q", i)
                    save_feature_map(k, f"{feature_type}_{block_idx}_self_attn_k", i)
                    save_feature_map(v, f"{feature_type}_{block_idx}_self_attn_v", i)

    def save_feature_maps_callback(i):
        save_feature_maps(unet_model.input_blocks, i, "input_block")
        save_feature_maps([unet_model.middle_block], i, "middle_block")
        save_feature_maps(unet_model.output_blocks, i, "output_block")

    def save_feature_map(feature_map, filename, time):
        global feat_maps
        cur_idx = idx_time_dict[time]
        feat_maps[cur_idx][f"{filename}"] = feature_map

    start_step = opt.T
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    uc = model.get_learned_conditioning([""])
    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
    ref_img_list = sorted(os.listdir(opt.ref))
    input_img_list = sorted(os.listdir(opt.input))

    begin = time.time()
    for ref_name, input_name in zip(ref_img_list, input_img_list):
        ### prepare ref and refL ###
        ref_name_ = os.path.join(opt.ref, ref_name)
        init_ref_image, init_refL_image = load_img(ref_name_, device)

        ### DDIM inversion of ref ###
        seed = -1
        ref_z_enc = None
        init_ref = model.get_first_stage_encoding(model.encode_first_stage(init_ref_image))
        ref_z_enc, _ = sampler.encode_ddim(init_ref.clone(), num_steps=ddim_inversion_steps, unconditional_conditioning=uc, \
                                            end_step=time_idx_dict[ddim_inversion_steps-1-start_step], \
                                            callback_ddim_timesteps=save_feature_timesteps,
                                            img_callback=ddim_sampler_callback)
        ref_feat = copy.deepcopy(feat_maps)

        ### DDIM inversion of refL ###
        seed = -1
        refL_z_enc = None
        init_refL = model.get_first_stage_encoding(model.encode_first_stage(init_refL_image))
        refL_z_enc, _ = sampler.encode_ddim(init_refL.clone(), num_steps=ddim_inversion_steps, unconditional_conditioning=uc, \
                                            end_step=time_idx_dict[ddim_inversion_steps-1-start_step], \
                                            callback_ddim_timesteps=save_feature_timesteps,
                                            img_callback=ddim_sampler_callback)
        refL_feat = copy.deepcopy(feat_maps)
        refL_z_enc = feat_maps[len(feat_maps)-1-start_step]['z_enc']

        ### prepare input ###
        input_name_ = os.path.join(opt.input, input_name)
        init_input_image, _ = load_img(input_name_, device)

        ### DDIM inversion of input ###
        input_feat = None
        init_input = model.get_first_stage_encoding(model.encode_first_stage(init_input_image))
        input_z_enc, _ = sampler.encode_ddim(init_input.clone(), num_steps=ddim_inversion_steps, unconditional_conditioning=uc, \
                                            end_step=time_idx_dict[ddim_inversion_steps-1-start_step], \
                                            callback_ddim_timesteps=save_feature_timesteps,
                                            img_callback=ddim_sampler_callback)
        input_feat = copy.deepcopy(feat_maps)


        ### Colorization ###
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    output_name = f"{os.path.basename(input_name).split('.')[0]}.png"

                    print(f"Inversion end: {time.time() - begin}")

                    if opt.without_init_adain:
                        output = input_z_enc
                    else:
                        output = adain(input_z_enc, ref_z_enc)

                    ### collecting features ###
                    feat_maps = feat_merge(opt, input_feat, ref_feat, refL_feat, start_step=start_step)

                    for n in range(opt.N):
                        for step_at in tqdm(range(ddim_inversion_steps)):
                            ### Dual attention-guided color transfer ###
                            _, color_transferred_intermediates = sampler.sample(S=ddim_steps,
                                                            batch_size=1,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_conditioning=uc,
                                                            eta=opt.ddim_eta,
                                                            x_T=output,
                                                            injected_features=feat_maps,
                                                            start_step=start_step-step_at,
                                                            end_step=start_step-step_at-1,
                                                            disable_tqdm=True,
                                                            log_every_t=1,
                                                            )
                            ### General usage of self-attention ###
                            _, non_color_transferred_intermediates = sampler.sample(S=ddim_steps,
                                                            batch_size=1,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_conditioning=uc,
                                                            eta=opt.ddim_eta,
                                                            x_T=output,
                                                            start_step=start_step-step_at,
                                                            end_step=start_step-step_at-1,
                                                            disable_tqdm=True,
                                                            log_every_t=1,
                                                            )
                            if "e_t" in color_transferred_intermediates.keys():
                                ### Classifier-free colorization guidance ###
                                e_t = color_transferred_intermediates['e_t'] * opt.w + non_color_transferred_intermediates['e_t'] * (1 - opt.w)
                                output = sampler.coef1 * output + sampler.coef2 * e_t

                        x_output = model.decode_first_stage(output)
                        x_output = torch.clamp((x_output + 1.0) / 2.0, min=0.0, max=1.0)
                        x_output = x_output.cpu().permute(0, 2, 3, 1).numpy()

                        ### Post-processing ###
                        x_image_torch = torch.from_numpy(x_output).permute(0, 3, 1, 2)
                        x_image_torch = torch.cat([(init_input_image.cpu() + 1.0) / 2.0, (init_ref_image.cpu() + 1.0) / 2.0, x_image_torch], axis=-1)
                        x_sample = 255. * rearrange(x_image_torch[0].cpu().numpy(), 'c h w -> h w c')
                        img = Image.fromarray(x_sample.astype(np.uint8))

                        lab_input = cv2.cvtColor(x_sample.astype(np.uint8)[:,:512], cv2.COLOR_RGB2LAB)
                        rgb_ref = cv2.resize(x_sample.astype(np.uint8)[:,512:512*2], (128, 128), interpolation=cv2.INTER_AREA)
                        rgb_ref = cv2.resize(rgb_ref, (512, 512))
                        lab_ref = cv2.cvtColor(rgb_ref, cv2.COLOR_RGB2LAB)
                        lab_out = cv2.cvtColor(x_sample.astype(np.uint8)[:,-512:], cv2.COLOR_RGB2LAB)
                        l_input = lab_input[:, :, 0]
                        a_out = lab_out[:, :, 1]
                        b_out = lab_out[:, :, 2]
                        a_out = ((a_out - a_out.mean()) / max(a_out.std(), 0) * lab_ref[:, :, 1].std() + lab_ref[:, :, 1].mean()).astype(np.uint8)
                        b_out = ((b_out - b_out.mean()) / max(b_out.std(), 0) * lab_ref[:, :, 2].std() + lab_ref[:, :, 2].mean()).astype(np.uint8)
                        combined_lab = np.stack([l_input, a_out, b_out], axis=2)
                        combined_rgb = cv2.cvtColor(combined_lab, cv2.COLOR_LAB2RGB)
                        if (not opt.visualization_mode) or n == 0:
                            x_sample_ = combined_rgb
                        else:
                            x_sample_ = np.hstack([x_sample_, combined_rgb])

                        if n+1 != opt.N:
                            rgb_out = torch.tensor(combined_rgb / 255.).cuda() * 2 - 1
                            rgb_out = rgb_out[None].permute(0, 3, 1, 2).to(torch.half)
                            rgb_out = model.get_first_stage_encoding(model.encode_first_stage(rgb_out))

                            output, _ = sampler.encode_ddim(rgb_out.clone(), num_steps=ddim_inversion_steps, unconditional_conditioning=uc, \
                                                                end_step=time_idx_dict[ddim_inversion_steps-1-start_step])

                    img = Image.fromarray(x_sample_.astype(np.uint8))

                    img.save(os.path.join(output_path, output_name))
        init_featmaps(opt)

    print(f"Total end: {time.time() - begin}")

if __name__ == "__main__":
    main()
