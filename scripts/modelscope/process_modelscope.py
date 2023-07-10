# Function calls referenced from https://github.com/modelscope/modelscope/tree/master/modelscope/pipelines/multi_modal

# Copyright (C) 2023 by Artem Khrapov (kabachuha)
# Read LICENSE for usage terms.

from base64 import b64encode
from tqdm import tqdm
from PIL import Image
from modelscope.t2v_pipeline import TextToVideoSynthesis, tensor2vid
from t2v_helpers.key_frames import T2VAnimKeys  # TODO: move to deforum_tools
from pathlib import Path
import numpy as np
import torch
import cv2
import gc
import modules.paths as ph
from types import SimpleNamespace
from t2v_helpers.general_utils import get_t2v_version, get_model_location
import time, math
from t2v_helpers.video_audio_utils import ffmpeg_stitch_video, get_quick_vid_info, vid2frames, duplicate_pngs_from_folder, clean_folder_name
from t2v_helpers.args import get_outdir, process_args
import t2v_helpers.args as t2v_helpers_args
from modules import shared, sd_hijack, lowvram
from modules.shared import opts, devices, state
import os

pipe = None

def setup_pipeline(model_name):
    return TextToVideoSynthesis(get_model_location(model_name))

def process_modelscope(args_dict):
    args, video_args = process_args(args_dict)

    global pipe
    print(f"\033[4;33m text2video extension for auto1111 webui\033[0m")
    print(f"Git commit: {get_t2v_version()}")
    init_timestring = time.strftime('%Y%m%d%H%M%S')
    outdir_current = os.path.join(get_outdir(), f"{init_timestring}")

    max_vids_to_pack = opts.data.get("modelscope_deforum_show_n_videos") if opts.data is not None and opts.data.get("modelscope_deforum_show_n_videos") is not None else -1
    cpu_vae = opts.data.get("modelscope_deforum_vae_settings") if opts.data is not None and opts.data.get("modelscope_deforum_vae_settings") is not None else 'GPU (half precision)'
    if shared.sd_model is not None:
        sd_hijack.model_hijack.undo_hijack(shared.sd_model)
        try:
            lowvram.send_everything_to_cpu()
        except Exception as e:
            pass
        # the following command actually frees the GPU vram from the sd.model, no need to do del shared.sd_model 22-05-23
        shared.sd_model = None
    gc.collect()
    devices.torch_gc()

    print('Starting text2video')
    print('Pipeline setup')

    # optionally store pipe in global between runs
    # also refresh the model if the user selected a newer one
    # if args.model is none (e.g. an API call, the model stays as the previous one)
    if pipe is None and args.model is None: # one more API call hack, falling back to <modelscope> if never used TODO: figure out how to permastore the model name the best way
        args.model = "<modelscope>"
        print(f"WARNING: received an API call with an empty model name, defaulting to {args.model} at {get_model_location(args.model)}")
    if pipe is None or pipe is not None and args.model is not None and get_model_location(args.model) != pipe.model_dir:
        pipe = setup_pipeline(args.model)

    pipe.keep_in_vram = opts.data.get("modelscope_deforum_keep_model_in_vram") if opts.data is not None and opts.data.get("modelscope_deforum_keep_model_in_vram") is not None else 'None'

    device = devices.get_optimal_device()
    print('device', device)

    mask = None

    if args.do_vid2vid:
        if args.vid2vid_frames is None and args.vid2vid_frames_path == "":
            raise FileNotFoundError("Please upload a video :()")

        # Overrides
        if args.vid2vid_frames is not None:
            vid2vid_frames_path = args.vid2vid_frames.name

        print("got a request to *vid2vid* an existing video.")

        in_vid_fps, _, _ = get_quick_vid_info(vid2vid_frames_path)
        folder_name = clean_folder_name(Path(vid2vid_frames_path).stem)
        outdir_no_tmp = os.path.join(os.getcwd(), 'outputs', 'frame-vid2vid', folder_name)
        i = 1
        while os.path.exists(outdir_no_tmp):
            outdir_no_tmp = os.path.join(os.getcwd(), 'outputs', 'frame-vid2vid', folder_name + '_' + str(i))
            i += 1

        outdir_v2v = os.path.join(outdir_no_tmp, 'tmp_input_frames')
        os.makedirs(outdir_v2v, exist_ok=True)

        vid2frames(video_path=vid2vid_frames_path, video_in_frame_path=outdir_v2v, overwrite=True, extract_from_frame=args.vid2vid_startFrame, extract_to_frame=args.vid2vid_startFrame + args.frames,
                   numeric_files_output=True, out_img_format='png')

        temp_convert_raw_png_path = os.path.join(outdir_v2v, "tmp_vid2vid_folder")
        duplicate_pngs_from_folder(outdir_v2v, temp_convert_raw_png_path, None, folder_name)

        videogen = []
        for f in os.listdir(temp_convert_raw_png_path):
            # double check for old _depth_ files, not really needed probably but keeping it for now
            if '_depth_' not in f:
                videogen.append(f)

        videogen.sort(key=lambda x: int(x.split('.')[0]))

        images = []
        for file in tqdm(videogen, desc="Loading frames"):
            image = Image.open(os.path.join(temp_convert_raw_png_path, file))
            image = image.resize((args.width, args.height), Image.ANTIALIAS)
            array = np.array(image)
            images += [array]

        # print(images)

        images = np.stack(images)  # f h w c
        batches = 1
        n_images = np.tile(images[np.newaxis, ...], (batches, 1, 1, 1, 1))  # n f h w c
        bcfhw = n_images.transpose(0, 4, 1, 2, 3)
        # convert to 0-1 float
        bcfhw = bcfhw.astype(np.float32) / 255
        bfchw = bcfhw.transpose(0, 2, 1, 3, 4)  # b c f h w

        print(f"Converted the frames to tensor {bfchw.shape}")

        vd_out = torch.from_numpy(bcfhw).to("cuda")

        # should be -1,1, not 0,1
        vd_out = 2 * vd_out - 1

        # latents should have shape num_sample, 4, max_frames, latent_h,latent_w
        print("Computing latents")
        latents = pipe.compute_latents(vd_out).to(device)

        skip_steps = int(math.floor(args.steps * max(0, min(1 - args.strength, 1))))
    else:
        latents = None
        args.strength = 1
        skip_steps = 0

    print('Working in txt2vid mode' if not args.do_vid2vid else 'Working in vid2vid mode')

    # Start the batch count loop
    pbar = tqdm(range(args.batch_count), leave=False)
    if args.batch_count == 1:
        pbar.disable = True

    vids_to_pack = []

    state.job_count = args.batch_count

    for batch in pbar:
        state.job_no = batch
        if state.skipped:
            state.skipped = False

        if state.interrupted:
            break

        shared.state.job = f"Batch {batch + 1} out of {args.batch_count}"
        # TODO: move to a separate function
        if args.inpainting_frames > 0 and hasattr(args.inpainting_image, "name"):
            keys = T2VAnimKeys(SimpleNamespace(**{'max_frames': args.frames, 'inpainting_weights': args.inpainting_weights}), args.seed, args.inpainting_frames)
            images = []
            print("Received an image for inpainting", args.inpainting_image.name)
            for i in range(args.frames):
                image = Image.open(args.inpainting_image.name).convert("RGB")
                image = image.resize((args.width, args.height), Image.ANTIALIAS)
                array = np.array(image)
                images += [array]

            images = np.stack(images)  # f h w c
            batches = 1
            n_images = np.tile(images[np.newaxis, ...], (batches, 1, 1, 1, 1))  # n f h w c
            bcfhw = n_images.transpose(0, 4, 1, 2, 3)
            # convert to 0-1 float
            bcfhw = bcfhw.astype(np.float32) / 255
            bfchw = bcfhw.transpose(0, 2, 1, 3, 4)  # b c f h w

            print(f"Converted the frames to tensor {bfchw.shape}")

            vd_out = torch.from_numpy(bcfhw).to("cuda")

            # should be -1,1, not 0,1
            vd_out = 2 * vd_out - 1

            # latents should have shape num_sample, 4, max_frames, latent_h,latent_w
            # but right now they have shape num_sample=1,4, 1 (only used 1 img), latent_h, latent_w
            print("Computing latents")
            image_latents = pipe.compute_latents(vd_out).numpy()
            # padding_width = [(0, 0), (0, 0), (0, frames-inpainting_frames), (0, 0), (0, 0)]
            # padded_latents = np.pad(image_latents, pad_width=padding_width, mode='constant', constant_values=0)

            latent_h = args.height // 8
            latent_w = args.width // 8
            latent_noise = np.random.normal(size=(1, 4, args.frames, latent_h, latent_w))
            mask = np.ones(shape=(1, 4, args.frames, latent_h, latent_w))

            mask_weights = [keys.inpainting_weights_series[frame_idx] for frame_idx in range(args.frames)]

            for i in range(args.frames):
                v = mask_weights[i]
                mask[:, :, i, :, :] = v

            masked_latents = image_latents * (1 - mask) + latent_noise * mask

            latents = torch.tensor(masked_latents).to(device)

            mask = torch.tensor(mask).to(device)

            args.strength = 1

        samples, vs, _ = pipe.infer(args.prompt, args.n_prompt, args.steps, args.frames, args.seed + batch if args.seed != -1 else -1, args.cfg_scale,
                                args.width, args.height, args.eta, cpu_vae, device, latents, strength=args.strength, skip_steps=skip_steps, mask=mask, is_vid2vid=args.do_vid2vid, sampler=args.sampler)

        
        if batch > 0:
            outdir_current = os.path.join(get_outdir(), f"{init_timestring}_{batch}")
        print(f'text2video finished, saving frames to {outdir_current}')

        # just deleted the folder so we need to make it again
        os.makedirs(outdir_current, exist_ok=True)
        for i in range(len(samples)):
            cv2.imwrite(outdir_current + os.path.sep +
                        f"{i:06}.png", samples[i])

        args_file = os.path.join(outdir_current,'args.txt')
        with open(args_file, 'w') as f:
          for key, value in vs.items(): 
            f.write('%s:%s\n' % (key, value))
        print(f'saving args to {args_file}')

        # TODO: add params to the GUI
        if not video_args.skip_video_creation:
            ffmpeg_stitch_video(ffmpeg_location=video_args.ffmpeg_location, fps=video_args.fps, outmp4_path=outdir_current + os.path.sep + f"vid.mp4", imgs_path=os.path.join(outdir_current,
                                                                                                                                                                              "%06d.png"),
                                stitch_from_frame=0, stitch_to_frame=-1, add_soundtrack=video_args.add_soundtrack,
                                audio_path=vid2vid_frames_path if video_args.add_soundtrack == 'Init Video' else video_args.soundtrack_path, crf=video_args.ffmpeg_crf, preset=video_args.ffmpeg_preset)
        print(f't2v complete, result saved at {outdir_current}')

        mp4 = open(outdir_current + os.path.sep + f"vid.mp4", 'rb').read()
        dataurl = "data:video/mp4;base64," + b64encode(mp4).decode()

        if max_vids_to_pack == -1 or len(vids_to_pack) < max_vids_to_pack:
            vids_to_pack.append(dataurl)
    t2v_helpers_args.i1_store_t2v = f'<p style=\"font-weight:bold;margin-bottom:0em\">text2video extension for auto1111 — version 1.2b </p>'
    for dataurl in vids_to_pack:
        t2v_helpers_args.i1_store_t2v += f'<video controls loop><source src="{dataurl}" type="video/mp4"></video><br>'
    pbar.close()
    return vids_to_pack
