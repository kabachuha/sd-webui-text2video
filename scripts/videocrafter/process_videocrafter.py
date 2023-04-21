from base64 import b64encode
from tqdm import tqdm
from omegaconf import OmegaConf
import time, os
from t2v_helpers.general_utils import get_t2v_version
from t2v_helpers.args import get_outdir, process_args
import modules.paths as ph
import t2v_helpers.args as t2v_helpers_args
from modules.shared import state

# VideoCrafter support is heavy WIP and sketchy, needs help and more devs!
def process_videocrafter(args_dict):
    args, video_args = process_args(args_dict)
    print(f"\033[4;33m text2video extension for auto1111 webui\033[0m")
    print(f"Git commit: {get_t2v_version()}")
    init_timestring = time.strftime('%Y%m%d%H%M%S')
    outdir_current = os.path.join(get_outdir(), f"{init_timestring}")

    os.makedirs(outdir_current, exist_ok=True)

    # load & merge config

    config_path = os.path.join(ph.models_path, "models/VideoCrafter/model_config.yaml")
    if not os.path.exists(config_path):
        config_path = os.path.join(os.getcwd(), "extensions/sd-webui-modelscope-text2video/scripts/videocrafter/base_t2v/model_config.yaml")
    if not os.path.exists(config_path):
        config_path = os.path.join(os.getcwd(), "extensions/sd-webui-text2video/scripts/videocrafter/base_t2v/model_config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Could not find config file at {os.path.join(ph.models_path, "models/VideoCrafter/model_config.yaml")}, nor at {os.path.join(os.getcwd(), "extensions/sd-webui-modelscope-text2video/scripts/videocrafter/base_t2v/model_config.yaml")}, nor at {os.path.join(os.getcwd(), "extensions/sd-webui-text2video/scripts/videocrafter/base_t2v/model_config.yaml")}')

    config = OmegaConf.load(config_path)
    print("VideoCrafter config: \n", config)

    from videocrafter.lvdm.samplers.ddim import DDIMSampler
    from videocrafter.sample_utils import load_model, get_conditions, make_model_input_shape, torch_to_np
    from videocrafter.sample_text2video import sample_text2video
    from videocrafter.lvdm.utils.saving_utils import npz_to_video_grid
    from t2v_helpers.video_audio_utils import add_soundtrack

    # get model & sampler
    model, _, _ = load_model(config, ph.models_path+'/VideoCrafter/model.ckpt', #TODO: support safetensors and stuff
                             inject_lora=False, # TODO
                             lora_scale=1, # TODO
                             lora_path=ph.models_path+'/VideoCrafter/LoRA/LoRA.ckpt', #TODO: support LoRA and stuff
                             )
    ddim_sampler = DDIMSampler(model)# if opt.sample_type == "ddim" else None

    # if opt.inject_lora:
    #     assert(opt.lora_trigger_word != '')
    #     prompts = [p + opt.lora_trigger_word for p in prompts]
    
    # go
    start = time.time()  

    pbar = tqdm(range(args.batch_count), leave=False)
    if args.batch_count == 1:
        pbar.disable=True
    
    state.job_count = args.batch_count
    
    for batch in pbar:
        state.job_no = batch + 1
        if state.skipped:
            state.skipped = False

        if state.interrupted:
            break

        state.job = f"Batch {batch+1} out of {args.batch_count}"
        ddim_sampler.noise_gen.manual_seed(args.seed + batch if args.seed != -1 else -1)
        # sample
        samples = sample_text2video(model, args.prompt, args.n_prompt, 1, 1,# todo:add batch size support
                        sample_type='ddim', sampler=ddim_sampler,
                        ddim_steps=args.steps, eta=args.eta, 
                        cfg_scale=args.cfg_scale,
                        decode_frame_bs=1,
                        ddp=False, show_denoising_progress=False,
                        )
        # save
        if batch > 0:
            outdir_current = os.path.join(get_outdir(), f"{init_timestring}_{batch}")
        print(f'text2video finished, saving frames to {outdir_current}')

        # just deleted the folder so we need to make it again
        # os.makedirs(outdir_current, exist_ok=True)
        # for i in range(len(samples)):
        #     cv2.imwrite(outdir_current + os.path.sep +
        #                 f"{i:06}.png", samples[i])

        # # TODO: add params to the GUI
        # if not skip_video_creation:
        #     ffmpeg_stitch_video(ffmpeg_location=ffmpeg_location, fps=fps, outmp4_path=outdir_current + os.path.sep + f"vid.mp4", imgs_path=os.path.join(outdir_current,
        #                         "%06d.png"), stitch_from_frame=0, stitch_to_frame=-1, add_soundtrack=add_soundtrack, audio_path=img2img_frames_path if add_soundtrack == 'Init Video' else soundtrack_path, crf=ffmpeg_crf, preset=ffmpeg_preset)

        npz_to_video_grid(samples[0:1,...],  # TODO: is this the reason only 1 second is saved?
                              os.path.join(outdir_current, f"vid.mp4"), 
                              fps=video_args.fps)
        if add_soundtrack != 'None':
            add_soundtrack(video_args.ffmpeg_location, video_args.fps, os.path.join(outdir_current, f"vid.mp4"), 0, -1, None, add_soundtrack, video_args.soundtrack_path, video_args.ffmpeg_crf, video_args.ffmpeg_preset)
        print(f't2v complete, result saved at {outdir_current}')

        mp4 = open(outdir_current + os.path.sep + f"vid.mp4", 'rb').read()
        dataurl = "data:video/mp4;base64," + b64encode(mp4).decode()
        t2v_helpers_args.i1_store_t2v = f'<p style=\"font-weight:bold;margin-bottom:0em\">text2video extension for auto1111 — version 1.1b </p><video controls loop><source src="{dataurl}" type="video/mp4"></video>'
        print("Finish sampling!")
        print(f"Run time = {(time.time() - start):.2f} seconds")
    pbar.close()
    # TODO: rework VideoCrafter
    return [dataurl]
    # if opt.ddp:
    #     dist.destroy_process_group()
