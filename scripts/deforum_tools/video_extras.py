import os
import gradio as gr
from gradio_funcs import *
import tempfile
from .general_utils import get_os, get_deforum_version, custom_placeholder_format, test_long_path_support, get_max_path_length, substitute_placeholders
from .upscaling import process_ncnn_upscale_vid_upload_logic
from .frame_interpolation import set_interp_out_fps, gradio_f_interp_get_fps_and_fcount, process_interp_vid_upload_logic, process_interp_pics_upload_logic
from .face_restore import process_face_restore_vid_upload_logic

def setup_extras_ui(fps, add_soundtrack, soundtrack_path, skip_video_creation, ffmpeg_crf, ffmpeg_preset, ffmpeg_location):
    dv = DeforumExtrasArgs()
    with gr.TabItem('Upscaling'):
        vid_to_upscale_chosen_file = gr.File(label="Video to Upscale", interactive=True, file_count="single", file_types=["video"], elem_id="vid_to_upscale_chosen_file")
        with gr.Column():
            # NCNN UPSCALE TAB
            with gr.Row(variant='compact') as ncnn_upload_vid_stats_row:
                ncnn_upscale_in_vid_frame_count_window = gr.Textbox(label="In Frame Count", lines=1, interactive=False, value='---') # Non interactive textbox showing uploaded input vid Frame Count
                ncnn_upscale_in_vid_fps_ui_window = gr.Textbox(label="In FPS", lines=1, interactive=False, value='---') # Non interactive textbox showing uploaded input vid FPS
                ncnn_upscale_in_vid_res = gr.Textbox(label="In Res", lines=1, interactive=False, value='---') # Non interactive textbox showing uploaded input resolution
                ncnn_upscale_out_vid_res = gr.Textbox(label="Out Res", value='---') # Non interactive textbox showing expected output resolution
            with gr.Column():
                with gr.Row(variant='compact', visible=True) as ncnn_actual_upscale_row:
                    ncnn_upscale_model = gr.Dropdown(label="Upscale model", choices=['realesr-animevideov3', 'realesrgan-x4plus', 'realesrgan-x4plus-anime'], interactive=True, value = "realesr-animevideov3", type="value")
                    ncnn_upscale_factor =  gr.Dropdown(choices=['x2', 'x3', 'x4'], label="Upscale factor", interactive=True, value="x2", type="value")
                    ncnn_upscale_keep_imgs = gr.Checkbox(label="Keep Imgs", value=True, interactive=True) # fix value
            ncnn_upscale_btn = gr.Button(value="*Upscale uploaded video*")
            ncnn_upscale_btn.click(ncnn_upload_vid_to_upscale,inputs=[vid_to_upscale_chosen_file, ncnn_upscale_in_vid_fps_ui_window, ncnn_upscale_in_vid_res, ncnn_upscale_out_vid_res, ncnn_upscale_model, ncnn_upscale_factor, ncnn_upscale_keep_imgs, ffmpeg_location, ffmpeg_crf, ffmpeg_preset])
    # FRAME INTERPOLATION TAB
    with gr.TabItem('Interpolation') as frame_interp_tab:
        with gr.Accordion('Important notes and Help', open=False):
            gr.HTML("""
            Use <a href="https://github.com/megvii-research/ECCV2022-RIFE">RIFE</a> / <a href="https://film-net.github.io/">FILM</a> Frame Interpolation to smooth out, slow-mo (or both) any video.</p>
            <p style="margin-top:1em">
                Supported engines:
                <ul style="list-style-type:circle; margin-left:1em; margin-bottom:1em">
                    <li>RIFE v4.6 and FILM.</li>
                </ul>
            </p>
            <p style="margin-top:1em">
                Important notes:
                <ul style="list-style-type:circle; margin-left:1em; margin-bottom:1em">
                    <li>Frame Interpolation will *not* run if any of the following are enabled: 'Store frames in ram' / 'Skip video for run all'.</li>
                    <li>Audio (if provided) will *not* be transferred to the interpolated video if Slow-Mo is enabled.</li>
                    <li>'add_soundtrack' and 'soundtrack_path' aren't being honoured in "Interpolate an existing video" mode. Original vid audio will be used instead with the same slow-mo rules above.</li>
                    <li>In "Interpolate existing pics" mode, FPS is determined *only* by output FPS slider. Audio will be added if requested even with slow-mo "enabled", as it does *nothing* in this mode.</li>
                </ul>
            </p>
            """)
        with gr.Column(variant='compact'):
            with gr.Row(variant='compact'):
                # Interpolation Engine
                frame_interpolation_engine = gr.Dropdown(label="Engine", choices=['None','RIFE v4.6','FILM'], value=dv.frame_interpolation_engine, type="value", elem_id="frame_interpolation_engine", interactive=True)
                frame_interpolation_slow_mo_enabled = gr.Checkbox(label="Slow Mo", elem_id="frame_interpolation_slow_mo_enabled", value=dv.frame_interpolation_slow_mo_enabled, interactive=True, visible=False)
                # If this is set to True, we keep all of the interpolated frames in a folder. Default is False - means we delete them at the end of the run
                frame_interpolation_keep_imgs = gr.Checkbox(label="Keep Imgs", elem_id="frame_interpolation_keep_imgs", value=dv.frame_interpolation_keep_imgs, interactive=True, visible=False)
            with gr.Row(variant='compact', visible=False) as frame_interp_amounts_row:
                with gr.Column(min_width=180) as frame_interp_x_amount_column:
                    # How many times to interpolate (interp X)
                    frame_interpolation_x_amount = gr.Slider(minimum=2, maximum=10, step=1, label="Interp X", value=dv.frame_interpolation_x_amount, interactive=True)
                with gr.Column(min_width=180, visible=False) as frame_interp_slow_mo_amount_column:
                    # Interp Slow-Mo (setting final output fps, not really doing anything direclty with RIFE/FILM)
                    frame_interpolation_slow_mo_amount =  gr.Slider(minimum=2, maximum=10, step=1, label="Slow-Mo X", value=dv.frame_interpolation_x_amount, interactive=True)
            with gr.Row(visible=False) as interp_existing_video_row:
                # Intrpolate any existing video from the connected PC
                with gr.Accordion('Interpolate existing Video/ Images', open=False) as interp_existing_video_accord:
                    with gr.Row(variant='compact') as interpolate_upload_files_row:
                        # A drag-n-drop UI box to which the user uploads a *single* (at this stage) video
                        vid_to_interpolate_chosen_file = gr.File(label="Video to Interpolate", interactive=True, file_count="single", file_types=["video"], elem_id="vid_to_interpolate_chosen_file")
                        # A drag-n-drop UI box to which the user uploads a pictures to interpolate
                        pics_to_interpolate_chosen_file = gr.File(label="Pics to Interpolate", interactive=True, file_count="multiple", file_types=["image"], elem_id="pics_to_interpolate_chosen_file")
                    with gr.Row(variant='compact', visible=False) as interp_live_stats_row:
                        # Non interactive textbox showing uploaded input vid total Frame Count
                        in_vid_frame_count_window = gr.Textbox(label="In Frame Count", lines=1, interactive=False, value='---')
                        # Non interactive textbox showing uploaded input vid FPS
                        in_vid_fps_ui_window = gr.Textbox(label="In FPS", lines=1, interactive=False, value='---')
                        # Non interactive textbox showing expected output interpolated video FPS
                        out_interp_vid_estimated_fps = gr.Textbox(label="Interpolated Vid FPS", value='---')
                    with gr.Row(variant='compact') as interp_buttons_row:
                        # This is the actual button that's pressed to initiate the interpolation:
                        interpolate_button = gr.Button(value="*Interpolate Video*")
                        interpolate_pics_button = gr.Button(value="*Interpolate Pics*")
                    # Show a text about CLI outputs:
                    gr.HTML("* check your CLI for outputs *", elem_id="below_interpolate_butts_msg") # TODO: CSS THIS TO CENTER OF ROW!
                    # make the functin call when the interpolation button is clicked
                    interpolate_button.click(upload_vid_to_interpolate,inputs=[vid_to_interpolate_chosen_file, frame_interpolation_engine, frame_interpolation_x_amount, frame_interpolation_slow_mo_enabled, frame_interpolation_slow_mo_amount, frame_interpolation_keep_imgs, ffmpeg_location, ffmpeg_crf, ffmpeg_preset, in_vid_fps_ui_window])
                    interpolate_pics_button.click(upload_pics_to_interpolate,inputs=[pics_to_interpolate_chosen_file, frame_interpolation_engine, frame_interpolation_x_amount, frame_interpolation_slow_mo_enabled, frame_interpolation_slow_mo_amount, frame_interpolation_keep_imgs, ffmpeg_location, ffmpeg_crf, ffmpeg_preset, fps, add_soundtrack, soundtrack_path])
    with gr.TabItem('Face restore'):
        vid_to_face_restore_chosen_file = gr.File(label="Video to Face restore", interactive=True, file_count="single", file_types=["video"], elem_id="vid_to_face_restore_chosen_file")
        with gr.Column():
            with gr.Column():
                with gr.Row(variant='compact', visible=True):
                    gr.HTML("Face restoration settings are in the webui's core 'Settings' tab")
                with gr.Row(variant='compact', visible=True):
                    face_restore_keep_imgs = gr.Checkbox(label="Keep Imgs", value=True, interactive=True) # fix value
            face_restore_btn = gr.Button(value="*Face restore uploaded video*")
            face_restore_btn.click(upload_vid_to_face_restore,inputs=[vid_to_face_restore_chosen_file, face_restore_keep_imgs, ffmpeg_location, ffmpeg_crf, ffmpeg_preset])

    ncnn_upscale_model.change(fn=update_r_upscale_factor, inputs=ncnn_upscale_model, outputs=ncnn_upscale_factor)
    ncnn_upscale_model.change(update_upscale_out_res_by_model_name, inputs=[ncnn_upscale_in_vid_res, ncnn_upscale_model], outputs=ncnn_upscale_out_vid_res)
    ncnn_upscale_factor.change(update_upscale_out_res, inputs=[ncnn_upscale_in_vid_res, ncnn_upscale_factor], outputs=ncnn_upscale_out_vid_res)
    vid_to_upscale_chosen_file.change(vid_upscale_gradio_update_stats,inputs=[vid_to_upscale_chosen_file, ncnn_upscale_factor],outputs=[ncnn_upscale_in_vid_fps_ui_window, ncnn_upscale_in_vid_frame_count_window, ncnn_upscale_in_vid_res, ncnn_upscale_out_vid_res])
    frame_interpolation_slow_mo_enabled.change(fn=hide_slow_mo,inputs=frame_interpolation_slow_mo_enabled,outputs=frame_interp_slow_mo_amount_column)
    frame_interpolation_engine.change(fn=change_interp_x_max_limit,inputs=[frame_interpolation_engine,frame_interpolation_x_amount],outputs=frame_interpolation_x_amount)
    [change_fn.change(set_interp_out_fps, inputs=[frame_interpolation_x_amount, frame_interpolation_slow_mo_enabled, frame_interpolation_slow_mo_amount, in_vid_fps_ui_window], outputs=out_interp_vid_estimated_fps) for change_fn in [frame_interpolation_x_amount, frame_interpolation_slow_mo_amount, frame_interpolation_slow_mo_enabled]]
    # Populate the FPS and FCount values as soon as a video is uploaded to the FileUploadBox (vid_to_interpolate_chosen_file)
    vid_to_interpolate_chosen_file.change(gradio_f_interp_get_fps_and_fcount,inputs=[vid_to_interpolate_chosen_file, frame_interpolation_x_amount, frame_interpolation_slow_mo_enabled, frame_interpolation_slow_mo_amount],outputs=[in_vid_fps_ui_window,in_vid_frame_count_window, out_interp_vid_estimated_fps])
    vid_to_interpolate_chosen_file.change(fn=hide_interp_stats,inputs=[vid_to_interpolate_chosen_file],outputs=[interp_live_stats_row])
    interp_hide_list = [frame_interpolation_slow_mo_enabled,frame_interpolation_keep_imgs,frame_interp_amounts_row,interp_existing_video_row]
    for output in interp_hide_list:
        frame_interpolation_engine.change(fn=hide_interp_by_interp_status,inputs=frame_interpolation_engine,outputs=output)



# Local gradio-to-frame-interoplation function. *Needs* to stay here since we do Root() and use gradio elements directly, to be changed in the future
def upload_vid_to_interpolate(file, engine, x_am, sl_enabled, sl_am, keep_imgs, f_location, f_crf, f_preset, in_vid_fps):
    # print msg and do nothing if vid not uploaded or interp_x not provided
    if not file or engine == 'None':
        return print("Please upload a video and set a proper value for 'Interp X'. Can't interpolate x0 times :)")

    root_params = ExtrasRoot()
    f_models_path = root_params['models_path']

    process_interp_vid_upload_logic(file, engine, x_am, sl_enabled, sl_am, keep_imgs, f_location, f_crf, f_preset, in_vid_fps, f_models_path, file.orig_name)

def upload_pics_to_interpolate(pic_list, engine, x_am, sl_enabled, sl_am, keep_imgs, f_location, f_crf, f_preset, fps, add_audio, audio_track):
    from PIL import Image
    
    if pic_list is None or len(pic_list) < 2:
        return print("Please upload at least 2 pics for interpolation.")
        
    # make sure all uploaded pics have the same resolution
    pic_sizes = [Image.open(picture_path).size for picture_path in pic_list]
    if len(set(pic_sizes)) != 1:
        return print("All uploaded pics need to be of the same Width and Height / resolution.")
        
    resolution = pic_sizes[0]
     
    root_params = ExtrasRoot()
    f_models_path = root_params['models_path']
    
    process_interp_pics_upload_logic(pic_list, engine, x_am, sl_enabled, sl_am, keep_imgs, f_location, f_crf, f_preset, fps, f_models_path, resolution, add_audio, audio_track)

def ncnn_upload_vid_to_upscale(vid_path, in_vid_fps, in_vid_res, out_vid_res, upscale_model, upscale_factor, keep_imgs, f_location, f_crf, f_preset):
    if vid_path is None:
        print("Please upload a video :)")
        return
    root_params = ExtrasRoot()
    f_models_path = root_params['models_path']
    current_user = root_params['current_user_os']
    process_ncnn_upscale_vid_upload_logic(vid_path, in_vid_fps, in_vid_res, out_vid_res, f_models_path, upscale_model, upscale_factor, keep_imgs, f_location, f_crf, f_preset, current_user)

def upload_vid_to_face_restore(vid_to_face_restore_chosen_file, depth_keep_imgs, ffmpeg_location, ffmpeg_crf, ffmpeg_preset):
    # print msg and do nothing if vid not uploaded
    if not vid_to_face_restore_chosen_file:
        return print("Please upload a video :(")
    
    process_face_restore_vid_upload_logic(vid_to_face_restore_chosen_file, vid_to_face_restore_chosen_file.orig_name, depth_keep_imgs, ffmpeg_location, ffmpeg_crf, ffmpeg_preset)    

def DeforumExtrasArgs():
    # End-Run upscaling
    r_upscale_video = False
    r_upscale_factor = 'x2'  # ['2x', 'x3', 'x4']
    # 'realesr-animevideov3' (default of realesrgan engine, does 2-4x), the rest do only 4x: 'realesrgan-x4plus', 'realesrgan-x4plus-anime'
    r_upscale_model = 'realesr-animevideov3'
    r_upscale_keep_imgs = True

    render_steps = False
    path_name_modifier = "x0_pred"  # ["x0_pred","x"]
    # **Interpolate Video Settings**
    frame_interpolation_engine = "None"  # ["None", "RIFE v4.6", "FILM"]
    frame_interpolation_x_amount = 2  # [2 to 1000 depends on the engine]
    frame_interpolation_slow_mo_enabled = False
    frame_interpolation_slow_mo_amount = 2  # [2 to 10]
    frame_interpolation_keep_imgs = False
    return locals()

from modules import shared
from modules.shared import cmd_opts
import modules.paths as ph
def ExtrasRoot():
    device = shared.device
    models_path = ph.models_path + '/Deforum' # the extras models are the same as in Deforum
    half_precision = not cmd_opts.no_half
    mask_preset_names = ['everywhere','video_mask']
    p = None
    frames_cache = []
    raw_batch_name = None
    raw_seed = None
    initial_seed = None
    initial_info = None
    first_frame = None
    outpath_samples = ""
    animation_prompts = None
    color_corrections = None 
    initial_clipskip = None
    current_user_os = get_os()
    tmp_deforum_run_duplicated_folder = os.path.join(tempfile.gettempdir(), 'tmp_run_deforum')
    return locals()