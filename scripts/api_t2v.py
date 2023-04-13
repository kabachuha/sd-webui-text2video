import base64
import hashlib
import io
import json
import logging
import os
import sys
import tempfile
from PIL import Image
import urllib
import traceback
from types import SimpleNamespace

import text2vid
from text2vid import T2VArgs_sanity_check, get_t2v_version
from fastapi import FastAPI, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from scripts.video_audio_utils import find_ffmpeg_binary

logger = logging.getLogger(__name__)

current_directory = os.path.dirname(os.path.abspath(__file__))
if current_directory not in sys.path:
    sys.path.append(current_directory)

def t2v_api(_, app: FastAPI):
    logger.debug(f"text2video extension for auto1111 webui")
    logger.debug(f"Git commit: {get_t2v_version()}")
    logger.debug("Loading text2video API endpoints")

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
        )
    
    @app.get("/t2v/api_version")
    async def t2v_api_version():
        return JSONResponse(content={"version": '0.1b'})
    
    @app.get("/t2v/version")
    async def t2v_version():
        return JSONResponse(content={"version": get_t2v_version()})

    @app.get("/t2v/run")
    async def t2v_run(prompt: str, n_prompt: str | None = None, steps:int | None = None, frames:int | None = None, seed:int | None = None, \
                      cfg_scale:int | None = None, width:int | None = None, height:int | None = None, eta:float | None = None, batch_count:int | None = None, \
                      do_img2img:bool = False, vid2vid_input:str = "",strength:float | None = None,img2img_startFrame:int | None = None, \
                      inpainting_image:str = "", inpainting_frames:int | None = None, inpainting_weights:str | None = None,):
        args_dict = locals()
        default_args_dict = text2vid.T2VArgs()
        for k, v in args_dict.items():
            if v is None and k in default_args_dict:
                v = default_args_dict[k]

        """
        Run t2v over api
        @return:
        """
        d = SimpleNamespace(**args_dict)
        dv = SimpleNamespace(**text2vid.T2VOutputArgs())

        tmp_inpainting = None

        # Wrap the process call in a try-except block to handle potential errors
        try:
            T2VArgs_sanity_check(d)

            if inpainting_frames > 0 and len(inpainting_image) > 0:
                img = Image.open(io.BytesIO(urllib.request.urlopen(inpainting_image).file.read()))
                tmp_inpainting = tempfile.NamedTemporaryFile()
                img.save(tmp_inpainting)
            
            if do_img2img and len(vid2vid_input) > 0:
                vid2vid_input = vid2vid_input[len("data:video/mp4;base64,"):] if vid2vid_input.startswith("data:video/mp4;base64,") else vid2vid_input
                tmp_vid2vid = tempfile.NamedTemporaryFile()
                tmp_vid2vid.write(io.BytesIO(base64.b64decode(vid2vid_input)).getbuffer())

            videodat = text2vid.run(
                # ffmpeg params
                dv.skip_video_creation, #skip_video_creation
                find_ffmpeg_binary(), #ffmpeg_location
                dv.ffmpeg_crf, #ffmpeg_crf
                dv.ffmpeg_preset,#ffmpeg_preset
                dv.fps,#fps
                dv.add_soundtrack,#add_soundtrack
                dv.soundtrack_path,#soundtrack_paths
                d.prompt,#prompt
                d.n_prompt,#n_prompt
                d.steps,#steps
                d.frames,#frames
                d.seed,#seed
                d.cfg_scale,#cfg_scale
                d.width,#width
                d.height,#height
                d.eta,#eta
                # The same, but for vid2vid. Will deduplicate later
                d.prompt,#prompt
                d.n_prompt,#n_prompt
                d.steps,#steps
                d.frames,#frames
                d.seed,#seed
                d.cfg_scale,#cfg_scale
                d.width,#width
                d.height,#height
                d.eta,#eta
                batch_count_v=d.batch_count,#batch_count_v
                batch_count=d.batch_count,#batch_count
                do_img2img=do_img2img,#do_img2img
                img2img_frames=tmp_vid2vid,#img2img_frames
                img2img_frames_path="",#img2img_frames_path
                strength=d.strength,#strength
                img2img_startFrame=d.img2img_startFrame,#img2img_startFrame
                inpainting_image=tmp_inpainting,
                inpainting_frames=d.inpainting_frames,
                inpainting_weights=d.inpainting_weights,#inpainting_weights
                model_type="ModelScope",#Only one has stable support at this moment
            )

            return JSONResponse(content={"mp4s": videodat})
        except Exception as e:
            # Log the error and return a JSON response with an appropriate status code and error message
            logger.error(f"Error processing the video: {e}")
            traceback.print_exc()
            return JSONResponse(
                status_code=500,
                content={"detail": "An error occurred while processing the video."},
            )
        finally:
            if tmp_vid2vid is not None:
                tmp_vid2vid.close()
            if tmp_inpainting is not None:
                tmp_inpainting.close()


try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(t2v_api)
    logger.debug("SD-Webui text2video API layer loaded")
except ImportError:
    logger.debug("Unable to import script callbacks.XXX")
