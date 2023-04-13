import base64
import hashlib
import io
import json
import logging
import os
import sys
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
    logger.debug("Loading T2V API Endpoints.")

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
    async def t2v_run(prompt: str):

        args_dict = locals()
        default_args_dict = text2vid.DeforumOutputArgs()
        for k, v in args_dict.items():
            if v is None and k in default_args_dict:
                v = default_args_dict[k]

        """
        Run t2v over api
        @return:
        """
        dv = SimpleNamespace(**args_dict)

        # Wrap the process call in a try-except block to handle potential errors
        try:
            T2VArgs_sanity_check(dv)

            videodat = text2vid.process(
                dv.skip_video_creation,
                find_ffmpeg_binary(),
                dv.ffmpeg_crf,
                "slow",
                dv.fps,
                dv.add_soundtrack,
                dv.soundtrack_path,
                prompt,
                "text, watermark, copyright, blurry",
                30,
                24,
                -1,
                7,
                256,
                256,
                0,
                "",
                "",
                "",
                "",
                -1,
                "",
                "",
                "",
                "",
                "",
            )

            return JSONResponse(content={"mp4": videodat})
        except Exception as e:
            # Log the error and return a JSON response with an appropriate status code and error message
            logger.error(f"Error processing the video: {e}")
            traceback.print_exc()
            return JSONResponse(
                status_code=500,
                content={"detail": "An error occurred while processing the video."},
            )


try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(t2v_api)
    logger.debug("SD-Webui API layer loaded XXX")
except ImportError:
    logger.debug("Unable to import script callbacks.XXX")
