import re
import numpy as np
import numexpr
import pandas as pd

class T2VAnimKeys():
    def __init__(self, anim_args, seed=-1):
        self.fi = FrameInterpolater(anim_args.max_frames, seed)
        self.inpainting_weights_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.inpainting_weights))

def check_is_number(value):
    float_pattern = r'^(?=.)([+-]?([0-9]*)(\.([0-9]+))?)$'
    return re.match(float_pattern, value)

class FrameInterpolater():
    def __init__(self, max_frames=0, seed=-1) -> None:
        self.max_frames = max_frames
        self.seed = seed

    def sanitize_value(self, value):
        return value.replace("'","").replace('"',"").replace('(',"").replace(')',"")

    def get_inbetweens(self, key_frames, integer=False, interp_method='Linear', is_single_string = False):
        key_frame_series = pd.Series([np.nan for a in range(self.max_frames)])
        # get our ui variables set for numexpr.evaluate
        max_f = self.max_frames -1
        s = self.seed
        for i in range(0, self.max_frames):
            if i in key_frames:
                value = key_frames[i]
                value_is_number = check_is_number(self.sanitize_value(value))
                if value_is_number: # if it's only a number, leave the rest for the default interpolation
                    key_frame_series[i] = self.sanitize_value(value)
            if not value_is_number:
                t = i
                # workaround for values formatted like 0:("I am test") //used for sampler schedules
                key_frame_series[i] = numexpr.evaluate(value) if not is_single_string else self.sanitize_value(value)
            elif is_single_string:# take previous string value and replicate it
                key_frame_series[i] = key_frame_series[i-1]
        key_frame_series = key_frame_series.astype(float) if not is_single_string else key_frame_series # as string
        
        if interp_method == 'Cubic' and len(key_frames.items()) <= 3:
            interp_method = 'Quadratic'    
        if interp_method == 'Quadratic' and len(key_frames.items()) <= 2:
            interp_method = 'Linear'
            
        key_frame_series[0] = key_frame_series[key_frame_series.first_valid_index()]
        key_frame_series[self.max_frames-1] = key_frame_series[key_frame_series.last_valid_index()]
        key_frame_series = key_frame_series.interpolate(method=interp_method.lower(), limit_direction='both')
        if integer:
            return key_frame_series.astype(int)
        return key_frame_series

    def parse_key_frames(self, string):
        # because math functions (i.e. sin(t)) can utilize brackets 
        # it extracts the value in form of some stuff
        # which has previously been enclosed with brackets and
        # with a comma or end of line existing after the closing one
        frames = dict()
        for match_object in string.split(","):
            frameParam = match_object.split(":")
            max_f = self.max_frames -1
            s = self.seed
            frame = int(self.sanitize_value(frameParam[0])) if check_is_number(self.sanitize_value(frameParam[0].strip())) else int(numexpr.evaluate(frameParam[0].strip().replace("'","",1).replace('"',"",1)[::-1].replace("'","",1).replace('"',"",1)[::-1]))
            frames[frame] = frameParam[1].strip()
        if frames == {} and len(string) != 0:
            raise RuntimeError('Key Frame string not correctly formatted')
        return frames
