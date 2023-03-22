import time
import subprocess
import os
from pkg_resources import resource_filename

def find_ffmpeg_binary():
    try:
        import google.colab
        return 'ffmpeg'
    except:
        pass
    for package in ['imageio_ffmpeg', 'imageio-ffmpeg']:
        try:
            package_path = resource_filename(package, 'binaries')
            files = [os.path.join(package_path, f) for f in os.listdir(
                package_path) if f.startswith("ffmpeg-")]
            files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            return files[0] if files else 'ffmpeg'
        except:
            return 'ffmpeg'
            
# Stitch images to a h264 mp4 video using ffmpeg
def ffmpeg_stitch_video(ffmpeg_location=None, fps=None, outmp4_path=None, stitch_from_frame=0, stitch_to_frame=None, imgs_path=None, add_soundtrack=None, audio_path=None, crf=17, preset='veryslow'):
    start_time = time.time()

    print(f"Got a request to stitch frames to video using FFmpeg.\nFrames:\n{imgs_path}\nTo Video:\n{outmp4_path}")
    msg_to_print = f"Stitching *video*..."
    print(msg_to_print)
    if stitch_to_frame == -1:
        stitch_to_frame = 999999999
    try:
        cmd = [
            ffmpeg_location,
            '-y',
            '-vcodec', 'png',
            '-r', str(float(fps)),
            '-start_number', str(stitch_from_frame),
            '-i', imgs_path,
            '-frames:v', str(stitch_to_frame),
            '-c:v', 'libx264',
            '-vf',
            f'fps={float(fps)}',
            '-pix_fmt', 'yuv420p',
            '-crf', str(crf),
            '-preset', preset,
            '-pattern_type', 'sequence',
            outmp4_path
        ]
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
    except FileNotFoundError:
        print("\r" + " " * len(msg_to_print), end="", flush=True)
        print(f"\r{msg_to_print}", flush=True)
        raise FileNotFoundError(
            "FFmpeg not found. Please make sure you have a working ffmpeg path under 'ffmpeg_location' parameter.")
    except Exception as e:
        print("\r" + " " * len(msg_to_print), end="", flush=True)
        print(f"\r{msg_to_print}", flush=True)
        raise Exception(
            f'Error stitching frames to video. Actual runtime error:{e}')

    if add_soundtrack != 'None':
        audio_add_start_time = time.time()
        try:
            cmd = [
                ffmpeg_location,
                '-i',
                outmp4_path,
                '-i',
                audio_path,
                '-map', '0:v',
                '-map', '1:a',
                '-c:v', 'copy',
                '-shortest',
                outmp4_path+'.temp.mp4'
            ]
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                print("\r" + " " * len(msg_to_print), end="", flush=True)
                print(f"\r{msg_to_print}", flush=True)
                raise RuntimeError(stderr)
            os.replace(outmp4_path+'.temp.mp4', outmp4_path)
            print("\r" + " " * len(msg_to_print), end="", flush=True)
            print(f"\r{msg_to_print}", flush=True)
            print(f"\rFFmpeg Video+Audio stitching \033[0;32mdone\033[0m in {time.time() - start_time:.2f} seconds!", flush=True)
        except Exception as e:
            print("\r" + " " * len(msg_to_print), end="", flush=True)
            print(f"\r{msg_to_print}", flush=True)
            print(f'\rError adding audio to video. Actual error: {e}', flush=True)
            print(f"FFMPEG Video (sorry, no audio) stitching \033[33mdone\033[0m in {time.time() - start_time:.2f} seconds!", flush=True)
    else:
        print("\r" + " " * len(msg_to_print), end="", flush=True)
        print(f"\r{msg_to_print}", flush=True)
        print(f"\rVideo stitching \033[0;32mdone\033[0m in {time.time() - start_time:.2f} seconds!", flush=True)