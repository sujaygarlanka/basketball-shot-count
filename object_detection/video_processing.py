import glob
import os
import cv2
import ffmpeg

def convert_videos_to_images(dirpath_for_videos, dirpath_for_images, approx_max_images=100, num_frames=4):
    num_images = 0
    for video_path in glob.glob(f'{dirpath_for_videos}/*.MOV'):
        vidcap = cv2.VideoCapture(video_path)
        frames = []
        # check if video requires rotation
        rotateCode = check_rotation(video_path)
        # extract frames
        while True:
            success, image = vidcap.read()
            if not success:
                break
            if rotateCode is not None:
                image = cv2.rotate(image, rotateCode) 
            frames.append(image)
        # downsample if desired and necessary
        if num_frames < len(frames):
            skip = len(frames) // num_frames
            for i in range(0, len(frames), skip):
                filename = os.path.basename(video_path)
                filename = os.path.splitext(filename)[0]
                frame_filename = filename + '_' + str(i) + '.jpg'
                frame_file_path = os.path.join(dirpath_for_images, frame_filename)
                cv2.imwrite(frame_file_path, frames[i])
                num_images += 1
                print(num_images)

        if (num_images >= approx_max_images): 
            return
    return

def check_rotation(path_video_file):
    # this returns meta-data of the video file in form of a dictionary
    meta_dict = ffmpeg.probe(path_video_file)

    # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
    # we are looking for
    rotateCode = None
    try:
        if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
            rotateCode = cv2.ROTATE_90_CLOCKWISE
        elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
            rotateCode = cv2.ROTATE_180
        elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
            rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE
    except KeyError:
        pass
    return rotateCode