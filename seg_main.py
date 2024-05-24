import gc
import os
import warnings
import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation

from aot_tracker import _palette
from video_impating_main import do_video_impating

warnings.filterwarnings("ignore")

aot_model2ckpt = {
    "deaotb": "./weights/DeAOTB_PRE_YTB_DAV.pth",
    "r50_deaotl": "./weights/R50_DeAOTL_PRE_YTB_DAV.pth",
    "swinb_deaotl": "./weights/SwinB_DeAOTL_PRE_YTB_DAV.pth",
}

def save_prediction(pred_mask, output_dir, file_name):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette([0, 0, 0, 255, 255, 255])
    save_mask.save(os.path.join(output_dir, file_name))

def colorize_mask(pred_mask):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask = save_mask.convert(mode='RGB')
    return np.array(save_mask)

def draw_mask(img, mask, alpha=0.5, id_countour=False):
    img_mask = img
    if id_countour:
        # very slow ~ 1s per image
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids != 0]

        for id in obj_ids:
            # Overlay color on  binary mask
            if id <= 255:
                color = _palette[id * 3:id * 3 + 3]
            else:
                color = [0, 0, 0]
            foreground = img * (1 - alpha) + np.ones_like(img) * alpha * np.array(color)
            binary_mask = (mask == id)

            # Compose image
            img_mask[binary_mask] = foreground[binary_mask]
            countours = binary_dilation(binary_mask, iterations=1) ^ binary_mask
            img_mask[countours, :] = 0
    else:
        binary_mask = (mask != 0)
        countours = binary_dilation(binary_mask, iterations=1) ^ binary_mask
        foreground = img * (1 - alpha) + colorize_mask(mask) * alpha
        img_mask[binary_mask] = foreground[binary_mask]
        img_mask[countours, :] = 0

    return img_mask.astype(img.dtype)

def create_dir(dir_path):
    if os.path.isdir(dir_path):
        os.system(f"rmdir /s /q \"{dir_path}\"")

    os.makedirs(dir_path)

def tracking_objects_in_video(segmentation, input_video, mask_dilation):
    if input_video is not None:
        video_name = os.path.basename(input_video).split('.')[0].replace(' ', '_')
    else:
        return None, None

    # create dir to save result
    tracking_result_dir = f'{os.path.join(os.path.dirname(__file__), "results", f"{video_name}")}'
    create_dir(tracking_result_dir)

    io_args = {
        'tracking_result_dir': tracking_result_dir,
        'output_mask_dir': f'{tracking_result_dir}/{video_name}_masks',
        'output_frame_dir': f'{tracking_result_dir}/{video_name}_frames',
        'output_video': f'{tracking_result_dir}/{video_name}_seg.mp4'
    }

    if input_video is not None:
        return video_type_input_tracking(segmentation, input_video, io_args, mask_dilation)

def video_type_input_tracking(segmentation, input_video, io_args, mask_dilation):
    # source video to segment
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # create dir to save predicted mask and masked frame
    create_dir(io_args['output_mask_dir'])
    create_dir(io_args['output_frame_dir'])

    torch.cuda.empty_cache()
    gc.collect()
    frame_idx = 0

    with torch.cuda.amp.autocast():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if frame_idx == 0:
                pred_mask = segmentation.first_frame_mask
            else:
                pred_mask = segmentation.track(frame, update_memory=True)
            '''
            elif (frame_idx % 3) == 0:
                seg_mask = segmentation.seg(frame)

                track_mask = segmentation.track(frame)
                new_obj_mask = segmentation.find_new_objs(track_mask, seg_mask)
                pred_mask = track_mask + new_obj_mask
                segmentation.add_reference(frame, pred_mask)
                '''
            torch.cuda.empty_cache()
            gc.collect()

            save_prediction(pred_mask, io_args['output_mask_dir'], str(frame_idx).zfill(5) + '.png')
            frame_idx += 1

        print("processed frame {}, obj_num {}".format(frame_idx, segmentation.get_obj_num()), end='\r')
        cap.release()

    del segmentation
    torch.cuda.empty_cache()
    gc.collect()

    cap = cv2.VideoCapture(input_video)
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imwrite(f"{io_args['output_frame_dir']}/{str(frame_idx).zfill(5)}.png", frame)
        frame_idx += 1

    cap.release()

    src = do_video_impating(io_args['output_frame_dir'], io_args['output_mask_dir'], mask_dilation)

    return src
