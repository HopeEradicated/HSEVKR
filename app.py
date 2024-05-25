import gc
import os

import cv2
import gradio as gr
import numpy as np
import torch

from seg_utils import segmentation
from model_args import segmentation_args, sam_args, aot_args
from seg_main import aot_model2ckpt, tracking_objects_in_video, draw_mask
from tool.transfer_tools import mask2bbox


def clean():
    return None, None, None, None, None, None, [[], []]


def get_click_prompt(click_stack, point):
    click_stack[0].append(point["coord"])
    click_stack[1].append(point["mode"]
                          )

    prompt = {
        "points_coord": click_stack[0],
        "points_mode": click_stack[1],
        "multimask": "false",
    }

    return prompt


def get_meta_from_video(input_video):
    if input_video is None:
        return None, None, None, ""

    cap = cv2.VideoCapture(input_video)

    _, first_frame = cap.read()
    cap.release()

    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

    return first_frame, first_frame, first_frame, ""


def segmentation_add_first_frame(Seg_Tracker, origin_frame, predicted_mask):
    with torch.cuda.amp.autocast():
        frame_idx = 0
        Seg_Tracker.restart_tracker()
        Seg_Tracker.add_reference(origin_frame, predicted_mask, frame_idx)
        Seg_Tracker.first_frame_mask = predicted_mask

    return Seg_Tracker


def init_segmentation(aot_model, origin_frame):
    if origin_frame is None:
        return None, origin_frame, [[], []], ""

    # reset aot args
    aot_args["model"] = aot_model
    aot_args["model_path"] = aot_model2ckpt[aot_model]

    # reset sam args
    segmentation_args["sam_gap"] = 9999
    segmentation_args["max_obj_num"] = 1
    sam_args["generator_args"]["points_per_side"] = 100

    Seg_Tracker = segmentation(segmentation_args, sam_args, aot_args)
    Seg_Tracker.restart_tracker()

    return Seg_Tracker, origin_frame, [[], []], ""


def init_segmentation_Stroke(aot_model, origin_frame):
    if origin_frame is None:
        return None, origin_frame, [[], []], origin_frame

    # reset aot args
    aot_args["model"] = aot_model
    aot_args["model_path"] = aot_model2ckpt[aot_model]

    # reset sam args
    segmentation_args["sam_gap"] = 9999
    segmentation_args["max_obj_num"] = 1
    sam_args["generator_args"]["points_per_side"] = 100

    Seg_Tracker = segmentation(segmentation_args, sam_args, aot_args)
    Seg_Tracker.restart_tracker()
    return Seg_Tracker, origin_frame, [[], []], origin_frame


def undo_click_stack_and_refine_seg(Seg_Tracker, origin_frame, click_stack):
    if Seg_Tracker is None:
        return Seg_Tracker, origin_frame, [[], []]

    if len(click_stack[0]) > 0:
        click_stack[0] = click_stack[0][: -1]
        click_stack[1] = click_stack[1][: -1]

    if len(click_stack[0]) > 0:
        prompt = {
            "points_coord": click_stack[0],
            "points_mode": click_stack[1],
            "multimask": "false",
        }

        masked_frame = seg_acc_click(Seg_Tracker, prompt, origin_frame)
        return Seg_Tracker, masked_frame, click_stack
    else:
        return Seg_Tracker, origin_frame, [[], []]


def seg_acc_click(Seg_Tracker, prompt, origin_frame):
    predicted_mask, masked_frame = Seg_Tracker.seg_acc_click(
        origin_frame=origin_frame,
        coords=np.array(prompt["points_coord"]),
        modes=np.array(prompt["points_mode"]),
        multimask=prompt["multimask"],
    )
    segmentation_add_first_frame(Seg_Tracker, origin_frame, predicted_mask)
    return masked_frame


def sam_click(Seg_Tracker, origin_frame, point_mode, click_stack, aot_model,
              evt: gr.SelectData):
    if point_mode == "Positive":
        point = {"coord": [evt.index[0], evt.index[1]], "mode": 1}
    else:
        point = {"coord": [evt.index[0], evt.index[1]], "mode": 0}

    if Seg_Tracker is None:
        Seg_Tracker, _, _, _ = init_segmentation(aot_model, origin_frame)

    click_prompt = get_click_prompt(click_stack, point)

    masked_frame = seg_acc_click(Seg_Tracker, click_prompt, origin_frame)

    return Seg_Tracker, masked_frame, click_stack


def sam_stroke(Seg_Tracker, origin_frame, drawing_board, aot_model):
    if Seg_Tracker is None:
        Seg_Tracker, _, _, _ = init_segmentation(aot_model, origin_frame)

    mask = drawing_board["mask"]
    bbox = mask2bbox(mask[:, :, 0])  # bbox: [[x0, y0], [x1, y1]]
    predicted_mask, masked_frame = Seg_Tracker.seg_acc_bbox(origin_frame, bbox)

    Seg_Tracker = segmentation_add_first_frame(Seg_Tracker, origin_frame, predicted_mask)

    return Seg_Tracker, masked_frame, origin_frame


def gd_detect(Seg_Tracker, origin_frame, grounding_caption, box_threshold, text_threshold, aot_model):
    if Seg_Tracker is None:
        Seg_Tracker, _, _, _ = init_segmentation(aot_model, origin_frame)

    predicted_mask, annotated_frame = Seg_Tracker.detect_and_seg(origin_frame, grounding_caption, box_threshold,
                                                                 text_threshold)

    Seg_Tracker = segmentation_add_first_frame(Seg_Tracker, origin_frame, predicted_mask)

    masked_frame = draw_mask(annotated_frame, predicted_mask)

    return Seg_Tracker, masked_frame, origin_frame


def segment_everything(Seg_Tracker, aot_model, origin_frame):
    if Seg_Tracker is None:
        Seg_Tracker, _, _, _ = init_segmentation(aot_model, origin_frame)

    frame_idx = 0

    with torch.cuda.amp.autocast():
        pred_mask = Seg_Tracker.seg(origin_frame)
        torch.cuda.empty_cache()
        gc.collect()
        Seg_Tracker.add_reference(origin_frame, pred_mask, frame_idx)
        Seg_Tracker.first_frame_mask = pred_mask

    masked_frame = draw_mask(origin_frame.copy(), pred_mask)

    return Seg_Tracker, masked_frame


def tracking_objects(Seg_Tracker, input_video, mask_dilation):
    print(f"Start tracking {os.path.basename(input_video).split('.')[0].replace(' ', '_')}")
    return tracking_objects_in_video(Seg_Tracker, input_video, mask_dilation)


def seg_app():
    ##########################################################
    ######################  Front-end ########################
    ##########################################################
    app = gr.Blocks()

    with app:
        gr.Markdown(
            '''
            <div style="text-align:center;">
                <span style="font-size:3em; font-weight:bold;">Video Inpainting</span>
            </div>
            '''
        )

        click_stack = gr.State([[], []])  # Storage clicks status
        origin_frame = gr.State(None)
        Seg_Tracker = gr.State(None)

        with gr.Row():
            with gr.Column(scale=0.5):
                input_video = gr.Video(label='Input video').style(height=550)

                input_first_frame = gr.Image(label='Selected mask', interactive=True).style(height=550)

                tab_click = gr.Tab(label="Click")
                with tab_click:
                    with gr.Row():
                        point_mode = gr.Radio(
                            choices=["Positive", "Negative"],
                            value="Positive",
                            label="Point Prompt",
                            interactive=True)

                        click_undo_but = gr.Button(
                            value="Undo",
                            interactive=True
                        )

                tab_stroke = gr.Tab(label="Stroke")
                with tab_stroke:
                    drawing_board = gr.Image(label='Drawing Board', tool="sketch", brush_radius=1, interactive=True)
                    with gr.Row():
                        seg_acc_stroke = gr.Button(value="Segment", interactive=True)

                tab_text = gr.Tab(label="Text")
                with tab_text:
                    grounding_caption = gr.Textbox(label="Detection Prompt")
                    detect_button = gr.Button(value="Detect")
                    with gr.Accordion("Advanced options", open=False):
                        with gr.Row():
                            with gr.Column(scale=0.5):
                                box_threshold = gr.Slider(
                                    label="Box Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.001
                                )
                            with gr.Column(scale=0.5):
                                text_threshold = gr.Slider(
                                    label="Text Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.001
                                )

                with gr.Row():
                    with gr.Column(scale=0.5):
                        with gr.Tab(label="Model args"):
                            aot_model = gr.Dropdown(
                                label="Selected model",
                                choices=[
                                    "deaotb",
                                    "r50_deaotl",
                                    "swinb_deaotl"
                                ],
                                value="r50_deaotl",
                                interactive=True,
                            )
                            mask_dilation = gr.Slider(
                                label="Mask dilation",
                                minimum=5,
                                step=1,
                                maximum=10,
                                value=5,
                                interactive=True
                            )
                    with gr.Column():
                        track_for_video = gr.Button(
                            value="Start Tracking",
                            interactive=True,
                        ).style(size="lg")

            with gr.Column(scale=0.5):
                output_video = gr.Video(label='Output video').style(height=550)

        ##########################################################
        ######################  back-end #########################
        ##########################################################

        input_video.change(
            fn=get_meta_from_video,
            inputs=[
                input_video
            ],
            outputs=[
                input_first_frame, origin_frame, drawing_board
            ]
        )

        tab_click.select(
            fn=init_segmentation,
            inputs=[
                aot_model,
                origin_frame
            ],
            outputs=[
                Seg_Tracker, input_first_frame, click_stack  # , grounding_caption
            ],
            queue=False,
        )

        tab_stroke.select(
            fn=init_segmentation_Stroke,
            inputs=[
                aot_model,
                origin_frame,
            ],
            outputs=[
                Seg_Tracker, input_first_frame, click_stack, drawing_board
            ],
            queue=False,
        )

        tab_text.select(
            fn=init_segmentation,
            inputs=[
                aot_model,
                origin_frame
            ],
            outputs=[
                Seg_Tracker, input_first_frame, click_stack, grounding_caption
            ],
            queue=False,
        )

        input_first_frame.select(
            fn=sam_click,
            inputs=[
                Seg_Tracker, origin_frame, point_mode, click_stack,
                aot_model,
            ],
            outputs=[
                Seg_Tracker, input_first_frame, click_stack
            ]
        )

        seg_acc_stroke.click(
            fn=sam_stroke,
            inputs=[
                Seg_Tracker, origin_frame, drawing_board,
                aot_model,
            ],
            outputs=[
                Seg_Tracker, input_first_frame, drawing_board
            ]
        )

        detect_button.click(
            fn=gd_detect,
            inputs=[
                Seg_Tracker, origin_frame, grounding_caption, box_threshold, text_threshold,
                aot_model
            ],
            outputs=[
                Seg_Tracker, input_first_frame
            ]
        )

        track_for_video.click(
            fn=tracking_objects,
            inputs=[
                Seg_Tracker,
                input_video,
                mask_dilation
            ],
            outputs=[
                output_video
            ]
        )

        click_undo_but.click(
            fn=undo_click_stack_and_refine_seg,
            inputs=[
                Seg_Tracker, origin_frame, click_stack
            ],
            outputs=[
                Seg_Tracker, input_first_frame, click_stack
            ]
        )

    app.queue(concurrency_count=1)
    app.launch(debug=True, enable_queue=True, share=True)

if __name__ == "__main__":
    seg_app()
