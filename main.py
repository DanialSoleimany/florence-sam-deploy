import os
import json
from typing import Tuple, Optional, Dict, Any

import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image
from tqdm import tqdm
from utils.video import generate_unique_name, create_directory, delete_directory

from utils.florence import load_florence_model, run_florence_inference, \
    FLORENCE_DETAILED_CAPTION_TASK, \
    FLORENCE_CAPTION_TO_PHRASE_GROUNDING_TASK, FLORENCE_OPEN_VOCABULARY_DETECTION_TASK
from utils.modes import IMAGE_INFERENCE_MODES, IMAGE_OPEN_VOCABULARY_DETECTION_MODE, \
    IMAGE_CAPTION_GROUNDING_MASKS_MODE, VIDEO_INFERENCE_MODES
from utils.sam import load_sam_image_model, run_sam_inference, load_sam_video_model

# Configuration
VIDEO_SCALE_FACTOR = 0.5
VIDEO_TARGET_DIRECTORY = "tmp"
create_directory(directory_path=VIDEO_TARGET_DIRECTORY)

DEVICE = torch.device("cuda")
# DEVICE = torch.device("cpu")

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Load models
FLORENCE_MODEL, FLORENCE_PROCESSOR = load_florence_model(device=DEVICE)

# Florence patch to fix SDPA error
import subprocess
try:
    modeling_path = subprocess.check_output([
        "find", os.path.expanduser("~/.cache/huggingface/"),
        "-type", "f", "-name", "modeling_florence2.py"
    ], text=True).strip()
    if os.path.exists(modeling_path):
        with open(modeling_path, "r") as f:
            content = f.read()
        if "_supports_sdpa = False" not in content:
            subprocess.run([
                "sed", "-i",
                "/class Florence2ForConditionalGeneration/a\\        _supports_sdpa = False",
                modeling_path
            ])
            print("Patched _supports_sdpa = False into:", modeling_path)
        else:
            print("Already patched:", modeling_path)
    else:
        print("Florence file not found.")
except Exception as e:
    print("Error during Florence patching:", e)

SAM_IMAGE_MODEL = load_sam_image_model(device=DEVICE)
SAM_VIDEO_MODEL = load_sam_video_model(device=DEVICE)

# Visualization setup
COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700', '#32CD32', '#8A2BE2']
COLOR_PALETTE = sv.ColorPalette.from_hex(COLORS)
BOX_ANNOTATOR = sv.BoxAnnotator(color=COLOR_PALETTE, color_lookup=sv.ColorLookup.INDEX)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLOR_PALETTE,
    color_lookup=sv.ColorLookup.INDEX,
    text_position=sv.Position.CENTER_OF_MASS,
    text_color=sv.Color.from_hex("#000000"),
    border_radius=5
)
MASK_ANNOTATOR = sv.MaskAnnotator(
    color=COLOR_PALETTE,
    color_lookup=sv.ColorLookup.INDEX
)


def detections_to_dict(detections: sv.Detections) -> Dict[str, Any]:
    """Convert detections to dictionary format for JSON serialization."""
    results = []
    
    for i in range(len(detections)):
        detection_data = {
            "bbox": {
                "x1": float(detections.xyxy[i][0]),
                "y1": float(detections.xyxy[i][1]), 
                "x2": float(detections.xyxy[i][2]),
                "y2": float(detections.xyxy[i][3])
            },
            "confidence": float(detections.confidence[i]) if detections.confidence is not None else None,
            "class_id": int(detections.class_id[i]) if detections.class_id is not None else None,
            "class_name": detections.data.get("class_name", [None])[i] if detections.data and "class_name" in detections.data else None
        }
        
        # Add segmentation mask if available
        if detections.mask is not None and i < len(detections.mask):
            mask = detections.mask[i]
            # Convert mask to polygon coordinates
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            polygons = []
            for contour in contours:
                if len(contour) >= 3:  # Valid polygon needs at least 3 points
                    polygon = contour.reshape(-1, 2).tolist()
                    polygons.append(polygon)
            
            detection_data["segmentation"] = {
                "polygons": polygons,
                "mask_shape": mask.shape
            }
        
        results.append(detection_data)
    
    return {
        "detections": results,
        "total_detections": len(detections)
    }


def save_detections_json(detections: sv.Detections, file_path: str, json_path: str):
    """Save detection results to JSON file."""
    # Load existing JSON data or create new
    json_data = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                json_data = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load existing JSON file: {e}")
            json_data = {}
    
    # Add detection results for this file
    json_data[file_path] = detections_to_dict(detections)
    
    # Save updated JSON
    try:
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"Detection results saved to: {json_path}")
    except Exception as e:
        print(f"Error saving JSON file: {e}")


def annotate_image(image, detections):
    """Annotate image with masks, boxes, and labels."""
    output_image = image.copy()
    output_image = MASK_ANNOTATOR.annotate(output_image, detections)
    output_image = BOX_ANNOTATOR.annotate(output_image, detections)
    output_image = LABEL_ANNOTATOR.annotate(output_image, detections)
    return output_image


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def process_image(
    mode: str, image_input: Image.Image, text_input: str = None, 
    file_path: str = None, json_output_path: str = "detections.json"
) -> Tuple[Optional[Image.Image], Optional[str]]:
    """Process an image with Florence2 + SAM2."""
    if not image_input:
        print("Error: No image provided.")
        return None, None

    if mode == IMAGE_OPEN_VOCABULARY_DETECTION_MODE:
        if not text_input:
            print("Error: No text prompt provided for open vocabulary detection.")
            return None, None

        texts = [prompt.strip() for prompt in text_input.split(",")]
        detections_list = []
        for text in texts:
            _, result = run_florence_inference(
                model=FLORENCE_MODEL,
                processor=FLORENCE_PROCESSOR,
                device=DEVICE,
                image=image_input,
                task=FLORENCE_OPEN_VOCABULARY_DETECTION_TASK,
                text=text
            )
            detections = sv.Detections.from_lmm(
                lmm=sv.LMM.FLORENCE_2,
                result=result,
                resolution_wh=image_input.size
            )
            detections = run_sam_inference(SAM_IMAGE_MODEL, image_input, detections)
            detections_list.append(detections)

        detections = sv.Detections.merge(detections_list)
        detections = run_sam_inference(SAM_IMAGE_MODEL, image_input, detections)
        
        # Save detection results to JSON if file_path provided
        if file_path:
            save_detections_json(detections, file_path, json_output_path)
        
        return annotate_image(image_input, detections), None

    if mode == IMAGE_CAPTION_GROUNDING_MASKS_MODE:
        _, result = run_florence_inference(
            model=FLORENCE_MODEL,
            processor=FLORENCE_PROCESSOR,
            device=DEVICE,
            image=image_input,
            task=FLORENCE_DETAILED_CAPTION_TASK
        )
        caption = result[FLORENCE_DETAILED_CAPTION_TASK]
        _, result = run_florence_inference(
            model=FLORENCE_MODEL,
            processor=FLORENCE_PROCESSOR,
            device=DEVICE,
            image=image_input,
            task=FLORENCE_CAPTION_TO_PHRASE_GROUNDING_TASK,
            text=caption
        )
        detections = sv.Detections.from_lmm(
            lmm=sv.LMM.FLORENCE_2,
            result=result,
            resolution_wh=image_input.size
        )
        detections = run_sam_inference(SAM_IMAGE_MODEL, image_input, detections)
        
        # Save detection results to JSON if file_path provided
        if file_path:
            save_detections_json(detections, file_path, json_output_path)
        
        return annotate_image(image_input, detections), caption
    
    print(f"Error: Unknown mode '{mode}'")
    return None, None


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def process_video(video_input: str, text_input: str, json_output_path: str = "video_detections.json") -> Optional[str]:
    """Process a video with Florence2 + SAM2."""
    if not video_input:
        print("Error: No video provided.")
        return None

    if not text_input:
        print("Error: No text prompt provided.")
        return None

    # Get first frame for detection
    frame_generator = sv.get_video_frames_generator(video_input)
    frame = next(frame_generator)
    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Run Florence2 detection on first frame
    texts = [prompt.strip() for prompt in text_input.split(",")]
    detections_list = []
    for text in texts:
        _, result = run_florence_inference(
            model=FLORENCE_MODEL,
            processor=FLORENCE_PROCESSOR,
            device=DEVICE,
            image=frame,
            task=FLORENCE_OPEN_VOCABULARY_DETECTION_TASK,
            text=text
        )
        detections = sv.Detections.from_lmm(
            lmm=sv.LMM.FLORENCE_2,
            result=result,
            resolution_wh=frame.size
        )
        detections = run_sam_inference(SAM_IMAGE_MODEL, frame, detections)
        detections_list.append(detections)

    detections = sv.Detections.merge(detections_list)
    detections = run_sam_inference(SAM_IMAGE_MODEL, frame, detections)

    if len(detections.mask) == 0:
        print(f"No objects of class '{text_input}' found in the first frame of the video. "
              "Trim the video to make the object appear in the first frame or try a different text prompt.")
        return None

    # Save initial frame detections to JSON
    save_detections_json(detections, f"{video_input}_frame_0", json_output_path)

    # Prepare video processing
    name = generate_unique_name()
    frame_directory_path = os.path.join(VIDEO_TARGET_DIRECTORY, name)
    frames_sink = sv.ImageSink(
        target_dir_path=frame_directory_path,
        image_name_pattern="{:05d}.jpeg"
    )

    video_info = sv.VideoInfo.from_video_path(video_input)
    video_info.width = int(video_info.width * VIDEO_SCALE_FACTOR)
    video_info.height = int(video_info.height * VIDEO_SCALE_FACTOR)

    # Split video into frames
    frames_generator = sv.get_video_frames_generator(video_input)
    with frames_sink:
        for frame in tqdm(
                frames_generator,
                total=video_info.total_frames,
                desc="splitting video into frames"
        ):
            frame = sv.scale_image(frame, VIDEO_SCALE_FACTOR)
            frames_sink.save_image(frame)

    # Initialize SAM2 video model
    inference_state = SAM_VIDEO_MODEL.init_state(
        video_path=frame_directory_path,
        device=DEVICE
    )

    # Add masks to SAM2
    for mask_index, mask in enumerate(detections.mask):
        _, object_ids, mask_logits = SAM_VIDEO_MODEL.add_new_mask(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=mask_index,
            mask=mask
        )

    # Process video with mask propagation
    video_path = os.path.join(VIDEO_TARGET_DIRECTORY, f"{name}.mp4")
    frames_generator = sv.get_video_frames_generator(video_input)
    masks_generator = SAM_VIDEO_MODEL.propagate_in_video(inference_state)
    
    frame_detections = {}  # Store detections for each frame
    
    with sv.VideoSink(video_path, video_info=video_info) as sink:
        for frame_idx, (frame, (_, tracker_ids, mask_logits)) in enumerate(zip(frames_generator, masks_generator)):
            frame = sv.scale_image(frame, VIDEO_SCALE_FACTOR)
            masks = (mask_logits > 0.0).cpu().numpy().astype(bool)
            if len(masks.shape) == 4:
                masks = np.squeeze(masks, axis=1)

            frame_detections_obj = sv.Detections(
                xyxy=sv.mask_to_xyxy(masks=masks),
                mask=masks,
                class_id=np.array(tracker_ids)
            )
            
            # Save frame detections every 30 frames (for performance)
            if frame_idx % 30 == 0:
                frame_detections[f"{video_input}_frame_{frame_idx}"] = detections_to_dict(frame_detections_obj)
            
            annotated_frame = frame.copy()
            annotated_frame = MASK_ANNOTATOR.annotate(
                scene=annotated_frame, detections=frame_detections_obj)
            annotated_frame = BOX_ANNOTATOR.annotate(
                scene=annotated_frame, detections=frame_detections_obj)
            sink.write_frame(annotated_frame)

    # Save all frame detections to JSON
    if frame_detections:
        json_data = {}
        if os.path.exists(json_output_path):
            try:
                with open(json_output_path, 'r') as f:
                    json_data = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load existing JSON file: {e}")
        
        json_data.update(frame_detections)
        
        try:
            with open(json_output_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            print(f"Video detection results saved to: {json_output_path}")
        except Exception as e:
            print(f"Error saving video JSON file: {e}")

    # Clean up
    delete_directory(frame_directory_path)
    return video_path


# Example usage functions
def process_image_from_path(image_path: str, mode: str, text_prompt: str = None, json_output_path: str = "detections.json") -> Tuple[Optional[Image.Image], Optional[str]]:
    """Process image from file path."""
    try:
        image = Image.open(image_path)
        return process_image(mode, image, text_prompt, image_path, json_output_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None


def save_processed_image(result_image: Image.Image, output_path: str):
    """Save processed image to file."""
    if result_image:
        result_image.save(output_path)
        print(f"Processed image saved to: {output_path}")
    else:
        print("No image to save.")


def load_detections_json(json_path: str) -> Dict[str, Any]:
    """Load detection results from JSON file."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return {}


if __name__ == "__main__":
    # Example usage
    print("Florence2 + SAM2 Pipeline (without Gradio)")
    print("Available modes:", IMAGE_INFERENCE_MODES)
    
    # Example: Process an image with open vocabulary detection
    image_path = "/content/1.png"
    result_image, caption = process_image_from_path(
        image_path, 
        IMAGE_OPEN_VOCABULARY_DETECTION_MODE, 
        "wheels",
        "image_detections2.json"
    )
    if result_image:
        save_processed_image(result_image, "output_image.jpg")
        # Load and view detection results
        detections_data = load_detections_json("image_detections.json")
        print("Detection results:", detections_data)
    
    # Example: Process a video
    # video_path = "path/to/your/video.mp4"
    # output_video = process_video(video_path, "ball, player, rim", "video_detections.json")
    # if output_video:
    #     print(f"Processed video saved to: {output_video}")
    #     # Load and view video detection results
    #     video_detections = load_detections_json("video_detections.json")
    #     print("Video detection results:", video_detections)