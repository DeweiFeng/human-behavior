import torch
import json
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from abs_vl import VisionLanguageModel  # your original class
import cv2
import os

def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    duration_sec = frame_count / fps if fps > 0 else 0
    cap.release()
    return duration_sec

class QwenVL(VisionLanguageModel):
    def fetch_inputs(self, vision_input_paths, prompt, is_video_path=False, fps=1.0, max_pixels=None):
        """Prepare video content dict and meta data"""
        if is_video_path:
            video_content = {
                "type": "video",
                "video": f"file://{vision_input_paths}",
                "fps": fps
            }
            if max_pixels:
                video_content["max_pixels"] = max_pixels

            # Get video duration
            duration_sec = get_video_duration(vision_input_paths)

            meta_data = {
                "input_type": "video",
                "prompt": prompt,
                "fps": fps,
                "video_path": vision_input_paths,
                "duration_sec": duration_sec
            }

        else:
            # passing the frames directly
            num_frames = len(vision_input_paths)
            duration_sec = num_frames / fps if fps > 0 else 0

            video_content = {
                "type": "video",
                "video": vision_input_paths,
                "fps": fps
            }

            meta_data = {
                "input_type": "frames",
                "prompt": prompt,
                "fps": fps,
                "num_frames": num_frames,
                "duration_sec": duration_sec
            }

        return video_content, meta_data

    def process_inputs(self, video_content, prompt):

        messages = [{
            "role": "user",
            "content": [
                video_content,
                {"type": "text", "text": prompt}
            ]
        }]

        """Process vision + text inputs for model"""
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs
        )
        inputs = inputs.to(self.device)
        return inputs

    def generate(self, inputs, max_new_tokens=1256):
        """Run inference"""
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

    def __init__(self, model_id="Qwen/Qwen2.5-VL-7B-Instruct", device="cuda", use_flash_attention=False):
        print("Loading model...")
        attn_impl = "flash_attention_2" if use_flash_attention else None

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if use_flash_attention else "auto",
            attn_implementation=attn_impl,
            device_map="auto"
        )
        self.model.eval()

        print("Loading processor...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.device = device

    def fetch_inputs(self, vision_input_paths, prompt, is_video_path=False, fps=1.0, max_pixels=None):
        """Prepare video content dict and meta data"""
        if is_video_path:
            video_content = {
                "type": "video",
                "video": f"file://{vision_input_paths}",
                "fps": fps
            }
            if max_pixels:
                video_content["max_pixels"] = max_pixels

            # Get video duration
            duration_sec = get_video_duration(vision_input_paths)

            meta_data = {
                "input_type": "video",
                "prompt": prompt,
                "fps": fps,
                "video_path": vision_input_paths,
                "duration_sec": duration_sec
            }

        else:
            # passing the frames directly
            num_frames = len(vision_input_paths)
            duration_sec = num_frames / fps if fps > 0 else 0

            video_content = {
                "type": "video",
                "video": vision_input_paths,
                "fps": fps
            }

            meta_data = {
                "input_type": "frames",
                "prompt": prompt,
                "fps": fps,
                "num_frames": num_frames,
                "duration_sec": duration_sec
            }

        return video_content, meta_data

    def process_inputs(self, video_content, prompt):

        messages = [{
            "role": "user",
            "content": [
                video_content,
                {"type": "text", "text": prompt}
            ]
        }]

        """Process vision + text inputs for model"""
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs
        )
        inputs = inputs.to(self.device)
        return inputs

    def generate(self, inputs, max_new_tokens=1256):
        """Run inference"""
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        # Trim input tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        return output_text
