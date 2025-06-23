import torch
import json
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
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

class QwenVL:
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

    def prepare_message(self, video_input, prompt, is_video_path=False, fps=1.0, max_pixels=None):
        """Prepare message dict and meta data"""
        if is_video_path:
            video_content = {
                "type": "video",
                "video": f"file://{video_input}",
                "fps": fps
            }
            if max_pixels:
                video_content["max_pixels"] = max_pixels

            # Get video duration
            duration_sec = get_video_duration(video_input)

            meta_data = {
                "input_type": "video",
                "prompt": prompt,
                "fps": fps,
                "video_path": video_input,
                "duration_sec": duration_sec
            }

        else:
            num_frames = len(video_input)
            duration_sec = num_frames / fps if fps > 0 else 0

            video_content = {
                "type": "video",
                "video": video_input,
                "fps": fps
            }

            meta_data = {
                "input_type": "frames",
                "prompt": prompt,
                "fps": fps,
                "num_frames": num_frames,
                "duration_sec": duration_sec
            }

        messages = [{
            "role": "user",
            "content": [
                video_content,
                {"type": "text", "text": prompt}
            ]
        }]

        return messages, meta_data

    def process_inputs(self, messages):
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


    def save_outputs(self, output_text, save_path, meta_data={}):
        """Append outputs + meta-data to json (or create new file if not exists)"""
        save_entry = {
            "outputs": output_text,
            "meta": meta_data
        }

        # If file exists → load existing list
        if os.path.exists(save_path):
            with open(save_path, "r", encoding="utf-8") as f:
                try:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        print(f"Warning: Existing JSON is not a list — overwriting.")
                        existing_data = []
                except json.JSONDecodeError:
                    print(f"Warning: JSON decode error — overwriting.")
                    existing_data = []
        else:
            existing_data = []

        # Append new entry
        existing_data.append(save_entry)

        # Write back to file
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=2)

        print(f"Appended output to {save_path} (total entries: {len(existing_data)})")