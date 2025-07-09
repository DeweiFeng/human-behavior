from abs_vl import VisionLanguageModel  # your original class
from intern_vl_utils import (
    load_video, split_model, generation_config
)
from transformers import AutoModel, AutoTokenizer
import torch

class InternVL(VisionLanguageModel):
    def __init__(self, model_id='OpenGVLab/InternVL3-8B', device='cuda'):
        print("Loading InternVL model...")
        self.model_id = model_id
        self.device = device
        self.device_map = split_model(model_id)

        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            load_in_8bit=True,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=self.device_map
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=False
        )

        print("InternVL model and tokenizer loaded.")
        self.generation_config = generation_config

    def fetch_inputs(self, vision_input_path, prompt, bound=None, input_size=448, num_segments=8, max_num=1):
        # Here vision_input_path is assumed to be a video
        pixel_values, num_patches_list = load_video(
            vision_input_path, bound=bound, input_size=input_size,
            num_segments=num_segments, max_num=max_num
        )
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        meta_data = {
            "input_type": "video",
            "video_path": vision_input_path,
            "num_patches_list": num_patches_list,
            "prompt": prompt
        }
        return pixel_values, num_patches_list, meta_data

    def generate(self, pixel_values, num_patches_list, prompt, history=None):
        """
        Run inference using InternVL model.
        """
        # Construct the question prefix
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        question = f"{video_prefix}{prompt}"

        response, history = self.model.chat(
            self.tokenizer,
            pixel_values,
            question,
            self.generation_config,
            num_patches_list=num_patches_list,
            history=history,
            return_history=True
        )
        return response, history
