from qwen_vl import QwenVL
from utils import get_frame_paths

if __name__ == "__main__":
    # Instantiate model wrapper
    qwen = QwenVL(use_flash_attention=False)

    # # Example 1: Inference with frame list

    # # define the frame config to retrieve the frame paths
    # frame_config = {
    #     "directory":  "/home/human-behavior/autism/data/ados_videos_and_frames/sample_1/anticipation_of_routine_1_frames",
    #     "slice_start": 0.0,
    #     "slice_end": 0.2}

    # frame_paths = get_frame_paths(
    #     directory=frame_config["directory"],
    #     slice_start=frame_config["slice_start"],
    #     slice_end=frame_config["slice_end"]
    # )

    # frame_config["frame_paths"] = frame_paths

    # # Inference with frames
    prompt = "Describe this video."
    # messages, meta_data = qwen.prepare_message(video_input=frame_paths, prompt=prompt, is_video_path=False, fps=1.0)
    # # add the frame config to the meta data
    # meta_data["frame_config"] = frame_config

    # inputs = qwen.process_inputs(messages)
    # outputs = qwen.generate(inputs)

    # qwen.save_outputs(outputs, save_path="/home/human-behavior/autism/data/qwen_vl_outputs/frames_output.json", meta_data=meta_data)

    # Example 2: Inference with mp4 video
    video_path = "/home/human-behavior/autism/data/ados_videos_and_frames/sample_1/video_segments/anticipation_of_routine_1.mp4"
    messages, meta_data = qwen.prepare_message(video_input=video_path, prompt=prompt, is_video_path=True, fps=1.0)
    inputs = qwen.process_inputs(messages)
    outputs = qwen.generate(inputs)

    # Save output
    qwen.save_outputs(outputs, save_path="/home/human-behavior/autism/data/qwen_vl_outputs/frames_output.json", meta_data=meta_data)