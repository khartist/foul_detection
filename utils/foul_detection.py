# Standard library imports
import os
from typing import List, Tuple

# Third-party imports
import torch
import torch.nn as nn
from torchvision.io.video import read_video, write_video
from torchvision.models.video import MViT_V2_S_Weights

# Local imports
from vars_model.config.classes import (
    INVERSE_EVENT_DICTIONARY_action_class,
    INVERSE_EVENT_DICTIONARY_offence_severity_class
)
from vars_model.model import MVNetwork
from video_processing import get_clip_duration, preprocess_video


def load_model():
    """
    Loads a pre-trained model for foul detection.

    This function initializes an MVNetwork model with specified parameters,
    loads the model's state from a file, and sets the model to evaluation mode.

    Returns:
        model (MVNetwork): The loaded and initialized model ready for inference.
    """
    print("Loading model...")
    model = MVNetwork(net_name="mvit_v2_s", agr_type="attention")
    model_path = os.path.join(os.getcwd(), 'vars_model', '14_model.pth.tar').replace('\\', '/')
    state = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state['state_dict'])
    model.eval()
    print("Model loaded successfully.")
    return model

# Run inference on a video
def run_inference(model, video_tensor):
    """
    Run inference on a given video tensor using the provided model.

    Args:
        model (torch.nn.Module): The trained model to use for inference.
        video_tensor (torch.Tensor): The input video tensor for which to run inference.

    Returns:
        dictionary: A dictionary containing:
            - offense_results (str): List of top 1 offense severity class prediction with its confidence scores.
            - action_results (str): List of top 1 action class prediction with its confidence scores.
            - offense_confidence (float): Confidence score of the top offense severity class prediction.
            - action_confidence (float): Confidence score of the top action class prediction.
    """
    # print("Running inference...")
    softmax = nn.Softmax(dim=1)

    videos = video_tensor.unsqueeze(0)  # Add batch dimension
    predictions = model(videos)

    pred_offense = predictions[0].unsqueeze(0)
    pred_action = predictions[1].unsqueeze(0)

    offense_scores = softmax(pred_offense)
    action_scores = softmax(pred_action)

    offense_values, offense_indices = torch.topk(offense_scores, 1)
    action_values, action_indices = torch.topk(action_scores, 1)

    result = {
        "offense_label": INVERSE_EVENT_DICTIONARY_offence_severity_class[offense_indices[0][0].item()],
        "offense_confidence": float(offense_values[0][0].item()),
        "action_label": INVERSE_EVENT_DICTIONARY_action_class[action_indices[0][0].item()],
        "action_confidence": float(action_values[0][0].item())
    }

    return result


def detect_foul(video_path: str, stride = 5):
    """
    Detect fouls from a video clip.
    
    Args:
        video_path (str): Path to the video file.
        stride (int, optional): The interval in seconds at which to sample the video. Defaults to 5.
    Returns:
        list: A list of segments where fouls are detected. 
        Each segment is a dictionary containing the detection results.
    """
    model = load_model()
    duration = get_clip_duration(video_path)
    segment = []

    for start_time in range(650, duration, stride):
        end_time = min(start_time + 30, duration)
        video, audio, _ = read_video(video_path, start_pts=start_time, end_pts=end_time, pts_unit="sec")
        
        processed_video = preprocess_video(video)
        result = run_inference(model, processed_video)
        if result["offense_confidence"] >= 0.5 and result["action_confidence"] >= 0.5:
            segment.append(result)

    return segment

# def write_foul_clips(video_path: str, start_end_times: List[Tuple[int, int]], output_path: str):
#     """
#     Write video clips containing fouls to a new video file.
    
#     Args:
#         video_path (str): Path to the video file
#         output_path (str): Path to save the output video file
#     """

#     model = load_model()
#     duration = get_clip_duration(video_path)

#     for start_time, end_time in start_end_times:
#         video, audio, _ = read_video(video_path, start_pts=start_time, end_pts=end_time, pts_unit="sec")
        
#         write_video(os.path.join(output_path, f"clip_{start_time}_{end_time}.mp4"), video_array=video, fps=30, video_codec="h264", audio_array=audio, audio_fps=44100, audio_codec="mp3")



# Main function
# def main():
#     model = load_model()


#     for start_time in range(650, duration, 10):
#         end_time = min(start_time + 30, duration)
#         video, audio, _ = read_video(base_path, start_pts=start_time, end_pts=end_time, pts_unit="sec")

#         print(f"Processing clip from {start_time} to {end_time}")

#         processed_video = preprocess_video(video)

#         offense_results, action_results, offense_confidence, action_confidence = run_inference(model, processed_video)

#         print("\nInference Results:")
#         print("\nOffense Predictions:")
#         for result in offense_results:
#             print(f"  - {result}")
#         print("\nAction Predictions:")
#         for result in action_results:
#             print(f"  - {result}")

    
#         if offense_confidence >= 0.5 and action_confidence >= 0.5:
#             write_video(os.path.join(output_path, f"clip_{start_time}_{end_time}.mp4"), video_array=video, fps=30, video_codec="h264", audio_array=audio, audio_fps=44100, audio_codec="mp3")



