from moviepy import VideoFileClip
from torchvision.models.video import MViT_V2_S_Weights
import torch

def get_clip_duration(video_path: str) -> int:
    """
    Get the duration of a video clip in seconds.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        int: Duration of the video in seconds
    """
    clip = VideoFileClip(video_path)
    duration = int(clip.duration)
    clip.close()
    return duration

# def preprocess_video(video):
#     """
#     Preprocesses the input video by selecting specific frames, applying transformations, 
#     and preparing it for model input.
#     Args:
#         video (torch.Tensor): A 4D tensor representing the video with shape 
#                               (num_frames, height, width, channels).
#     Returns:
#         torch.Tensor: A 5D tensor representing the preprocessed video with shape 
#                       (1, num_selected_frames, channels, height, width).
#     """
    
#     frames = video[65:85, :, :, :]
#     factor = (85 - 65) / (((85 - 65) / 25) * 21)

#     final_frames = None
#     transforms_model = MViT_V2_S_Weights.KINETICS400_V1.transforms()
#     selected_frames = []
#     for j in range(len(frames)):
#         if j % factor < 1:
#             selected_frames.append(j)
#             if final_frames is None:
#                 final_frames = frames[j, :, :, :].unsqueeze(0)
#             else:
#                 final_frames = torch.cat((final_frames, frames[j, :, :, :].unsqueeze(0)), 0)

#     final_frames = final_frames.permute(0, 3, 1, 2)  # Convert to CHW format
#     final_frames = transforms_model(final_frames)
#     return final_frames.unsqueeze(0)  # Add batch dimension


# Preprocess the video
def preprocess_video(video):  
    frames = video[5:25, :, :, :]
    factor = (25 - 5) / (((25 - 5) / 25) * 21)

    final_frames = None
    transforms_model = MViT_V2_S_Weights.KINETICS400_V1.transforms()
    selected_frames = []
    for j in range(len(frames)):
        if j % factor < 1:
            selected_frames.append(j)
            if final_frames is None:
                final_frames = frames[j, :, :, :].unsqueeze(0)
            else:
                final_frames = torch.cat((final_frames, frames[j, :, :, :].unsqueeze(0)), 0)
    final_frames = final_frames.permute(0, 3, 1, 2)  # Convert to CHW format
    final_frames = transforms_model(final_frames)
    return final_frames.unsqueeze(0)  # Add batch dimension