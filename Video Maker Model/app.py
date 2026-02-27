from prompt_guard import is_prompt_allowed
from prompt_refiner import refine_prompt
from video_generator import generate_frames
from postprocess import frames_to_video
from ui import build_ui


def run_pipeline(user_prompt):
    allowed, msg = is_prompt_allowed(user_prompt)
    if not allowed:
        return None, msg

    refined = refine_prompt(user_prompt)
    frames = generate_frames(refined)
    video_path = frames_to_video(frames)

    return video_path, "Video generated successfully!"


if __name__ == "__main__":
    ui = build_ui(run_pipeline)
    ui.launch()
