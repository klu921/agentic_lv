
#!/usr/bin/env python3
import os
import json
import time
import subprocess
import asyncio
from pathlib import Path
from google import genai
from together.types.chat_completions import PromptPart
from google.genai import types
from PIL import Image
import base64


with open("env.json", "r") as f:
    env_data = json.load(f)
    gemini_key_PRIV = env_data["gemini_key"]
    os.environ['GEMINI_API_KEY'] = gemini_key_PRIV
    client_gemini = genai.Client(api_key=gemini_key_PRIV)

async def process_single_frame(frame_path, prompt, semaphore, results, output_file, file_lock, frame_num, total_frames):
    """process a single frame with async API call and semaphore control"""

    async with semaphore:
        print(f"Processing {frame_path}")
        try:
            image_bytes = open(frame_path, "rb").read()

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client_gemini.models.generate_content(
                    model = "gemini-2.5-flash",
                    contents = [types.Content(
                        parts = [
                            types.Part(
                                inline_data = types.Blob(data = image_bytes, mime_type = "image/jpeg")
                            ),
                            types.Part(
                                text = prompt
                            )
                        ]
                    )]
                )
            )

            print(f"✓ API response received for {frame_path}")

            result_entry = frame_path.split(".jpg")[0] + " seconds: " + response.text

            async with file_lock:
                results.append(result_entry)
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent = 2)
            
            print(f"✓ Completed {frame_path}")
        except Exception as e:
            print(f"Error processing {frame_path}: {e}")
            return None

async def caption_frames_with_gemini(frames_dir = 'frames', output_file = 'captions/frame_captions.json', api_key = gemini_key_PRIV, max_concurrent = 500):
    """caption frames with gemini"""

    prompt = """Please write in ONE sentence the SUBJECTS, possible actions, and main objects. Please also check the scene-log and the character-log to see if there are any NEW characters, or repeated characters. If there are any NEW characters, add their descriptions to the character-log. If there are repeated characters, add this timestamp to the character-log. If there are NEW scenes/locations, describe them in depth in the scene-log. If they already exist in the log, add the timestamp to the scene-log.
    """

    

    frames_path = Path(frames_dir)
    print(type(frames_path))
    frame_files = sorted([str(f) for f in frames_path.glob("*.jpg")])

    print(f"Processing {len(frame_files)} frames with Gemini API...")

    processed_frames = set()
    results = []
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                existing_results = json.load(f)
                for entry in existing_results:
                    if entry.startswith("frame"):
                        name = entry.split(" seconds:")[0]
                        processed_frames.add(name)
                results = existing_results
        except Exception as e:
            print(f"Warning: Could not load {output_file}: {e}")
            results = []
    
    else:
        with open(output_file, 'w') as f:
            json.dump([], f)
    
    frames_to_process = [frame for frame in frame_files if frame.split(".")[0] not in processed_frames]

    semaphore = asyncio.Semaphore(max_concurrent)
    
    file_lock = asyncio.Lock()

    tasks = [process_single_frame(frame_path, prompt, semaphore, results, output_file, file_lock, i+1, len(frames_to_process)) for i, frame_path in enumerate(frames_to_process)]

    completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)

    return results

if __name__ == "__main__":
    asyncio.run(caption_frames_with_gemini())

    with open("captions/frame_captions.json", "r") as f:
        print("Loading captions...")
        captions = json.load(f)
        sorted_captions = sorted(captions, key=lambda x: x.split(" seconds:")[0])
        print("Sorting captions...")
        with open("captions/frame_captions_sorted.json", "w") as f:
            json.dump(sorted_captions, f, indent=2)
        print("Saved sorted captions.")
        