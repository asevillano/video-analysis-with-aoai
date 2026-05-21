# Import libraries
import streamlit as st
import cv2
import os
import platform
import time
import json
import atexit
import glob
import signal
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy import VideoFileClip
from openai import AzureOpenAI
import base64
import yt_dlp
from yt_dlp.utils import download_range_func
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from prompts import *
from video_summary import (
    _parse_analysis_json,
    _extract_summary_items,
    render_final_summary,
)

# Silence harmless ConnectionResetError cleanup noise from the Windows ProactorEventLoop
# (sockets closed by the remote server after async HTTPS requests complete).
import asyncio
if sys.platform == 'win32':
    def _silence_proactor_reset(loop, context):
        exc = context.get('exception')
        if isinstance(exc, ConnectionResetError):
            return
        loop.default_exception_handler(context)
    try:
        asyncio.get_event_loop().set_exception_handler(_silence_proactor_reset)
    except RuntimeError:
        pass

# Helper to locate a TrueType font on the current OS
def _get_font_path() -> str:
    if platform.system() == "Windows":
        win_font = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "arial.ttf")
        if os.path.isfile(win_font):
            return win_font
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        if os.path.isfile(path):
            return path
    return ""

# Stamp 'video_time: MM:SS:mmm' on a black stripe added below the frame
def _stamp_video_time(frame, timestamp_sec, font_size=16):
    minutes = int(timestamp_sec // 60)
    seconds = int(timestamp_sec % 60)
    milliseconds = int((timestamp_sec - int(timestamp_sec)) * 1000)
    timestamp = f"{minutes:02}:{seconds:02}:{milliseconds:03}"
    timestamp_text = f"video_time: {timestamp}"

    font_path = _get_font_path()
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()

    stripe_height = font_size + 4
    new_frame_height = frame.shape[0] + stripe_height
    new_frame = np.zeros((new_frame_height, frame.shape[1], 3), dtype=np.uint8)
    new_frame[:frame.shape[0], :] = frame

    pil_img = Image.fromarray(new_frame)
    draw = ImageDraw.Draw(pil_img)
    draw.rectangle([(0, frame.shape[0]), (frame.shape[1], new_frame_height)], fill=(0, 0, 0))
    draw.text((5, frame.shape[0] + 1), timestamp_text, font=font, fill=(255, 255, 255))
    return np.array(pil_img)

# Default configuration
SEGMENT_DURATION = 16 # In seconds, Set to 0 to not split the video
DEFAULT_TEMPERATURE = 0.5
RESIZE_OF_FRAMES = 1
FRAMES_PER_SECOND = 3
REASONING_EFFORT = "medium" # "none", "low", "medium" or "high"

# Pricing for the Azure OpenAI model (USD per 1M tokens). Override in .env to match your
# deployment's pricing. Defaults reflect GPT-5.2 Global list prices (Azure OpenAI).
AOAI_PRICE_INPUT_PER_1M = float(os.environ.get("AOAI_PRICE_INPUT_PER_1M", "1.75"))
AOAI_PRICE_OUTPUT_PER_1M = float(os.environ.get("AOAI_PRICE_OUTPUT_PER_1M", "14.00"))
# Whisper pricing (USD per minute of audio). Default: Azure OpenAI Whisper list price.
WHISPER_PRICE_PER_MIN = float(os.environ.get("WHISPER_PRICE_PER_MIN", "0.006"))

def _compute_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens / 1_000_000.0) * AOAI_PRICE_INPUT_PER_1M + \
           (completion_tokens / 1_000_000.0) * AOAI_PRICE_OUTPUT_PER_1M

def _compute_whisper_cost(duration_sec: float) -> float:
    return (duration_sec / 60.0) * WHISPER_PRICE_PER_MIN

# Tiktoken-based token counter for prompt text. Falls back to a coarse char/4
# estimate if tiktoken is not installed or fails to load an encoder.
try:
    import tiktoken
    try:
        _TIKTOKEN_ENC = tiktoken.get_encoding("o200k_base")
    except Exception:
        _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TIKTOKEN_ENC = None

def _count_text_tokens(text: str) -> int:
    if not text:
        return 0
    if _TIKTOKEN_ENC is not None:
        try:
            return len(_TIKTOKEN_ENC.encode(text))
        except Exception:
            pass
    # Fallback: rough estimate (~4 chars per token).
    return max(1, len(text) // 4)

def _get_video_duration_sec(path: str) -> float:
    try:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return (n / fps) if fps > 0 else 0.0
    except Exception:
        return 0.0

# Load configuration
load_dotenv(override=True)

# System prompt for the Purpose
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", GENERIC_SYSTEM_PROMPT)
#SYSTEM_PROMPT = SYSTEM_PROMPT_COMBINED

# Use cases shown in the sidebar selector. The first one is the default.
USE_CASES = {
    "Generic video description": GENERIC_SYSTEM_PROMPT,
    "Riots / violent behavior detection": SYSTEM_PROMPT_RIOTS,
    "Abandoned objects detection": SYSTEM_PROMPT_ABANDONED_OBJECTS,
    "Combined (riots + abandoned objects)": SYSTEM_PROMPT_COMBINED,
    "Shoplifting detection": SYSTEM_PROMPT_SHOPLIFTING,
}

# Whisper: enable/disable from .env (USE_WHISPER=true|false). Defaults to False.
USE_WHISPER = os.environ.get("USE_WHISPER", "False").strip().lower() in ("true", "1", "yes")

# Configuration of OpenAI
aoai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
aoai_api_version = os.environ.get("AZURE_OPENAI_API_VERSION", '2025-04-01-preview')
aoai_model_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")

# Create AOAI client once per Streamlit session/process. @st.cache_resource ensures
# the client (and the underlying credential / token provider) is created only on
# the first run and reused across every script rerun.
# Authentication: if the AZURE_OPENAI_API_KEY environment variable is set, the
# client authenticates using that API key. Otherwise it falls back to Microsoft
# Entra ID via DefaultAzureCredential (which tries env vars, Managed Identity,
# Azure CLI, VS Code, etc., in order) and a bearer token provider scoped to
# Cognitive Services.
@st.cache_resource(show_spinner=False)
def _get_aoai_client():
    print(f'aoai_endpoint: {aoai_endpoint}, aoai_model_name: {aoai_model_name} with reasoning: {REASONING_EFFORT}')
    if api_key := os.environ.get("AZURE_OPENAI_API_KEY"):
        print("Using API key authentication for Azure OpenAI")
        aoai_client = AzureOpenAI(
            azure_deployment=aoai_model_name,
            api_version=aoai_api_version,
            azure_endpoint=aoai_endpoint,
            api_key=api_key
        )
    else:
        print("Using Azure AD authentication for Azure OpenAI") 
        credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(
            credential, "https://cognitiveservices.azure.com/.default"
        )
        aoai_client = AzureOpenAI(
            azure_deployment=aoai_model_name,
            api_version=aoai_api_version,
            azure_endpoint=aoai_endpoint,
            azure_ad_token_provider=token_provider
        )
    return aoai_client

aoai_client = _get_aoai_client()

# Configuration of Whisper
if USE_WHISPER:
    whisper_endpoint = os.environ.get("WHISPER_ENDPOINT")
    whisper_model_name = os.environ.get("WHISPER_DEPLOYMENT_NAME")

    @st.cache_resource(show_spinner=False)
    def _get_whisper_client():
        print(f'whisper_endpoint: {whisper_endpoint}, whisper_model_name: {whisper_model_name}')
        if whisper_apikey:= os.environ.get("WHISPER_API_KEY"):
            print("Using API key authentication for Whisper in Azure OpenAI")
            whisper_client = AzureOpenAI(
                azure_deployment=whisper_model_name,
                api_version=os.environ.get("WHISPER_API_VERSION", '2024-02-01'),
                azure_endpoint=whisper_endpoint,
                api_key=whisper_apikey
            )
        else:
            print("Using Azure AD authentication for Whisper in Azure OpenAI") 
            credential = DefaultAzureCredential()
            token_provider = get_bearer_token_provider(
                credential, "https://cognitiveservices.azure.com/.default"
            )
            whisper_client = AzureOpenAI(
                azure_deployment=whisper_model_name,
                api_version=os.environ.get("WHISPER_API_VERSION", '2024-02-01'),
                azure_endpoint=whisper_endpoint,
                azure_ad_token_provider=token_provider
            )
        return whisper_client

    whisper_client = _get_whisper_client()

# Function to encode a local video into frames
def process_video(video_path, frames_per_second=FRAMES_PER_SECOND, resize=RESIZE_OF_FRAMES, output_dir='', temperature = DEFAULT_TEMPERATURE, segment_offset=0):
    base64Frames = []

    # Prepare the video analysis
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    # Number of source frames to skip between extracted frames, derived from the desired frames-per-second sampling rate
    if frames_per_second <= 0:
        frames_to_skip = 1
    else:
        frames_to_skip = max(1, int(round(fps / frames_per_second)))
    curr_frame=0

    # Prepare to write the frames to disk
    if output_dir != '': # if we want to write the frame to disk
        os.makedirs(output_dir, exist_ok=True)
        frame_count = 1

    # Loop through the video reading frames sequentially and keeping one every `frames_to_skip`.
    # Sequential read avoids the keyframe-snapping behavior of cv2.CAP_PROP_POS_FRAMES on H.264.
    while True:
        success, frame = video.read()
        if not success:
            break

        if curr_frame % frames_to_skip == 0:
            # Resize the frame to save tokens and get faster answer from the model. resize<=1 means no resize.
            if resize > 1:
                height, width, _ = frame.shape
                frame = cv2.resize(frame, (width // resize, height // resize))

            # Compute absolute timestamp from the start of the ORIGINAL video and stamp it on the frame
            timestamp_sec = segment_offset + (curr_frame / fps if fps > 0 else 0)
            frame = _stamp_video_time(frame, timestamp_sec)

            _, buffer = cv2.imencode(".jpg", frame)

            # Save frame as JPG file
            if output_dir != '': # if we want to write the frame to disk
                frame_filename = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_frame_{frame_count}.jpg")
                print(f'Saving frame {frame_filename}')
                with open(frame_filename, "wb") as f:
                    f.write(buffer)
                frame_count += 1

            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

        curr_frame += 1
    video.release()
    print(f"Extracted {len(base64Frames)} frames")
    
    return base64Frames

# Function to transcript the audio from the local video with Whisper
def process_audio(video_path):

    transcription_text = ''
    try:
        base_video_path, _ = os.path.splitext(video_path)
        audio_path = f"{base_video_path}.mp3"
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(audio_path, bitrate="32k")
        clip.audio.close()
        clip.close()
        print(f"Extracted audio to {audio_path}")

        # Transcribe the audio. Open inside a `with` so the file handle is
        # released as soon as the request returns (otherwise on Windows the
        # subsequent os.remove(segment) can fail with WinError 32).
        with open(audio_path, "rb") as audio_f:
            transcription = whisper_client.audio.transcriptions.create(
                model=whisper_model_name,
                file=audio_f,
            )
        transcription_text = transcription.text
        print("Transcript: ", transcription_text + "\n\n")
    except Exception as ex:
        print(f'ERROR: {ex}')
        transcription_text = ''
    finally:
        # Clean up the intermediate mp3 so it doesn't linger in temp/.
        try:
            if 'audio_path' in locals() and os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception:
            pass

    return transcription_text

# Function to analyze the video with AOAI
def analyze_video(base64frames, system_prompt, user_prompt, transcription, temperature):
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    # Pre-compute text-token breakdown so we can attribute prompt tokens to
    # system / user-text / images (the API only returns the aggregate).
    system_tokens = _count_text_tokens(system_prompt)
    user_text_tokens = _count_text_tokens(user_prompt)
    transcription_tokens = _count_text_tokens(transcription) if transcription else 0
    usage["system_tokens"] = system_tokens
    usage["user_text_tokens"] = user_text_tokens
    usage["transcription_tokens"] = transcription_tokens
    usage["image_tokens"] = 0
    try:
        if transcription != '': # Include the audio transcription
            response = aoai_client.chat.completions.create(
                model=aoai_model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}, #"These are the frames from the video.",},
                    {"role": "user", "content": [
                        *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "high"}}, base64frames),
                        {"type": "text", "text": f"The audio transcription is: {transcription}"}
                    ]}
                ],
                #temperature=temperature, #0.5,
                #max_tokens=4096,
                max_completion_tokens=8192,
                reasoning_effort=REASONING_EFFORT,
                response_format={"type": "json_object"},
            )
        else: # Without the audio transcription
            response = aoai_client.chat.completions.create(
                model=aoai_model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}, #"These are the frames from the video.",},
                    {"role": "user", "content": [
                        *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "high"}}, base64frames),
                    ]}
                ],
                #temperature=temperature,
                #max_tokens=4096,
                max_completion_tokens=8192,
                reasoning_effort=REASONING_EFFORT,
                response_format={"type": "json_object"},
            )

        json_response = json.loads(response.model_dump_json())
        #print(f'RESPONSE: [{response.model_dump_json(indent=2)}]')
        usage_raw = json_response.get('usage') or {}
        usage["prompt_tokens"] = int(usage_raw.get('prompt_tokens', 0) or 0)
        usage["completion_tokens"] = int(usage_raw.get('completion_tokens', 0) or 0)
        usage["total_tokens"] = int(usage_raw.get('total_tokens', 0) or 0)
        # Image tokens = prompt_tokens - (text tokens + small chat-format overhead).
        # We don't know the exact overhead, so we attribute the remainder to images
        # and floor at 0.
        text_total = system_tokens + user_text_tokens + transcription_tokens
        usage["image_tokens"] = max(0, usage["prompt_tokens"] - text_total)
        response = json_response['choices'][0]['message']['content']

    except Exception as ex:
        print(f'ERROR: {ex}')
        response = f'ERROR: {ex}'

    return response, usage

# Split the video in segments of N seconds (by default 3 minutes). If segment_length is 0 the full video is processed
def _cleanup_temp_dir(temp_dir='temp'):
    """Delete leftover files inside the temp/ folder.

    Called at the start of every new analysis (to wipe segments from the
    previous run) and registered with atexit so the folder is also emptied
    when the app shuts down. We don't fight Streamlit's MediaFileManager on
    Windows by deleting segments mid-run; we just clean up between runs.
    """
    if not os.path.isdir(temp_dir):
        return
    for path in glob.glob(os.path.join(temp_dir, '*')):
        try:
            if os.path.isfile(path):
                os.remove(path)
        except Exception as ex:
            # Silently ignore: file may still be locked by the browser; it
            # will be cleaned up on the next run or on exit.
            print(f'INFO: skip cleanup of {path}: {ex}')

atexit.register(_cleanup_temp_dir)

# On Ctrl+C / SIGTERM, Streamlit (Tornado) sometimes hangs in "Stopping..."
# until WebSocket clients disconnect, which prevents atexit from running and
# leaves segments in temp/. Hook the signals to force-clean and exit.
def _handle_shutdown_signal(signum, frame):
    print(f'\nReceived signal {signum}. Cleaning temp/ and exiting...')
    try:
        _cleanup_temp_dir()
    finally:
        os._exit(0)

for _sig in (signal.SIGINT, signal.SIGTERM):
    try:
        signal.signal(_sig, _handle_shutdown_signal)
    except (ValueError, OSError):
        # signal.signal can only be called from the main thread. Streamlit
        # reruns the script in the main thread on every interaction, so this
        # normally succeeds; ignore otherwise.
        pass

def split_video(video_path, output_dir, segment_length=180, start_second=0):
    # Make sure the output directory exists so moviepy doesn't fall back to writing
    # intermediate files in the project root.
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()

    if segment_length == 0: # Do not split
        segment_length = int(duration)

    # Clamp the starting second to the video duration
    start_second = max(0, min(int(start_second), int(duration)))

    # Open a fresh VideoFileClip per segment. Sharing one VideoFileClip across
    # multiple subclipped()/close() calls breaks moviepy's underlying FFmpeg
    # reader (subclips share the reader; closing one leaves the parent's
    # `proc` set to None, causing 'NoneType has no attribute stdout' on the
    # next iteration). Reopening per segment is slightly slower but reliable.
    for start_time in range(start_second, int(duration), segment_length):
        end_time = min(start_time + segment_length, duration)
        output_file = os.path.join(
            output_dir,
            f'{os.path.splitext(os.path.basename(video_path))[0]}_segment_{start_time}-{int(end_time)}_secs.mp4'
        )
        clip = VideoFileClip(video_path)
        try:
            # moviepy 2.x uses subclipped(); fall back to subclip() for moviepy 1.x
            sub = clip.subclipped(start_time, end_time) if hasattr(clip, 'subclipped') else clip.subclip(start_time, end_time)
            sub.write_videofile(
                output_file,
                codec='libx264',
                audio_codec='aac',
                # Keep moviepy's temp audio file (*_TEMP_MPY_wvf_snd.*) inside
                # the same temp folder instead of polluting the project root.
                temp_audiofile_path=output_dir or '.',
                logger=None,
            )
        finally:
            try:
                clip.close()
            except Exception:
                pass
        yield output_file, start_time

# Process the video
def execute_video_processing(st, segment_path, system_prompt, user_prompt, temperature, segment_offset=0):
    # Show the video on the screen
    st.write(f"Video: {segment_path}:")
    st.video(segment_path)

    with st.spinner(f"Analyzing video segment: {segment_path}"):
        # Extract frames at the configured frames-per-second sampling rate. Adjust `frames_per_second` to change it
        with st.spinner(f"Extracting frames..."):
            inicio = time.time()
            if save_frames:
                output_dir = 'frames'
            else:
                output_dir = ''
            base64frames = process_video(segment_path, frames_per_second=frames_per_second, resize=resize, output_dir=output_dir, temperature=temperature, segment_offset=segment_offset)
            fin = time.time()
            print(f'\t>>>> Frames extraction took {(fin - inicio):.3f} seconds <<<<')
            ### st.write(f'Extracted {len(base64frames)} frames in {(fin - inicio):.3f} seconds')

        # Extract the transcription of the audio
        if audio_transcription:
            msg = f'Analyzing frames and audio with {aoai_model_name}...'
            with st.spinner(f"Transcribing audio from video file..."):
                inicio = time.time()
                transcription = process_audio(segment_path)
                fin = time.time()
            ### st.write(f'Transcription finished in {(fin - inicio):.3f} seconds')
            print(f'Transcription: [{transcription}]')
            if show_transcription:
                st.markdown(f"**Transcription**: {transcription}", unsafe_allow_html=True)
            print(f'\t>>>> Audio transcription took {(fin - inicio):.3f} seconds <<<<')
        else:
            msg = f'Analyzing frames with {aoai_model_name}...'
            transcription = ''
        # Analyze the video frames and the audio transcription with AOAI
        with st.spinner(msg):
            inicio = time.time()
            analysis, usage = analyze_video(base64frames, system_prompt, user_prompt, transcription, temperature)
            fin = time.time()
        print(f'\t>>>> Analysys with {aoai_model_name} took {(fin - inicio):.3f} seconds <<<<')

    ### st.write(f"**Analysis of segment {segment_path}** ({(fin - inicio):.3f} seconds)")
    fin = time.time()
    print(f'\t>>>> {(fin - inicio):.6f} segundos <<<<')
    st.success("Segment analysys completed.")

    # ---- Cost per minute reporting (console only) ----
    segment_duration_sec = _get_video_duration_sec(segment_path)
    aoai_cost = _compute_cost(usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0))
    # Only charge Whisper if transcription actually produced text. If the video has no
    # audio track, process_audio() returns '' and no request was billable.
    whisper_used = bool(audio_transcription and transcription)
    whisper_cost = _compute_whisper_cost(segment_duration_sec) if whisper_used else 0.0
    if audio_transcription and not transcription:
        print('\t>>>> Note: audio transcription requested but no audio was transcribed — Whisper cost not charged for this segment <<<<')
    segment_cost = aoai_cost + whisper_cost
    minutes = segment_duration_sec / 60.0 if segment_duration_sec > 0 else 0.0
    cost_per_minute = (segment_cost / minutes) if minutes > 0 else 0.0
    print(
        f'\t>>>> COST [segment] tokens(in/out/total)={usage.get("prompt_tokens", 0)}/'
        f'{usage.get("completion_tokens", 0)}/{usage.get("total_tokens", 0)} '
        f'[sys={usage.get("system_tokens", 0)} user={usage.get("user_text_tokens", 0)} '
        f'transcript={usage.get("transcription_tokens", 0)} images={usage.get("image_tokens", 0)}] '
        f'duration={segment_duration_sec:.2f}s ({minutes:.3f} min) '
        f'aoai=${aoai_cost:.6f} whisper=${whisper_cost:.6f} '
        f'cost=${segment_cost:.6f} cost/min=${cost_per_minute:.6f} <<<<'
    )

    return analysis, usage, segment_duration_sec, whisper_cost

# Helper to display the model response: pretty-print JSON when possible, otherwise markdown
def display_analysis(st, analysis, label='Description'):
    if not isinstance(analysis, str):
        st.json(analysis, expanded=True)
        return

    text = analysis.strip()
    # Strip ```json ... ``` or ``` ... ``` fences if the model wrapped the JSON in a code block
    if text.startswith('```'):
        lines = text.splitlines()
        if lines[0].startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].startswith('```'):
            lines = lines[:-1]
        text = '\n'.join(lines).strip()

    try:
        parsed = json.loads(text)
        st.markdown(f"**{label}**")
        st.json(parsed, expanded=True)

        #print(f'Parsed analysis JSON: {json.dumps(parsed, indent=2)}')

    except (json.JSONDecodeError, ValueError):
        # Fallback: show as markdown so newlines/markdown formatting are respected
        st.markdown(f"**{label}**\n\n{analysis}", unsafe_allow_html=True)

# Streamlit User Interface
st.set_page_config(
    page_title=f"Video Analysis with {aoai_model_name}",
    layout="centered",
    initial_sidebar_state="auto",
)
st.image("microsoft.png", width=100)
st.title(f'Video Analysis with {aoai_model_name}')

# Initialise session state flags BEFORE rendering any widget so we can disable
# the sidebar / inputs while an analysis is in progress (avoiding a rerun that
# would abort the loop if the user touches the panel).
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'cancel_requested' not in st.session_state:
    st.session_state.cancel_requested = False

# Persistent result accumulators. These survive reruns triggered by the Stop
# button: each segment's result is appended as soon as it is produced, so even
# if Streamlit aborts the loop mid-execution (RerunException), the data is
# still in session_state and can be re-rendered on the next run.
if 'completed_segments' not in st.session_state:
    # list of {'segment_path', 'analysis', 'segment_start'}
    st.session_state.completed_segments = []
if 'summary_items' not in st.session_state:
    st.session_state.summary_items = []
if 'segment_results' not in st.session_state:
    st.session_state.segment_results = []
if 'last_show_summary' not in st.session_state:
    st.session_state.last_show_summary = False
if 'last_seconds_split' not in st.session_state:
    st.session_state.last_seconds_split = SEGMENT_DURATION

# If the previous run requested cancel, the script was rerun by the on_click callback.
# At this point the previous in-flight loop is gone (Streamlit aborted it), so we
# must clear both flags BEFORE rendering the sidebar — otherwise inputs_disabled
# would still be True and the sidebar would render in disabled state.
_cancelled_this_run = st.session_state.cancel_requested
if _cancelled_this_run:
    st.session_state.processing = False
    st.session_state.cancel_requested = False

inputs_disabled = st.session_state.processing

# Surface a deferred validation error from the previous rerun (e.g. user clicked
# Analyze without selecting a video). We store the message in session_state and
# pop it here so it is shown AFTER the sidebar has been re-rendered enabled.
_pending_err = st.session_state.pop('_pending_validation_error', None)
if _pending_err:
    st.error(_pending_err)

with st.sidebar:
    file_or_url = st.selectbox("Video source:", ["File", "URL"], index=0, help="Select the source, file or url", disabled=inputs_disabled)
    initial_split = SEGMENT_DURATION
    if file_or_url == "URL":
        continuous_transmision = st.checkbox('Continuous transmision', False, help="Video of a continuous transmision", disabled=inputs_disabled)

    if USE_WHISPER:
        audio_transcription = st.checkbox('Transcript audio', True, help="Extract the audio transcription and use in the analysis or not", disabled=inputs_disabled)
        if audio_transcription:
            show_transcription = st.checkbox('Show audio transcription', True, help="Present the audio transcription or not", disabled=inputs_disabled)
    else:
        audio_transcription = False
        show_transcription = False

    starting_second = int(st.number_input('Starting second', 0, help="Second of the video at which to start processing. Frames before this second will be skipped.", disabled=inputs_disabled))
    seconds_split = int(st.number_input('Number of seconds to split the video', min_value=0, value=initial_split, step=1, help="The video will be processed in smaller segments based on the number of seconds specified in this field. (0 to not split)", disabled=inputs_disabled))
    frames_per_second = float(st.text_input('Frames per second to extract', FRAMES_PER_SECOND, help="Number of frames to extract per second of video. It can be a decimal number, like 0.5 (one frame every 2 seconds) or 2 (two frames per second).", disabled=inputs_disabled))
    resize = st.number_input("Frames resizing ratio", min_value=1, value=RESIZE_OF_FRAMES, step=1, help="Divider applied to width and height of each frame. 1 = original size (no resize), 2 = half size, 3 = one third, etc. Useful to reduce latency and token consumption.", disabled=inputs_disabled)
    show_summary = st.checkbox('Show final consolidated summary', False, help="Render a final summary across all analyzed segments (extra LLM call at the end).", disabled=inputs_disabled)
    save_frames = st.checkbox('Save the frames to the folder "frames"', False, disabled=inputs_disabled)
    #temperature = float(st.number_input('Temperature for the model', DEFAULT_TEMPERATURE))
    temperature = 0.0

    # Use case selector: choosing a use case overwrites the System Prompt text area.
    def _on_use_case_change():
        st.session_state['system_prompt_text'] = USE_CASES[st.session_state['use_case_select']]

    if 'system_prompt_text' not in st.session_state:
        st.session_state['system_prompt_text'] = SYSTEM_PROMPT

    st.selectbox(
        'Use case',
        list(USE_CASES.keys()),
        index=0,
        help="Pick a predefined use case to populate the System Prompt below.",
        key='use_case_select',
        on_change=_on_use_case_change,
        disabled=inputs_disabled,
    )

    system_prompt = st.text_area('System Prompt', key='system_prompt_text', disabled=inputs_disabled)
    user_prompt = st.text_area('User Prompt', USER_PROMPT, disabled=inputs_disabled)
    print(f'SYSTEM PROMPT: [{SYSTEM_PROMPT}]')
    print(f'USER PROMPT:   [{USER_PROMPT}]')

    # Validate that the number of frames per segment doesn't exceed the model limit (50)
    MAX_FRAMES_PER_SEGMENT = 50
    estimated_frames = int(seconds_split * frames_per_second) if seconds_split > 0 else 0
    if estimated_frames > MAX_FRAMES_PER_SEGMENT:
        st.error(
            f"⚠️ The combination of {seconds_split}s × {frames_per_second} fps = {estimated_frames} frames per segment "
            f"exceeds the model limit of {MAX_FRAMES_PER_SEGMENT} frames. "
            f"Reduce the seconds to split or the frames per second."
        )
        exceeds_frame_limit = True
    else:
        if seconds_split > 0:
            st.caption(f"Estimated frames per segment: {estimated_frames} / {MAX_FRAMES_PER_SEGMENT}")
        exceeds_frame_limit = False

# Prepare the segment directory (segments are transient: written here and deleted after processing)
output_dir = "temp"
os.makedirs(output_dir, exist_ok=True)

# Video file or Video URL
if file_or_url == 'File':
    video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"], disabled=inputs_disabled)
else:
    url = st.text_area("Enter de url:", value='https://www.youtube.com/watch?v=Y6kHpAeIr4c', height=10, disabled=inputs_disabled)

# Analyze the video when the button is pressed
# The button lives inside a placeholder so we can re-render it as a 'Cancel'
# button while the analysis is running. Cancellation is best-effort and is
# checked between segments (not in the middle of a single AOAI call).
analyze_btn_slot = st.empty()

# If the previous run requested cancel, the script was rerun by the on_click callback.
# At this point the previous in-flight loop is gone (Streamlit aborted it), so we
# must clear both flags before rendering the button — otherwise it would stay
# disabled forever showing "Analyze video".
if st.session_state.cancel_requested:
    st.session_state.processing = False
    st.session_state.cancel_requested = False
    st.warning("Analysis stopped by user.")

def _request_cancel():
    st.session_state.cancel_requested = True

def _request_analyze():
    # Fires BEFORE the script body re-executes on this rerun, so the sidebar
    # below will read processing=True and render its widgets disabled.
    st.session_state.processing = True
    st.session_state.cancel_requested = False
    # Wipe previous analysis results immediately so the next rerun renders a
    # clean page even before the new analysis loop has produced its first
    # segment (otherwise the bottom re-render block would briefly show stale
    # results from the previous run).
    st.session_state.completed_segments = []
    st.session_state.summary_items = []
    st.session_state.segment_results = []
    st.session_state.last_show_summary = False

analyze_clicked = analyze_btn_slot.button(
    "Analyze video",
    width='stretch',
    type='primary',
    disabled=exceeds_frame_limit or st.session_state.processing,
    on_click=_request_analyze,
    key='analyze_btn',
)

if analyze_clicked:
    # Validate inputs before flipping into "processing" state so we don't end up
    # with a half-initialised run (e.g. user clicks Analyze without uploading
    # a file, which would crash later when accessing video_file.name).
    # NOTE: `_request_analyze` (the on_click callback) already set processing=True
    # for this rerun, so the sidebar was already rendered as disabled. If validation
    # fails we must reset the flag AND trigger a fresh rerun so the sidebar / uploader
    # come back enabled; st.stop() alone is not enough because no widget can be
    # interacted with to trigger the next rerun.
    _validation_error = None
    if file_or_url == 'File' and 'video_file' in dir() and video_file is None:
        _validation_error = "Please upload a video file before clicking Analyze."
    elif file_or_url == 'URL' and not (url and url.strip()):
        _validation_error = "Please enter a video URL before clicking Analyze."

    if _validation_error is not None:
        st.session_state.processing = False
        st.session_state.cancel_requested = False
        st.session_state['_pending_validation_error'] = _validation_error
        st.rerun()

    st.session_state.processing = True
    st.session_state.cancel_requested = False
    # Wipe any leftover segments from a previous analysis. We don't try to
    # delete them mid-run because Streamlit's MediaFileManager keeps file
    # handles open on Windows.
    _cleanup_temp_dir()
    # Reset persisted result accumulators for this new run (also done in the
    # on_click callback, but repeated here in case the callback path was
    # bypassed). Capture the per-run UI options snapshot.
    st.session_state.completed_segments = []
    st.session_state.summary_items = []
    st.session_state.segment_results = []
    st.session_state.last_show_summary = show_summary
    st.session_state.last_seconds_split = seconds_split
    # Swap the slot to a Cancel button. on_click sets the cancel flag and
    # Streamlit will trigger a rerun, which raises a RerunException at the
    # next st.* call, aborting the loop early.
    analyze_btn_slot.button(
        "Stop Analysis",
        width='stretch',
        type='secondary',
        on_click=_request_cancel,
        key='cancel_btn_active',
        help="Stop processing after the current segment. Results already obtained are preserved and (if enabled) the final summary will be generated from them.",
    )

    # Accumulator for the final summary across all segments (events / incidents / elements).
    # Aliased to session_state lists so progress survives a Stop-triggered rerun.
    summary_items = st.session_state.summary_items
    # Raw per-segment analyses fed to the final consolidation LLM call.
    segment_results = st.session_state.segment_results

    # Cost / usage totals across all segments (console-only reporting).
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_system_tokens = 0
    total_user_text_tokens = 0
    total_transcription_tokens = 0
    total_image_tokens = 0
    total_aoai_cost_usd = 0.0
    total_whisper_cost_usd = 0.0
    total_video_seconds = 0.0
    print(
        f'>>>> PRICING: input=${AOAI_PRICE_INPUT_PER_1M}/1M tokens, '
        f'output=${AOAI_PRICE_OUTPUT_PER_1M}/1M tokens (model={aoai_model_name}), '
        f'whisper=${WHISPER_PRICE_PER_MIN}/min <<<<'
    )

    try:
        # Placeholder shown while the first segment is being prepared (downloaded for URL,
        # written + split for File). It is cleared right before the first segment is
        # processed so it doesn't overlap with the per-segment "Analyzing video..." spinner.
        startup_status = st.empty()
        startup_status.info(f"⏳ Starting video analysis: {url if file_or_url == 'URL' else video_file.name}")

        # Show parameters:
        print(f"PARAMETERS:")
        print(f"file_or_url: {file_or_url}, audio_transcription: {audio_transcription}, seconds to split: {seconds_split}")
        print(f"frames_per_second: {frames_per_second}, resize ratio: {resize}, save_frames: {save_frames}, temperature: {temperature}")

        if file_or_url == 'URL': # Process Youtube video
            st.write(f'Analyzing video from url {url}...')

            # Helper: count actual decodable frames in a downloaded segment.
            # Used to auto-detect refresh-style endpoints (e.g. TfL jamcams), where
            # download_ranges yields a 0-frame mp4 once `start` exceeds the source
            # clip's natural duration.
            def _segment_frame_count(path):
                try:
                    if not os.path.exists(path) or os.path.getsize(path) < 1024:
                        return 0
                    cap = cv2.VideoCapture(path)
                    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    return n
                except Exception:
                    return 0

            # Refresh-style threshold: sources whose natural duration is smaller
            # than this (in seconds) are treated as polling endpoints — we
            # re-download the entire URL on each iteration instead of using
            # download_ranges.
            REFRESH_DURATION_THRESHOLD = 120

            ydl_opts = {
                    #'format': 'best',
                    'format': '(bestvideo[vcodec^=av01]/bestvideo[vcodec^=vp9]/bestvideo)+bestaudio/best',
                    'outtmpl': os.path.join(output_dir, 'segment_%(start)s.mp4'),
                    'force_keyframes_at_cuts': True,
            }
            ydl = yt_dlp.YoutubeDL(ydl_opts)

            # Probe the source to decide strategy.
            source_duration = None
            try:
                with st.spinner("Probing video source..."):
                    info_dict = ydl.extract_info(url, download=False)
                    source_duration = info_dict.get('duration', None)
                print(f'Probed source_duration: {source_duration}')
            except Exception as ex:
                print(f'WARNING: could not probe source duration: {ex}')

            # Decide initial mode: 'dvr' (download_ranges) or 'refresh' (full re-download per poll).
            if continuous_transmision:
                if source_duration is not None and source_duration < REFRESH_DURATION_THRESHOLD:
                    stream_mode = 'refresh'
                else:
                    stream_mode = 'dvr'
                video_duration = 48*60*60
                duracion_segmento = seconds_split if seconds_split > 0 else 180
            else:
                stream_mode = 'dvr'
                video_duration = source_duration if source_duration else 0
                if seconds_split == 0:
                    duracion_segmento = video_duration if video_duration else 180
                else:
                    duracion_segmento = seconds_split

            print(f'Initial stream_mode: {stream_mode}, video_duration: {video_duration}, duracion_segmento: {duracion_segmento}')
            if stream_mode == 'refresh':
                st.info(
                    f"🔄 Refresh-polling mode detected (source duration ≈ {source_duration:.1f}s). "
                    f"The full clip will be re-downloaded every ~{duracion_segmento}s."
                )

            start = starting_second
            iteration = 0
            consecutive_empty = 0
            last_hash = None
            consecutive_same_hash = 0
            MAX_CONSECUTIVE_SAME_HASH = 6  # stop if the source keeps returning the exact same bytes

            while start < video_duration:
                if st.session_state.cancel_requested:
                    st.warning("Analysis stopped by user. Showing results obtained so far.")
                    break

                iteration += 1
                end = start + duracion_segmento

                if stream_mode == 'refresh':
                    # Always overwrite the same filename; the timeline is virtual.
                    filename = os.path.join(output_dir, f'segment_{start}-{end}.mp4')
                    poll_opts = dict(ydl_opts)
                    poll_opts['outtmpl'] = filename
                    poll_opts.pop('download_ranges', None)
                    iter_start_time = time.time()
                    with st.spinner(f"Polling source (virtual t={start}s)..."):
                        try:
                            with yt_dlp.YoutubeDL(poll_opts) as poll_ydl:
                                poll_ydl.download([url])
                        except Exception as ex:
                            print(f'WARNING: refresh download failed: {ex}')
                            time.sleep(min(duracion_segmento, 10))
                            start += duracion_segmento
                            continue
                    segment_path = filename if os.path.exists(filename) else (
                        filename + '.mkv' if os.path.exists(filename + '.mkv') else filename + '.webm'
                    )
                    # Detect a stuck/dead camera by hashing the downloaded bytes.
                    try:
                        import hashlib
                        with open(segment_path, 'rb') as fh:
                            cur_hash = hashlib.md5(fh.read()).hexdigest()
                        if cur_hash == last_hash:
                            consecutive_same_hash += 1
                        else:
                            consecutive_same_hash = 0
                        last_hash = cur_hash
                        if consecutive_same_hash >= MAX_CONSECUTIVE_SAME_HASH:
                            st.warning(
                                f"Source returned identical content {consecutive_same_hash + 1} times in a row. Stopping."
                            )
                            break
                    except Exception:
                        pass
                else:
                    # DVR mode: slice the live timeline with download_ranges.
                    filename = os.path.join(output_dir, f'segment_{start}-{end}.mp4')
                    with st.spinner(f"Downloading video from second {start} to {end}..."):
                        ydl_opts['outtmpl'] = {'default': filename}
                        ydl_opts['download_ranges'] = download_range_func(None, [(start, end)])
                        print(f'start: {start}, video_duration: {video_duration}, duracion_segmento: {duracion_segmento}')
                        try:
                            with yt_dlp.YoutubeDL(ydl_opts) as dvr_ydl:
                                dvr_ydl.download([url])
                        except Exception:
                            break
                    if os.path.exists(filename):
                        segment_path = filename
                    else:
                        segment_path = filename + '.mkv'
                        if not os.path.exists(segment_path):
                            segment_path = filename + '.webm'
                    iter_start_time = None

                print(f"Segment downloaded: {segment_path}")

                # Empty-segment detection: if a DVR segment yields 0 frames after
                # the first one succeeded, the source is actually a refresh-style
                # endpoint — switch strategy mid-run and retry this iteration.
                frame_count = _segment_frame_count(segment_path)
                if frame_count == 0:
                    consecutive_empty += 1
                    print(f'WARNING: segment {segment_path} has 0 frames (consecutive_empty={consecutive_empty})')
                    if stream_mode == 'dvr' and continuous_transmision and iteration > 1 and consecutive_empty >= 1:
                        st.info("🔄 Empty segment detected — switching to refresh-polling mode.")
                        stream_mode = 'refresh'
                        consecutive_empty = 0
                        # Do not advance start; retry this iteration in refresh mode.
                        continue
                    if consecutive_empty >= 3:
                        st.warning("Too many empty segments. Stopping.")
                        break
                    # Advance and try again.
                    start += duracion_segmento
                    if stream_mode == 'refresh':
                        # Pace polling.
                        elapsed = time.time() - iter_start_time if iter_start_time else 0
                        time.sleep(max(0, duracion_segmento - elapsed))
                    continue
                else:
                    consecutive_empty = 0

                # First successful segment: remove the startup placeholder.
                startup_status.empty()

                # Process the video segment (start = absolute offset from the beginning of the original video)
                analysis, usage, segment_duration_sec, whisper_cost = execute_video_processing(st, segment_path, system_prompt, user_prompt, temperature, segment_offset=start)
                display_analysis(st, analysis, label='Description')
                total_prompt_tokens += usage.get('prompt_tokens', 0)
                total_completion_tokens += usage.get('completion_tokens', 0)
                total_system_tokens += usage.get('system_tokens', 0)
                total_user_text_tokens += usage.get('user_text_tokens', 0)
                total_transcription_tokens += usage.get('transcription_tokens', 0)
                total_image_tokens += usage.get('image_tokens', 0)
                total_aoai_cost_usd += _compute_cost(usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0))
                total_whisper_cost_usd += whisper_cost
                total_video_seconds += segment_duration_sec

                # Collect items for the final summary
                parsed_analysis = _parse_analysis_json(analysis)
                summary_items.extend(_extract_summary_items(parsed_analysis, segment_path, start))
                segment_results.append({
                    'segment': os.path.basename(segment_path),
                    'segment_start': start,
                    'analysis_text': analysis if isinstance(analysis, str) else json.dumps(analysis),
                })
                st.session_state.completed_segments.append({
                    'segment_path': segment_path,
                    'analysis': analysis,
                    'segment_start': start,
                })

                # Pace polling so we don't hammer refresh endpoints faster than they update.
                if stream_mode == 'refresh' and iter_start_time is not None:
                    elapsed = time.time() - iter_start_time
                    remaining = duracion_segmento - elapsed
                    if remaining > 0:
                        with st.spinner(f"Waiting {remaining:.1f}s before next poll..."):
                            time.sleep(remaining)

                start += duracion_segmento

        else: # Process the fideo file
            if video_file is not None:
                os.makedirs("temp", exist_ok=True)
                video_path = os.path.join("temp", video_file.name)
            try:
                with open(video_path, "wb") as f:
                    f.write(video_file.getbuffer())

                # Splitting video in segment of N seconds (if seconds is 0 t will not split the video).
                # Wrap the generator in a spinner so the user sees activity while moviepy/ffmpeg
                # writes the next segment to disk (this can take several seconds for large videos
                # and otherwise leaves the UI looking idle between segments).
                segment_iter = split_video(video_path, output_dir, seconds_split, start_second=starting_second)
                segment_index = 0
                while True:
                    segment_index += 1
                    with st.spinner(f"Preparing segment {segment_index}..."):
                        try:
                            segment_path, segment_start = next(segment_iter)
                        except StopIteration:
                            break
                    if st.session_state.cancel_requested:
                        st.warning("Analysis stopped by user. Showing results obtained so far.")
                        break
                    # First segment ready: remove the startup placeholder before showing the video.
                    startup_status.empty()
                    # Process the video segment passing the absolute offset from the beginning of the original video
                    analysis, usage, segment_duration_sec, whisper_cost = execute_video_processing(st, segment_path, system_prompt, user_prompt, temperature, segment_offset=segment_start)
                    display_analysis(st, analysis, label='Description')
                    total_prompt_tokens += usage.get('prompt_tokens', 0)
                    total_completion_tokens += usage.get('completion_tokens', 0)
                    total_system_tokens += usage.get('system_tokens', 0)
                    total_user_text_tokens += usage.get('user_text_tokens', 0)
                    total_transcription_tokens += usage.get('transcription_tokens', 0)
                    total_image_tokens += usage.get('image_tokens', 0)
                    total_aoai_cost_usd += _compute_cost(usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0))
                    total_whisper_cost_usd += whisper_cost
                    total_video_seconds += segment_duration_sec

                    # Collect items for the final summary
                    parsed_analysis = _parse_analysis_json(analysis)
                    summary_items.extend(_extract_summary_items(parsed_analysis, segment_path, segment_start))
                    segment_results.append({
                        'segment': os.path.basename(segment_path),
                        'segment_start': segment_start,
                        'analysis_text': analysis if isinstance(analysis, str) else json.dumps(analysis),
                    })
                    st.session_state.completed_segments.append({
                        'segment_path': segment_path,
                        'analysis': analysis,
                        'segment_start': segment_start,
                    })

            except Exception as ex:
                print(f'ERROR: {ex}')
                st.write(f'ERROR: {ex}')
        # Render the final summary across all analyzed segments
        video_source_label = url if file_or_url == 'URL' else (video_file.name if 'video_file' in dir() and video_file is not None else '')
        st.success(f"Video Analysys completed: {video_source_label}")
        summary_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if show_summary:
            try:
                summary_usage = render_final_summary(st, summary_items, segment_results, aoai_client, aoai_model_name, 'medium', segment_duration_hint=seconds_split) or summary_usage
            except Exception as ex:
                print(f'ERROR rendering summary: {ex}')
        summary_cost_usd = _compute_cost(summary_usage.get('prompt_tokens', 0), summary_usage.get('completion_tokens', 0))

        # ---- Final cost-per-minute report (console only) ----
        total_minutes = total_video_seconds / 60.0 if total_video_seconds > 0 else 0.0
        per_segment_cost_usd = total_aoai_cost_usd + total_whisper_cost_usd
        total_cost_usd = per_segment_cost_usd + summary_cost_usd
        # Cost per minute reflects only per-segment processing (AOAI + Whisper).
        # The final consolidated summary is a one-off call independent of video length,
        # so it is excluded from the per-minute rate but added to TOTAL VIDEO COST.
        avg_cost_per_minute = (per_segment_cost_usd / total_minutes) if total_minutes > 0 else 0.0

        # Probe the first completed segment to report original / resized frame dimensions.
        original_w, original_h = 0, 0
        try:
            first_seg = next(
                (s['segment_path'] for s in st.session_state.completed_segments if os.path.exists(s['segment_path'])),
                None,
            )
            if first_seg:
                cap = cv2.VideoCapture(first_seg)
                original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
        except Exception:
            pass
        resized_w = original_w // resize if resize > 1 and original_w else original_w
        resized_h = original_h // resize if resize > 1 and original_h else original_h
        estimated_frames_per_seg = int(seconds_split * frames_per_second) if seconds_split > 0 else 0
        total_frames_sent = int(round(total_video_seconds * frames_per_second)) if frames_per_second > 0 else 0

        print('')
        print('================ COST SUMMARY ================')
        print(f'Model:                 {aoai_model_name}')
        print(f'Reasoning effort:      {REASONING_EFFORT}')
        print(f'Image detail mode:     high')
        print(f'Frames per second:     {frames_per_second}')
        print(f'Resize ratio:          {resize}x  (1 = no resize)')
        if original_w and original_h:
            print(f'Frame size original:   {original_w}x{original_h} px')
            print(f'Frame size sent:       {resized_w}x{resized_h} px')
        print(f'Segment duration:      {seconds_split} s  (0 = no split)')
        print(f'Frames per segment:    ~{estimated_frames_per_seg}')
        print(f'Frames sent (total):   ~{total_frames_sent}')
        print(f'Audio transcription:   {"enabled (Whisper)" if audio_transcription else "disabled"}')
        print('----------------------------------------------')
        print(f'Pricing: AOAI input=${AOAI_PRICE_INPUT_PER_1M}/1M, output=${AOAI_PRICE_OUTPUT_PER_1M}/1M, whisper=${WHISPER_PRICE_PER_MIN}/min')
        print(f'Segments processed:    {len(segment_results)}')
        print(f'Video duration:        {total_video_seconds:.2f} s ({total_minutes:.3f} min)')
        print(f'Prompt tokens:         {total_prompt_tokens}')
        print(f'  ├─ system prompt:    {total_system_tokens}')
        print(f'  ├─ user prompt:      {total_user_text_tokens}')
        if total_transcription_tokens:
            print(f'  ├─ transcription:    {total_transcription_tokens}')
        print(f'  └─ frames (images):  {total_image_tokens}')
        print(f'Completion tokens:     {total_completion_tokens}')
        print(f'Total tokens:          {total_prompt_tokens + total_completion_tokens}')
        print(f'AOAI cost:             ${total_aoai_cost_usd:.6f} USD')
        print(f'Whisper cost:          ${total_whisper_cost_usd:.6f} USD')
        if show_summary:
            print('---- Final consolidated summary (extra LLM call) ----')
            print(f'Summary prompt tokens:     {summary_usage.get("prompt_tokens", 0)}')
            print(f'Summary completion tokens: {summary_usage.get("completion_tokens", 0)}')
            print(f'Summary total tokens:      {summary_usage.get("prompt_tokens", 0) + summary_usage.get("completion_tokens", 0)}')
            print(f'Final summary cost:        ${summary_cost_usd:.6f} USD')
            print('-----------------------------------------------------')
        print(f'TOTAL VIDEO COST:      ${total_cost_usd:.6f} USD  (full video, all segments'
              + (' + final summary)' if show_summary else ')'))
        print(f'Cost per minute video: ${avg_cost_per_minute:.6f} USD/min'
              + ('  (excludes final summary)' if show_summary else ''))
        print('==============================================')

    finally:
        # Re-enable the button in the same slot so the user can launch a new analysis.
        st.session_state.processing = False
        st.session_state.cancel_requested = False
        analyze_btn_slot.button(
            "Analyze video",
            width='stretch',
            type='primary',
            disabled=exceeds_frame_limit,
            key='analyze_btn_done',
        )
        # The sidebar was rendered earlier in this run with inputs_disabled=True
        # (because processing was True at that point). Flipping the flag now is
        # not enough — the widgets above us in the script have already been
        # drawn for this run and will stay disabled until the next rerun.
        # Force a fresh rerun so the sidebar re-renders with widgets enabled.
        # Results are preserved in session_state and re-rendered by the
        # "Previous analysis results" block below.
        st.rerun()

# Re-render persisted results from previous runs (e.g. after Stop was pressed,
# which aborts the running script via RerunException and would otherwise wipe
# everything from the page). Only runs when there is no active analysis AND
# the current rerun is NOT the one that just launched a new analysis — in that
# rerun the live loop above already rendered every segment, so this block
# would duplicate the output. The list is cleared in the Analyze on_click
# callback, so a fresh click starts from an empty list and nothing stale is
# shown.
if (
    not st.session_state.processing
    and not analyze_clicked
    and st.session_state.completed_segments
):
    st.markdown('---')
    st.subheader('Previous analysis results')
    for seg in st.session_state.completed_segments:
        seg_path = seg['segment_path']
        st.write(f"Video: {seg_path}:")
        if os.path.exists(seg_path):
            st.video(seg_path)
        else:
            st.caption('_(segment file no longer available)_')
        display_analysis(st, seg['analysis'], label='Description')
    if st.session_state.last_show_summary:
        try:
            render_final_summary(
                st,
                st.session_state.summary_items,
                st.session_state.segment_results,
                aoai_client,
                aoai_model_name,
                'medium',
                segment_duration_hint=st.session_state.last_seconds_split,
            )
        except Exception as ex:
            print(f'ERROR rendering persisted summary: {ex}')
