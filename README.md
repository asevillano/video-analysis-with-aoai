# Video Analysis with Azure OpenAI

The aim of this repo is to demonstrate the capabilities of Azure OpenAI multimodal models (GPT-4o, GPT-4.1, o-series reasoning models, etc.) to analyze and extract insights from a video file or a video URL (e.g. YouTube).

The steps to process a video are the following:
1. Split the video in segments of N seconds (or process it whole if `0` seconds is specified).
2. Extract frames from each segment at a configurable frames-per-second sampling rate, stamping the absolute video timestamp on each frame.
3. Optionally transcribe the audio with Whisper.
4. Send the frames (and the optional audio transcription) to Azure OpenAI to extract a description, summary, or any other insight driven by the system/user prompt.

## Prerequisites
+ An Azure subscription with [access to Azure OpenAI](https://aka.ms/oai/access).
+ An Azure OpenAI resource (endpoint).
+ A deployment of a multimodal model (e.g. `gpt-4o`, `gpt-4.1`, `o4-mini`, etc.).
+ *(Optional)* A Whisper deployment if you want audio transcription (`USE_WHISPER=True` in the script).
+ Python 3.10 or later. Tested with Python 3.12.
+ [Visual Studio Code](https://code.visualstudio.com/) with the [Python extension](https://code.visualstudio.com/docs/python/python-tutorial).
+ **[ffmpeg](https://ffmpeg.org/)** available in your `PATH`. Required by `yt-dlp` to download partial YouTube segments and by the frame/audio extraction pipeline. On Windows you can install it with:

  ```powershell
  winget install --id=Gyan.FFmpeg -e
  ```

  See the [Troubleshooting](#troubleshooting) section if `ffmpeg` is installed but not detected.

## Set up a Python virtual environment in Visual Studio Code

1. Open the Command Palette (`Ctrl+Shift+P`).
2. Search for **Python: Create Environment**.
3. Select **Venv**.
4. Select a Python interpreter (3.10 or later).
5. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

If you run into problems, see [Python environments in VS Code](https://code.visualstudio.com/docs/python/environments).

## Configuration

Copy [.env-sample](.env-sample) to `.env` and fill in your values. The endpoint can use either the classic `*.openai.azure.com` form or the newer Foundry-style `*.cognitiveservices.azure.com` form:

```env
# --- Azure OpenAI (multimodal model) ---
AZURE_OPENAI_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com/
# Or the classic form:
# AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=<your-multimodal-deployment-name>     # e.g. gpt-5.2, gpt-4o, gpt-4.1, o4-mini

# Optional — only required if you authenticate with API key (see Authentication below)
# AZURE_OPENAI_API_KEY=<your-api-key>

# --- Optional: Whisper for audio transcription ---
USE_WHISPER=False
# Only required if USE_WHISPER=True
WHISPER_ENDPOINT=https://<your-whisper-resource>.openai.azure.com/
WHISPER_API_KEY=<your-whisper-api-key>
WHISPER_DEPLOYMENT_NAME=whisper
```

> **Note:** Whisper currently uses API key authentication, while the multimodal Azure OpenAI client supports both API key and Entra ID (see below). Keep the Whisper resource and its key only if you actually need audio transcription.

### Authentication

The application supports two authentication modes for Azure OpenAI, selected automatically:

- **API key** — used when `AZURE_OPENAI_API_KEY` is defined in the environment.
- **Microsoft Entra ID** (recommended) — used as a fallback when `AZURE_OPENAI_API_KEY` is **not** set. It uses [`DefaultAzureCredential`](https://learn.microsoft.com/python/api/azure-identity/azure.identity.defaultazurecredential), which tries (in order): environment variables, Managed Identity, Azure CLI (`az login`), Visual Studio Code, etc.

To use Entra ID locally:
1. Run `az login`.
2. Make sure your user has the **Cognitive Services OpenAI User** role on the Azure OpenAI resource:

   ```powershell
   az role assignment create `
     --assignee-object-id <YOUR_USER_OBJECT_ID> `
     --assignee-principal-type User `
     --role "Cognitive Services OpenAI User" `
     --scope "/subscriptions/<SUB_ID>/resourceGroups/<RG>/providers/Microsoft.CognitiveServices/accounts/<AOAI_ACCOUNT>"
   ```
3. Make sure `AZURE_OPENAI_API_KEY` is **not** set in your `.env` (or comment it out).

## Running the application

The main script is [video-analysis-with-aoai.py](video-analysis-with-aoai.py). Launch it with Streamlit:

```powershell
streamlit run video-analysis-with-aoai.py
```

A screenshot:

<img src="./Screenshot.png" alt="Sample Screenshot"/>

## UI options (sidebar)

- **Video source**: `File` (upload) or `URL` (YouTube).
- **Continuous transmission** (URL only): treat the source as a live stream.
- **Transcript audio / Show audio transcription**: only available if `USE_WHISPER=True`.
- **Starting second**: skip the first N seconds of the video.
- **Number of seconds to split the video**: segment length. `0` processes the whole video as a single segment.
- **Frames per second to extract**: sampling rate (decimal allowed, e.g. `0.5`).
- **Frames resizing ratio**: divider applied to width/height to reduce token usage and latency.
- **Save the frames to the folder `frames`**: persist extracted frames to disk for inspection.
- **System Prompt / User Prompt**: editable, defaulted from [prompts.py](prompts.py).

> ⚠️ The model accepts a maximum of **50 images per request**. The UI validates that `seconds_to_split × frames_per_second ≤ 50` and disables the **Analyze video** button otherwise.

## Default tunables (in the script)

These are defined at the top of [video-analysis-with-aoai.py](video-analysis-with-aoai.py) and can be edited there:

| Constant | Default | Description |
| --- | --- | --- |
| `SEGMENT_DURATION` | `16` | Default segment length in seconds (`0` = no split). |
| `USE_WHISPER` | `False` | Enable audio transcription via Whisper. Read from the `USE_WHISPER` env var (`true`/`false`). |
| `FRAMES_PER_SECOND` | `3` | Default sampling rate. |
| `RESIZE_OF_FRAMES` | `1` | Default resize divider (1 = original size). |
| `REASONING_EFFORT` | `"medium"` | Reasoning effort for o-series models (`none`, `low`, `medium`, `high`). |
| `DEFAULT_TEMPERATURE` | `0.5` | Default temperature (currently overridden to `0.0` in the UI). |

## Troubleshooting

### `ERROR: You have requested downloading the video partially, but ffmpeg is not installed. Aborting`

`yt-dlp` requires **ffmpeg** to cut and remux YouTube streams when only a segment of the video is requested (which is what this app does whenever `seconds_to_split > 0`).

1. Install ffmpeg (Windows):

   ```powershell
   winget install --id=Gyan.FFmpeg -e
   ```

2. Make sure `ffmpeg.exe` is in your `PATH`. If `winget` reports it is already installed but `ffmpeg -version` fails, locate the binary and add its `bin` folder to the user `PATH`:

   ```powershell
   $ffmpegBin = (Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\Gyan.FFmpeg*" -Recurse -Filter ffmpeg.exe | Select-Object -First 1).DirectoryName
   $userPath  = [Environment]::GetEnvironmentVariable("Path", "User")
   if ($userPath -notlike "*$ffmpegBin*") {
       [Environment]::SetEnvironmentVariable("Path", "$userPath;$ffmpegBin", "User")
   }
   ```

3. **Restart VS Code / your terminal** so the new `PATH` is picked up, then re-run the app.

### `WARNING: [youtube] No supported JavaScript runtime could be found`

A recent `yt-dlp` warning. It does not break downloads today, but YouTube will eventually require a JS runtime. Install Deno to silence it and future-proof the extractor:

```powershell
winget install DenoLand.Deno
```

`yt-dlp` will detect it automatically.

## Deploying to Azure

The repo ships with a ready-to-use [Azure Developer CLI (`azd`)](https://learn.microsoft.com/azure/developer/azure-developer-cli/) template that provisions all required infrastructure as Bicep and deploys the container image in a single command.

### What gets deployed

Defined in [infra/main.bicep](infra/main.bicep):

- **Resource Group**
- **Log Analytics workspace**
- **Azure Container Registry** (Basic, admin disabled)
- **Container Apps Environment** wired to Log Analytics
- **Azure Container App** running the Streamlit app on port `8501`, external ingress, transport `auto` (WebSockets), System-Assigned Managed Identity, ACR pull via identity
- **RBAC**:
  - `AcrPull` on the ACR (for the app's Managed Identity)
  - `Cognitive Services OpenAI User` on the existing Azure OpenAI account (keyless auth via `DefaultAzureCredential`)

The Azure OpenAI account is **not created** by the template — it is referenced as an existing resource (potentially in a different resource group).

### Prerequisites

- [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli) and [Azure Developer CLI (`azd`)](https://learn.microsoft.com/azure/developer/azure-developer-cli/install-azd) installed.
- An existing Azure OpenAI account with a multimodal chat deployment (e.g. `gpt-4o`).
- Owner or User Access Administrator permissions on the target subscription (the template assigns RBAC roles).

### Deploy with `azd`

```powershell
# 1) Sign in
az login
azd auth login

# 2) Create a new azd environment
azd env new video-analysis-dev

# 3) Set required parameters (consumed by infra/main.parameters.json)
azd env set AZURE_LOCATION                westeurope
azd env set AZURE_OPENAI_RESOURCE_GROUP   <rg-of-your-existing-aoai>
azd env set AZURE_OPENAI_ACCOUNT_NAME     <name-of-your-existing-aoai>
azd env set AZURE_OPENAI_DEPLOYMENT_NAME  <your-multimodal-deployment-name>

# 4) Provision infra + build image in ACR + deploy the Container App
azd up
```

When it finishes, `azd` prints the public URL of the app (also exposed as the `SERVICE_WEB_URI` output).

### Common follow-up commands

```powershell
# Redeploy just the application (rebuilds and pushes a new image)
azd deploy

# Re-run only the infra provisioning
azd provision

# Tear everything down
azd down --purge
```

### Notes

- Authentication to Azure OpenAI is **keyless** via the Container App's Managed Identity. No API keys are stored.
- `USE_WHISPER` defaults to `false`. To enable Whisper, set the optional parameters in [infra/main.parameters.json](infra/main.parameters.json) (`useWhisper`, `whisperEndpoint`, `whisperDeploymentName`, `whisperApiKey`) — the key is stored as a Container App secret.
- The Dockerfile installs `ffmpeg`, `libgl1` and DejaVu fonts so that OpenCV, MoviePy and the timestamp overlay work in the container.
- Ephemeral folders (`frames/`, `segments/`, `temp/`) are recreated per run and are lost on container restart. If persistence is needed, mount Azure Files via the Container Apps Environment.
