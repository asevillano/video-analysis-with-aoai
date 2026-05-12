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

Copy [.env-sample](.env-sample) to `.env` and fill in your values:

```env
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=<your-multimodal-deployment-name>

# Optional — only required if you authenticate with API key (see Authentication below)
AZURE_OPENAI_API_KEY=<your-api-key>

# Set to True to enable audio transcription via Whisper. Defaults to False.
USE_WHISPER=False
# Only required if USE_WHISPER=True
WHISPER_ENDPOINT=https://<your-whisper-resource>.openai.azure.com/
WHISPER_API_KEY=<your-whisper-api-key>
WHISPER_DEPLOYMENT_NAME=<your-whisper-deployment-name>
```

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

## Deploying to Azure

To deploy the application to Azure as a containerized web app:

1. Build and push the Docker image to Azure Container Registry — see [Build and store an image by using Azure Container Registry](https://learn.microsoft.com/training/modules/deploy-run-container-app-service/3-exercise-build-images).
2. Create and deploy the web app from the image — see [Create and deploy a web app from a Docker image](https://learn.microsoft.com/training/modules/deploy-run-container-app-service/5-exercise-deploy-web-app).

When deploying to Azure App Service, prefer **Managed Identity** (Entra ID) over API keys: assign the *Cognitive Services OpenAI User* role to the App Service's managed identity on the Azure OpenAI resource, and **do not** set `AZURE_OPENAI_API_KEY` in the app settings.
