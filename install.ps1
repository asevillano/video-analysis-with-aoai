# Installs project dependencies in the current Python environment.
#
# Workaround: moviepy 2.2.1 declares `pillow<12.0`, but this project pins
# Pillow 12.2.0 for security reasons. Pillow is only used by our own frame-
# stamping code (`_stamp_video_time`), not by moviepy APIs, so the override
# is safe at runtime — but pip's resolver refuses to install both together.
#
# Strategy: install moviepy with --no-deps and pin its transitive deps
# explicitly via requirements.txt, then install everything else normally.

$ErrorActionPreference = "Stop"

Write-Host "Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

Write-Host "Installing moviepy 2.2.1 without its declared dependencies..." -ForegroundColor Cyan
python -m pip install --no-deps moviepy==2.2.1

Write-Host "Installing the rest of the requirements..." -ForegroundColor Cyan
python -m pip install -r requirements.txt

Write-Host "Done. Pillow $(python -c 'import PIL; print(PIL.__version__)') and moviepy installed." -ForegroundColor Green
