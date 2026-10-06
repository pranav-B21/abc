# Source before any `uv run` on TACC Vista (GH200, aarch64):  source vista_env.sh
# Upstream assumes apt-installed ffmpeg and a roomy $HOME; neither holds on Vista.

# uv itself (installed to ~/.local/bin by the astral install script).
export PATH="$HOME/.local/bin:$PATH"

# FFmpeg shared libs for torchcodec / export_mcap.py. The system ffmpeg is broken
# on compute nodes (missing libunwind.so.8), so use the conda-forge build.
ABC_FFMPEG_PREFIX=/work/11138/pranavbelligundu/vista/envs/ffmpeg
export LD_LIBRARY_PATH="$ABC_FFMPEG_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Prepend: /usr/bin/ffmpeg (RHEL) only has the openh264 decoder, which fails here.
export PATH="$ABC_FFMPEG_PREFIX/bin:$PATH"

# Data, checkpoints and package caches run to tens of GB: keep them on $SCRATCH.
export ABC_CACHE=/scratch/11138/pranavbelligundu/abc_cache
export UV_CACHE_DIR=/scratch/11138/pranavbelligundu/.uv_cache
export WARP_CACHE_PATH=/scratch/11138/pranavbelligundu/.warp_cache
export HF_HOME=/scratch/11138/pranavbelligundu/.hf_cache
# Hugging Face token (gated DINOv3 / XDOF/ABC-130k). Kept in a private file outside
# the repo (chmod 600, never committed); loaded here so it does not depend on how the
# terminal was started (IDE / tmux shells may not re-read ~/.bashrc).
[ -f "$HOME/.hf_token" ] && source "$HOME/.hf_token"

# $WORK and $SCRATCH are different filesystems, so uv cannot hardlink.
export UV_LINK_MODE=copy

# Login nodes cap per-user threads; uv sizes its thread pool to the core count and
# panics ("failed to initialize global rayon pool ... WouldBlock"). Cap it there only.
case "$(hostname -s)" in
  login*) export RAYON_NUM_THREADS=4 UV_CONCURRENT_INSTALLS=4 UV_CONCURRENT_DOWNLOADS=8 UV_CONCURRENT_BUILDS=2 ;;
esac
