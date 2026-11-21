# Checkpoints: a folder with the weights (best_model_params.pt, a PyTorch state dict) and model_config.json
# (the GPTConfig under "model", plus how the model was trained).
#
# Weights are not in git. Models are looked up by name under the weights folder: ./weights by default, or
# $MINI_LLM_WEIGHTS. scripts/link_weights.py fills it from a tinystories-ARM-model library.

import json
import os
from dataclasses import asdict, fields

import torch

from .model import GPT, GPTConfig

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_FILE = "best_model_params.pt"
CONFIG_FILE = "model_config.json"

# name -> what it is, and its id in the source project's model library
MODELS = {
    "mini-llm-base": {
        "source_id": "mini-fineweb-edu-ful-202611151103-bb160c", "chat": False, "stage": "stage 2",
        "about": "text completion: pretrained on 2.5B tokens of Mini FineWeb-Edu + Cosmopedia"},
    "mini-llm-chat-sft": {
        "source_id": "chat-mini-fineweb-ed-202611151211-947df2", "chat": True, "stage": "SFT",
        "about": "the base, instruction-tuned on smol-smoltalk"},
    "mini-llm-chat": {
        "source_id": "chat-mini-fineweb-ed-202611152150-2746ba", "chat": True, "stage": "DPO",
        "about": "the instruction-tuned model after DPO against repetition: the final chat model"},
}
DEFAULT_CHAT = "mini-llm-chat"
DEFAULT_BASE = "mini-llm-base"


def weights_root():
    return os.environ.get("MINI_LLM_WEIGHTS") or os.path.join(REPO_DIR, "weights")


def resolve(name_or_path):
    """The checkpoint folder for a model name (under the weights folder) or a path to a folder."""
    for folder in (name_or_path, os.path.join(weights_root(), name_or_path)):
        if os.path.isfile(os.path.join(folder, WEIGHTS_FILE)) and os.path.isfile(os.path.join(folder, CONFIG_FILE)):
            return folder
    raise FileNotFoundError(
        f"No checkpoint for '{name_or_path}': expected {WEIGHTS_FILE} and {CONFIG_FILE} in that folder or in "
        f"{os.path.join(weights_root(), name_or_path)}. See the README for how to get the weights.")


def available():
    """Names of the checkpoints present in the weights folder."""
    root = weights_root()
    if not os.path.isdir(root):
        return []
    return sorted(d for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, WEIGHTS_FILE)))


def pick_device(device=None):
    """Nvidia GPU (cuda) > Apple Silicon GPU (mps) > CPU, unless one is asked for."""
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_config(folder):
    """(GPTConfig, the whole model_config.json). Fields this code doesn't know are ignored, so a config written by a
    newer version still loads."""
    with open(os.path.join(folder, CONFIG_FILE)) as f:
        info = json.load(f)
    known = {f.name for f in fields(GPTConfig)}
    return GPTConfig(**{k: v for k, v in info["model"].items() if k in known}), info


def save_config(folder, config, extra=None):
    """Write model_config.json: the model config plus any training summary in `extra`."""
    with open(os.path.join(folder, CONFIG_FILE), "w") as f:
        json.dump({"model": asdict(config), **(extra or {})}, f, indent=2)


def load(name_or_path, device=None):
    """Load a checkpoint. Returns (model in eval mode on the device, info, device). info is model_config.json plus
    "name", "folder" and "chat"."""
    folder = resolve(name_or_path)
    config, info = load_config(folder)
    device = pick_device(device)
    model = GPT(config)
    state = torch.load(os.path.join(folder, WEIGHTS_FILE), map_location="cpu")
    model.load_state_dict(state)  # strict: a missing or unexpected key is an error
    name = os.path.basename(os.path.normpath(folder))
    info = {**info, "name": name, "folder": folder, "chat": bool(info.get("chat", MODELS.get(name, {}).get("chat")))}
    return model.to(device).eval(), info, device
