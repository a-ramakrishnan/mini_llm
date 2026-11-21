# Checkpoints: a folder with the weights and model_config.json (the GPTConfig under "model", plus how the model was
# trained). The weights are one of:
#   model.safetensors       what the Hugging Face Hub repos hold: tensors only, loading it cannot run code
#   best_model_params.pt    a PyTorch state dict: what training writes
#
# Weights are not in git. Models are looked up by name under the weights folder: ./weights by default, or
# $MINI_LLM_WEIGHTS. scripts/download_weights.py fills it from the Hub.

import json
import os
from dataclasses import asdict, fields

import torch

from .model import GPT, GPTConfig

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_FILE = "best_model_params.pt"
SAFETENSORS_FILE = "model.safetensors"
TIED = "lm_head.weight"     # the output layer is the token embedding matrix; safetensors files hold it once
CONFIG_FILE = "model_config.json"

# name -> what it is, and its id in the source project's model library
MODELS = {
    "mini-llm-base": {
        "hub": "aramkris/mini-llm-base",
        "source_id": "mini-fineweb-edu-ful-202611151103-bb160c", "chat": False, "stage": "stage 2",
        "about": "text completion: pretrained on 2.5B tokens of Mini FineWeb-Edu + Cosmopedia"},
    "mini-llm-chat-sft": {
        "hub": "aramkris/mini-llm-chat-sft",
        "source_id": "chat-mini-fineweb-ed-202611151211-947df2", "chat": True, "stage": "SFT",
        "about": "the base, instruction-tuned on smol-smoltalk"},
    "mini-llm-chat": {
        "hub": "aramkris/mini-llm-chat",
        "source_id": "chat-mini-fineweb-ed-202611152150-2746ba", "chat": True, "stage": "DPO",
        "about": "the instruction-tuned model after DPO against repetition: the final chat model"},
}
DEFAULT_CHAT = "mini-llm-chat"
DEFAULT_BASE = "mini-llm-base"


def weights_root():
    return os.environ.get("MINI_LLM_WEIGHTS") or os.path.join(REPO_DIR, "weights")


def weights_file(folder):
    """The weights in a checkpoint folder: model.safetensors if it is there, else best_model_params.pt, else None."""
    for name in (SAFETENSORS_FILE, WEIGHTS_FILE):
        if os.path.isfile(os.path.join(folder, name)):
            return os.path.join(folder, name)
    return None


def resolve(name_or_path):
    """The checkpoint folder for a model name (under the weights folder) or a path to a folder."""
    for folder in (name_or_path, os.path.join(weights_root(), name_or_path)):
        if weights_file(folder) and os.path.isfile(os.path.join(folder, CONFIG_FILE)):
            return folder
    hint = f" Get it with: python3 scripts/download_weights.py {name_or_path}" if name_or_path in MODELS else ""
    raise FileNotFoundError(
        f"No checkpoint for '{name_or_path}': expected {SAFETENSORS_FILE} or {WEIGHTS_FILE}, and {CONFIG_FILE}, in "
        f"that folder or in {os.path.join(weights_root(), name_or_path)}.{hint}")


def read_state(path):
    """The state dict in a weights file. A safetensors file holds the tied matrix once; it is tied again here."""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(path, device="cpu")
        state.setdefault(TIED, state["transformer.wte.weight"])
        return state
    return torch.load(path, map_location="cpu")


def write_safetensors(state, path, metadata=None):
    """Write a state dict as safetensors, without the output layer's copy of the tied embedding matrix."""
    from safetensors.torch import save_file
    tensors = {k: v.detach().cpu().contiguous() for k, v in state.items() if k != TIED}
    save_file(tensors, path, metadata={"format": "pt", **(metadata or {})})


def download(name, revision=None):
    """Fetch a named model from the Hugging Face Hub into the weights folder. Returns the folder."""
    from huggingface_hub import snapshot_download
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Known: {', '.join(MODELS)}")
    folder = os.path.join(weights_root(), name)
    if os.path.islink(folder):      # a link into a local library: leave that alone
        raise FileExistsError(f"{folder} is a link to a local checkpoint; remove it first to download instead")
    snapshot_download(MODELS[name]["hub"], local_dir=folder, revision=revision,
                      allow_patterns=[SAFETENSORS_FILE, CONFIG_FILE, "README.md"])
    return folder


def available():
    """Names of the checkpoints present in the weights folder."""
    root = weights_root()
    if not os.path.isdir(root):
        return []
    return sorted(d for d in os.listdir(root) if weights_file(os.path.join(root, d)))


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
    state = read_state(weights_file(folder))
    model.load_state_dict(state)  # strict: a missing or unexpected key is an error
    name = os.path.basename(os.path.normpath(folder))
    info = {**info, "name": name, "folder": folder, "chat": bool(info.get("chat", MODELS.get(name, {}).get("chat")))}
    return model.to(device).eval(), info, device
