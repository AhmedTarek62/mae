import torch
from pathlib import Path
from torch import profiler

# ────────────────────────── 1.  Set‑up ──────────────────────────
from models_vit import vit_small_patch16  # ← your model factory

CKPT_PATH = Path("checkpoints/pretrained_all_data.pth")  # ← edit if needed
IN_CH = 1  # input channels (spectrogram/CSI)
IMG_SIZE = 224  # input resolution 224×224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype_bytes = {torch.float32: 4, torch.float16: 2}  # helper

# ────────────────────────── 2.  Build & load model ──────────────
ckpt = torch.load(CKPT_PATH, map_location="cpu")["model"]
model = vit_small_patch16(global_pool="token", num_classes=10, in_chans=IN_CH)
msg = model.load_state_dict(ckpt, strict=False)
print("Load‑status:", msg)

# ────────────────────────── 3.  Parameter & static size ─────────
param_cnt = sum(p.numel() for p in model.parameters())
sizes = {
    "FP32": param_cnt * dtype_bytes[torch.float32] / 1024 ** 2,  # MB
    "FP16": param_cnt * dtype_bytes[torch.float16] / 1024 ** 2,
    "INT8": param_cnt * 1 / 1024 ** 2
}
print(f"\nParameters : {param_cnt / 1e6:.1f} M")
print(f"Model size : {sizes['FP32']:.1f} MB (FP32) | "
      f"{sizes['FP16']:.1f} MB (FP16) | {sizes['INT8']:.1f} MB (INT8)")

# ────────────────────────── 4.  Activation memory (GPU only) ────
model.to(device).eval()
with torch.no_grad():
    dummy_gpu = torch.randn(1, IN_CH, IMG_SIZE, IMG_SIZE, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        model(dummy_gpu)
        act_mem = torch.cuda.max_memory_allocated(device) / 1024 ** 2  # MB
        print(f"Peak activation memory : {act_mem:.1f} MB (on {device})")

# ────────────────────────── 5.  Empirical FLOPs & CPU memory ────
activities = [profiler.ProfilerActivity.CPU]
if device.type == "cuda":
    activities.append(profiler.ProfilerActivity.CUDA)

with profiler.profile(activities=activities,
                      profile_memory=True,
                      record_shapes=True,
                      with_flops=True) as prof:
    with torch.no_grad():
        dummy = torch.randn(1, IN_CH, IMG_SIZE, IMG_SIZE, device=device)
        model(dummy)

prog = prof.key_averages().total_average()
flops = prog.flops  # integer
cpu_mem = prog.self_cpu_memory_usage  # bytes

print(f"Empirical FLOPs / forward : {flops / 1e9:.2f} GFLOPs")
print(f"Peak CPU memory (PyTorch) : {cpu_mem / 1024 ** 2:.1f} MB")
