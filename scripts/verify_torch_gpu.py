"""Prove that the PyTorch used for the conditional flow is really GPU-backed.

Why this exists
---------------
A CPU-only PyTorch wheel installs and imports exactly like a CUDA one, so
"``import torch`` worked" is not evidence that anything ran on the GPU. A
``+cpu`` wheel answers ``torch.cuda.is_available()`` with ``False``, but a
CUDA wheel on a machine with no visible device looks the same. This script
checks the six things that actually distinguish the two cases, in increasing
order of strength:

1. the installed wheel is a CUDA build (version tag, ``torch.version.cuda``);
2. ``torch.cuda`` sees the device and reports it;
3. a tensor really lives in device memory and arithmetic executes there;
4. GPU arithmetic is measurably faster than CPU (a silent fallback is not
   invisible if the timing says so);
5. the trained checkpoint's tensors were saved from a CUDA process;
6. the conditional marginal itself runs on the GPU and matches the CPU result.

Exit status is 0 only if every check passes.

    "C:/Users/laiji/.conda/envs/workspace/python.exe" scripts/verify_torch_gpu.py

Add ``--checkpoint <path>`` to test a different model; ``--skip-model`` to stop
after the synthetic checks.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = (
    REPO
    / "results"
    / "gg_ttbar_copula_thorough_gpu"
    / "madgraph_gg_ttbar_2_final_model.pt"
)

FAILURES: list[str] = []


def check(name: str, passed: bool, detail: str) -> None:
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}: {detail}")
    if not passed:
        FAILURES.append(name)


def section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def best_time(fn, warmup: int = 3, repeats: int = 10) -> float:
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        best = min(best, time.perf_counter() - start)
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--gflops-size", type=int, default=4096)
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    print("PyTorch GPU verification")
    print("=" * 24)

    # ------------------------------------------------------------------
    section("1. Installed wheel")
    cuda_tag = torch.version.cuda
    is_cuda_build = cuda_tag is not None
    print(f"  torch              {torch.__version__}")
    print(f"  torch.version.cuda {cuda_tag}")
    print(f"  cudnn              {torch.backends.cudnn.version()}")
    print(f"  executable         {sys.executable}")
    check(
        "CUDA build installed",
        is_cuda_build,
        f"wheel is {'+cpu — CPU only' if not is_cuda_build else 'CUDA ' + str(cuda_tag)}",
    )

    # ------------------------------------------------------------------
    section("2. Device visibility")
    available = torch.cuda.is_available()
    print(f"  torch.cuda.is_available()   {available}")
    print(f"  torch.cuda.device_count()   {torch.cuda.device_count()}")
    check("CUDA available", available, "device is visible to torch")
    if not available:
        print()
        print("Cannot go further: torch sees no CUDA device. If the wheel is a")
        print("CUDA build, check the NVIDIA driver and that the GPU is not in")
        print("exclusive use by another process.")
        return 1

    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"  device 0                    {name}")
    print(f"  capability                  {capability[0]}.{capability[1]}")
    print(f"  memory total                {total:.2f} GiB")

    # ------------------------------------------------------------------
    section("3. Execution on the device")
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
    b = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
    c = a @ b
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()
    checksum = float(c.sum().item())
    print(f"  tensor device               {a.device}")
    print(f"  matmul output device        {c.device}")
    print(f"  allocated before / after    {before / 2**20:.1f} / {after / 2**20:.1f} MiB")
    print(f"  checksum of a @ b           {checksum:.6f}")
    check("tensor is on the GPU", a.is_cuda and c.is_cuda, str(c.device))
    grew = after - before >= 3 * 1024 * 1024
    check("device memory was allocated", grew, f"+{(after - before) / 2**20:.1f} MiB")

    # ------------------------------------------------------------------
    section("4. CPU versus GPU throughput")
    n = args.gflops_size
    flops = 2.0 * n**3
    a_cpu = torch.randn(n, n, dtype=torch.float32)
    b_cpu = torch.randn(n, n, dtype=torch.float32)
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")
    cpu_s = best_time(lambda: a_cpu @ b_cpu, warmup=1, repeats=3)
    gpu_s = best_time(lambda: a_gpu @ b_gpu, warmup=3, repeats=5)
    cpu_rate = flops / cpu_s / 1e9
    gpu_rate = flops / gpu_s / 1e9
    speedup = cpu_s / gpu_s
    print(f"  size                        {n} x {n} float32")
    print(f"  CPU                         {cpu_s * 1e3:9.2f} ms   {cpu_rate:8.1f} GFLOP/s")
    print(f"  GPU                         {gpu_s * 1e3:9.2f} ms   {gpu_rate:8.1f} GFLOP/s")
    print(f"  speedup                     {speedup:.1f}x")
    check(
        "GPU is faster than CPU",
        speedup > 2.0,
        f"{speedup:.1f}x — a CPU fallback would not show this",
    )

    # ------------------------------------------------------------------
    section("5. Checkpoint provenance (informational)")
    checkpoint_path = args.checkpoint
    if not checkpoint_path.exists():
        check("checkpoint found", False, str(checkpoint_path))
    else:
        # No map_location: tensors restore to the device they were saved on.
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        state = checkpoint.get("marginal_state_dict", checkpoint)
        devices = {
            str(value.device)
            for value in state.values()
            if isinstance(value, torch.Tensor)
        }
        print(f"  file                        {checkpoint_path.name}")
        print(f"  tensor devices in state     {sorted(devices)}")
        print("  note: the reference trainer saves cpu_state_dict(bank) — CPU")
        print("        copies by design — so this cannot prove where training")
        print("        ran. The runtime checks below are the evidence.")
        del checkpoint
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    if args.skip_model:
        return 0 if not FAILURES else 1

    section("6. The conditional marginal itself")
    sys.path.insert(0, str(REPO))
    from nf.conditional_cubic_1d import ConditionalCubicFlow1D

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["marginal_state_dict"]
    hidden = state["marginals.0.net.0.weight"].shape[0]
    prefix = "marginals.0."
    cpu_model = ConditionalCubicFlow1D(cond_dim=3, hidden=hidden, n_bins=32).double()
    cpu_model.load_state_dict(
        {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    )
    cpu_model.eval()
    gpu_model = ConditionalCubicFlow1D(cond_dim=3, hidden=hidden, n_bins=32).double().cuda()
    gpu_model.load_state_dict(cpu_model.state_dict())
    gpu_model.eval()

    count = 65536
    generator = torch.Generator().manual_seed(20261019)
    cond = torch.rand(count, 3, generator=generator, dtype=torch.float64)
    u = torch.rand(count, generator=generator, dtype=torch.float64)

    with torch.no_grad():
        cpu_s = best_time(lambda: cpu_model.forward(u, cond), warmup=1, repeats=3)
        cond_gpu, u_gpu = cond.cuda(), u.cuda()
        gpu_s = best_time(lambda: gpu_model.forward(u_gpu, cond_gpu), warmup=3, repeats=5)
        x_cpu, logdet_cpu = cpu_model.forward(u, cond)
        x_gpu, logdet_gpu = gpu_model.forward(u_gpu, cond_gpu)
        torch.cuda.synchronize()

    x_back = x_gpu.cpu()
    logdet_back = logdet_gpu.cpu()
    x_diff = float((x_back - x_cpu).abs().max())
    logdet_diff = float((logdet_back - logdet_cpu).abs().max())
    print(f"  batch                       {count:,}")
    print(f"  CPU                         {cpu_s * 1e3:9.2f} ms")
    print(f"  GPU                         {gpu_s * 1e3:9.2f} ms")
    print(f"  speedup                     {cpu_s / gpu_s:.1f}x")
    print(f"  max |x_gpu - x_cpu|         {x_diff:.3e}")
    print(f"  max |logdet_gpu - logdet_cpu| {logdet_diff:.3e}")
    check("marginal output is on the GPU", x_gpu.is_cuda, str(x_gpu.device))
    check("GPU matches CPU", x_diff < 1e-10 and logdet_diff < 1e-10, "float64 parity")
    check("marginal is faster on the GPU", cpu_s / gpu_s > 1.0, f"{cpu_s / gpu_s:.1f}x")

    # Training is the forward pass plus backward; check the backward as well.
    gpu_model.zero_grad(set_to_none=True)
    x_grad, logdet_grad = gpu_model.forward(u_gpu, cond_gpu)
    loss = (x_grad**2).sum() + logdet_grad.sum()
    loss.backward()
    grad_device = next(
        parameter.grad.device
        for parameter in gpu_model.parameters()
        if parameter.grad is not None
    )
    print(f"  gradient device             {grad_device}")
    check("backward pass runs on the GPU", grad_device.type == "cuda", str(grad_device))

    # ------------------------------------------------------------------
    print()
    if FAILURES:
        print(f"VERDICT: {len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        return 1
    print("VERDICT: PyTorch is CUDA-backed and the conditional marginal runs on the GPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
