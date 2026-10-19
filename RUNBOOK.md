# RUNBOOK — running `conditional-nf-integration` on the local Windows box

*Every command in this file was run on this machine and works as written.
Shell is Git Bash; Python paths are written Windows-style because they are
passed as arguments to `python.exe`.*

This supersedes the Linux/EOS procedure in `PRD.md` Part II for work done
locally. The physics is unchanged; what changed is **where MadGraph is built
and how its Python wrapper is compiled**.

---

## 1. Machine layout

| what | where |
|---|---|
| repository | `C:\Users\laiji\test\conditional-nf-integration` |
| MadGraph 5 3.6.7 | `C:\Users\laiji\test\MG5_aMC_v3_6_7` |
| MG5 launcher wrapper | `C:\Users\laiji\test\bin\mg5_aMC.bat` |
| generated standalones | `C:\Users\laiji\test\conditional-nf-integration\generated\` |
| Python | conda env `workspace`, Python 3.10.14, torch 2.3.0+cu121 (see `RUNBOOK_GPU.md` §2) |
| Fortran/compilers | MSYS2 MinGW-w64 at `C:\msys64\mingw64\bin` (gcc/gfortran 14.1.0) |
| scratch logs | `C:\Users\laiji\test\mg5_scratch\` |

Set these once per shell:

```bash
export REPO="C:/Users/laiji/test/conditional-nf-integration"
export PY="/c/Users/laiji/.conda/envs/workspace/python.exe"
export MG5="C:/Users/laiji/test/bin/mg5_aMC.bat"
```

---

## 2. One-time setup

Already done on this machine — recorded so it can be rebuilt.

### 2.1 Python environment

```bash
conda activate workspace
pip install "numpy<2" pytest
pip install --only-binary :all: vegas
```

`numpy<2` is required **only** because this env has torch 2.3.0, which is
compiled against NumPy 1.x.  (The Linux notes correctly point out that with
torch 2.14 the pin is unnecessary.)

### 2.2 MinGW Fortran toolchain

```bash
# from a MSYS2 shell
pacman -S --needed mingw-w64-x86_64-gcc-fortran
```

### 2.3 MadGraph

```bash
cd /c/Users/laiji/test
tar xzf /c/Users/laiji/Downloads/MG5_aMC_v3.6.7.tar.gz
```

> The tarball contains three **symlinks** that Windows refuses to create
> without Developer Mode (`gpu/rambo.cc`, `pineappl_maxproc.inc`,
> `NNPDFDriver.f`). `tar` exits with status 2. All three belong to GPU/NLO/PDF
> features that this repository never touches, so the extraction is usable
> as-is.

MG5's `bin/mg5_aMC` is a POSIX script, so it is launched through
`C:\Users\laiji\test\bin\mg5_aMC.bat`, which also puts MSYS2 on `PATH` (MG5
hard-requires `gzip` at startup) and uses conda's Python 3.10 (MG5 3.6.7 has
dropped ≤3.9).

---

## 3. Patches applied — and why

Four changes were needed. Two are in the repository, two are in the local
MadGraph install.

| # | file | change |
|---|---|---|
| 1 | `MG5_aMC_v3_6_7/madgraph/various/misc.py` | `which()` now tries `PATHEXT` suffixes |
| 2 | `MG5_aMC_v3_6_7/input/mg5_configuration.txt` | `auto_update = 0`, `automatic_html_opening = False` |
| 3 | `madgraph_process_integrand.py` | loader globs `matrix2py*.pyd` as well as `*.so` |
| 4 | `experiments/setup_madgraph_tth_benchmark.py` | `compile_libraries()` delegates to the Windows builder |
| — | `madgraph_windows_build.py` | **new** — builds the f2py wrapper with MinGW |

**(1) MG5's `which()` never appends `.exe`.** It joins the bare program name
onto each `PATH` entry and tests `os.access(..., X_OK)`, which is POSIX-only.
On Windows `which('gzip')` returns `None`, and MG5 then dies *before it starts*
with:

```
TypeError: argument of type 'NoneType' is not iterable
  File ".../madgraph/various/misc.py", line 1302, in configure_gzip
```

The patch returns the bare name first (so POSIX behaviour is untouched) and
then the `PATHEXT` variants.

**(2) The updater uses `signal.SIGALRM`**, which does not exist on Windows, so
MG5 crashed on exit. `auto_update = 0` disables the check.

**(3) The loader globbed only `P*/matrix2py*.so`.** MinGW/f2py produces
`matrix2py.cp310-win_amd64.pyd`. Both extensions are now accepted.

**(4) `compile_libraries()` shells out to MG5's makefiles**, which are
POSIX-shaped (`SHELL = /bin/bash`, `uname -s`, `-Wl,-rpath`, `.so` suffixes).
On Windows they build no importable wrapper at all. See §4.

### What `madgraph_windows_build.py` does

It replaces *only the final compile step*. MG5's own make still builds
`lib/libdhelas.a` and `lib/libmodel.a` during generation, and those are reused
verbatim. For each `SubProcesses/P*/` holding an `f2py_matrix_wrapper.f` it:

1. copies `coupl.inc` next to `matrix.f` — MG5 tries to *symlink* it there and
   the link silently fails on Windows;
2. runs f2py over `f2py_matrix_wrapper.f` + `matrix.f`, linking the two
   archives.

Two flags in that call are load-bearing:

* `--compiler=mingw32 --fcompiler=gnu95` — without these f2py assumes MSVC.
  With them it also generates the `libpython310.a` import library it needs
  from `python310.dll`, which is what lets a **MinGW-built** extension be
  imported by conda's **MSVC-built** CPython.
* `LDFLAGS=-static` — otherwise the extension depends on `libgfortran-5.dll`,
  `libgcc_s_seh-1.dll`, `libquadmath-0.dll` and `libwinpthread-1.dll` from the
  MSYS2 tree. CPython 3.8+ does **not** search `PATH` for the dependencies of
  extension modules, so the import fails with
  `DLL load failed while importing matrix2py`. Statically linked, the module
  depends only on `KERNEL32.dll`, `msvcrt.dll` and `python310.dll`.

---

## 4. Build a process

```bash
cd "$REPO"
export PYTHONPATH="$REPO"

$PY experiments/setup_madgraph_process.py \
    --config configs/processes/gg_ttbar.json \
    --mg5 "$MG5"
```

This generates the standalone **and** compiles the wrapper. Repeat with
`gg_tth.json` and `uu_ww_4l.json`. The output directory must not already
exist — delete `generated/<name>_standalone` to rebuild.

To rebuild only the wrapper of an existing standalone:

```bash
$PY -m madgraph_windows_build --standalone generated/gg_ttbar_standalone
```

---

## 5. Run the pipeline

> ### ⚠ The working-directory rule (this is the one that bites)
>
> **Every MadGraph-dependent script must run with the CWD inside the
> standalone directory**, and every other path must be absolute.
>
> `Source/MODEL/lha_read.f` resolves its auxiliary Les Houches files relative to
> the *current working directory*, walking up five parent levels looking for
> `Cards/ident_card.dat`. If it never finds one it executes a Fortran `stop`,
> **which exits with status 0**. You get no traceback, no error, and no output
> from any print statement after the failing call — a run that looks like it
> succeeded did nothing.
>
> This contradicts the repository's own advice ("always run from the repo
> root"), which is correct for the synthetic drivers and **wrong for every
> MadGraph one**.

### 5.1 Check the integrand (seconds)

```bash
cd "$REPO/generated/gg_ttbar_standalone"
PYTHONPATH="$REPO" $PY "$REPO/experiments/validate_madgraph_process_integrand.py" \
    --config "$REPO/configs/processes/gg_ttbar.json" --points 500
```

### 5.2 Train

```bash
cd "$REPO/generated/gg_ttbar_standalone"
PYTHONPATH="$REPO" $PY -u \
    "$REPO/experiments/run_madgraph_process_conditional_copula.py" \
    --config "$REPO/configs/processes/gg_ttbar.json" \
    --speed-profile smoke \
    --no-compile-kernels \
    --output "$REPO/results/gg_ttbar_copula_smoke"
```

* Use `$PY -u`. Redirected stdout is block-buffered, so without `-u` the log
  file stays empty for the whole run and progress is invisible.
* `--no-compile-kernels` is right on this box: the kernel-compile path is
  CUDA-only and there is no GPU.
* `--speed-profile` is the compute dial — `smoke` 300 KL steps, `fast` 6 000,
  `balanced` 12 000, `thorough` 30 000 (the default). Always start with
  `smoke`.

### 5.3 Held-out validation

```bash
cd "$REPO/generated/gg_ttbar_standalone"
PYTHONPATH="$REPO" $PY -u \
    "$REPO/experiments/validate_madgraph_process_integral.py" \
    --config "$REPO/configs/processes/gg_ttbar.json" \
    --checkpoint "$REPO/results/gg_ttbar_copula_smoke/madgraph_gg_ttbar_2_final_model.pt" \
    --output "$REPO/results/gg_ttbar_validation_smoke"
```

### 5.4 Scripted, all three processes

```bash
bash scripts/run_all_windows.sh          # see that file for the full sweep
```

---

## 6. Fast health checks

```bash
cd "$REPO"
PYTHONPATH="$PWD" $PY -m pytest tests/ -q          # 12 passed, ~4 s
PYTHONPATH="$PWD" $PY experiments/run_copula_bin_sweep.py --smoke   # ~1 min
```

`pytest` must not be run from a directory containing a stray `py.py`. MG5's
ALOHA code generator writes one into the CWD; Python puts the CWD first on
`sys.path`, so it shadows the `py` shim pytest ships and every run dies with
`AttributeError: module 'py' has no attribute 'path'`. `setup_madgraph_process.py`
now runs MG5 from a throwaway directory so this cannot happen, but if you invoke
`bin/mg5_aMC` by hand, delete the `py.py` (and `nsqso_born.inc`) it leaves behind.

---

## 7. Results

See `RESULTS.md` for the tables produced by this session, and
`results/` for the raw CSVs and checkpoints.

---

## 8. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `TypeError: argument of type 'NoneType' is not iterable` in `configure_gzip` | MG5 can't find `gzip` | patch 1 in §3, or put a directory containing `gzip.exe` on `PATH` |
| Script exits 0 but prints nothing | CWD is not inside the standalone dir, so `lha_read` hit its Fortran `stop` | `cd generated/<proc>_standalone` first |
| `ImportError: DLL load failed while importing matrix2py` | wrapper linked against MSYS2 runtime DLLs | rebuild with `LDFLAGS=-static` (the builder does this) |
| `AttributeError: module 'py' has no attribute 'path'` | stray `py.py` in the CWD | delete `py.py` |
| `FileNotFoundError: No matrix2py library below ...` | wrapper not built, or loader glob does not match the extension | run `python -m madgraph_windows_build --standalone <dir>` |
| `RuntimeError: ... resolve to exactly one subprocess` | the MG5 process string lacks the `@1` diagram filter | keep `@1` in `configs/processes/*.json` |
