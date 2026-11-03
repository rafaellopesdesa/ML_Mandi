"""Local setup and model handoff for the five application notebooks."""

import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from urllib.request import urlretrieve
import zipfile

UPSTREAM_COMMIT = "b1d3cb8c704ee7a54c99367cf6578e5244ac2671"
PACKAGE_FILES = ["nsbi_common_utils/__init__.py","nsbi_common_utils/calibration.py","nsbi_common_utils/inference.py","nsbi_common_utils/lightning_tools/__init__.py","nsbi_common_utils/lightning_tools/callbacks.py","nsbi_common_utils/lightning_tools/datasets.py","nsbi_common_utils/lightning_tools/density_ratio_model.py","nsbi_common_utils/lightning_tools/multiclass_model.py","nsbi_common_utils/models/__init__.py","nsbi_common_utils/models/sbi_parametric_model.py","nsbi_common_utils/plotting.py","nsbi_common_utils/training/__init__.py","nsbi_common_utils/training/neural_ratio_estimation.py","nsbi_common_utils/training/preselection_training.py","nsbi_common_utils/training/utils.py"]
TRAINING_NOTEBOOKS = {
    "models_SigvsRef": "Exercise_2_2a_SigvsRef_training.ipynb",
    "models_BkgvsRef": "Exercise_2_2b_BkgvsRef_training.ipynb",
}


def download_if_missing(url, destination):
    destination = Path(destination)
    if not destination.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".download")
        urlretrieve(url, temporary)
        temporary.replace(destination)


def seed_everything(seed=42):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)


def setup(workdir, repository, ref="main", n_bkg=20_000, n_sig=20_000, seed=42):
    """Install dependencies, load the pinned toolkit, and generate local data."""
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    base = f"https://raw.githubusercontent.com/{repository}/{ref}/activities/application"
    for name in ["requirements.txt", "utils.py", "generate_events.py", *TRAINING_NOTEBOOKS.values()]:
        download_if_missing(f"{base}/{name}", workdir / name)
    # These small exercises use CPU, including ONNX export and JAX fitting.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(workdir / "requirements.txt")],
        check=True,
    )
    source = workdir / ".toolkit" / UPSTREAM_COMMIT / "src"
    upstream = f"https://raw.githubusercontent.com/iris-hep/nsbi-lhc-toolkit/{UPSTREAM_COMMIT}"
    for name in PACKAGE_FILES:
        download_if_missing(f"{upstream}/src/{name}", source / name)
    download_if_missing(f"{upstream}/LICENSE.txt", source.parent / "LICENSE.txt")
    for path in [str(source), str(workdir)]:
        if path not in sys.path:
            sys.path.insert(0, path)
    os.chdir(workdir)
    seed_everything(seed)
    import jax
    jax.config.update("jax_enable_x64", True)

    from generate_events import generate_events
    config = {
        "n_bkg": n_bkg, "n_sig": n_sig, "seed": seed,
        "generator": hashlib.sha256((workdir / "generate_events.py").read_bytes()).hexdigest(),
        "distributions": hashlib.sha256((workdir / "utils.py").read_bytes()).hexdigest(),
    }
    manifest = Path("dataframes/generation.json")
    old_config = json.loads(manifest.read_text()) if manifest.is_file() else None
    if old_config != config or not all(Path(f"dataframes/{s}.parquet").is_file() for s in ["signal", "background"]):
        generate_events(n_bkg, n_sig, seed)
        manifest.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Working directory: {workdir}")
    print(f"Local samples: {n_sig:,} signal + {n_bkg:,} background events; seed {seed}.")


def data_signature():
    digest = hashlib.sha256()
    for name in ["background", "signal"]:
        digest.update(Path(f"dataframes/{name}.parquet").read_bytes())
    return digest.hexdigest()


def record_training(model_dir):
    """Record which dataset produced a model; prevent silently reusing stale models."""
    Path(model_dir, "training_data.sha256").write_text(data_signature() + "\n")


def export_models(model_dir):
    """Create a small ZIP for optional transfer between local notebook runtimes."""
    archive = Path(model_dir + ".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        for path in sorted(Path(model_dir).iterdir()):
            if path.is_file():
                output.write(path, arcname=path.as_posix())
    print(f"Optional handoff file: {archive.resolve()}")
    return archive


def ensure_trained_models():
    """Reuse local models, import uploaded ZIPs, or run the two small trainings."""
    signature = data_signature()
    for model_dir, notebook_name in TRAINING_NOTEBOOKS.items():
        required = [Path(model_dir, name) for name in ["model0.onnx", "model_scaler0.bin", "training_data.sha256"]]
        if all(path.is_file() for path in required) and required[-1].read_text().strip() == signature:
            print(f"Using local model: {model_dir}")
            continue
        # In Colab, files uploaded with the Files sidebar initially land in /content.
        for archive in dict.fromkeys([Path(model_dir + ".zip"), Path("/content") / (model_dir + ".zip")]):
            if archive.is_file():
                with zipfile.ZipFile(archive) as bundle:
                    for member in bundle.infolist():
                        path = Path(member.filename)
                        if len(path.parts) != 2 or path.parts[0] != model_dir or ".." in path.parts:
                            raise ValueError(f"Unexpected archive entry: {member.filename}")
                    bundle.extractall(".")
                break
        required = [Path(model_dir, name) for name in ["model0.onnx", "model_scaler0.bin", "training_data.sha256"]]
        if all(path.is_file() for path in required) and required[-1].read_text().strip() == signature:
            print(f"Using local model: {model_dir}")
            continue
        print(f"Training {model_dir} locally from {notebook_name}.")
        seed_everything(42)
        notebook = json.loads(Path(notebook_name).read_text())
        namespace = {"__name__": "__main__"}
        trained = False
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code" or "application-setup" in cell.get("metadata", {}).get("tags", []):
                continue
            code = "".join(cell["source"])
            exec(compile(code, f"{notebook_name}:cell-{index}", "exec"), namespace)
            if ".train(" in code:
                trained = True
                break
        if not trained or not all(path.is_file() for path in required):
            raise RuntimeError(f"Training did not produce the required files in {model_dir}.")
        import matplotlib.pyplot as plt
        plt.close("all")
