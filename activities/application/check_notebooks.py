"""Execute the five notebooks and a fresh-runtime fit on CPU.

Run: python -m pip install nbclient nbformat ipykernel
     python activities/application/check_notebooks.py
All generated files stay in temporary directories.
"""

import json
from pathlib import Path
import shutil
import tempfile
import time

import nbformat
from nbclient import NotebookClient

SOURCE = Path(__file__).resolve().parent
NOTEBOOKS = ["Exercise_1_summary_statistics.ipynb","Exercise_2_1_visualize_data.ipynb","Exercise_2_2a_SigvsRef_training.ipynb","Exercise_2_2b_BkgvsRef_training.ipynb","Exercise_2_3_parameter_fitting.ipynb"]
CHECKS = r"""
import numpy as np
import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory
from generate_events import generate_events
from utils import split_train_inference

for process, expected in [("background", 1_000_000.0), ("signal", 1_100.0)]:
    frame = pd.read_parquet(f"dataframes/{process}.parquet")
    assert len(frame) == 20_000
    assert set(frame.columns) == {*(f"x{i}" for i in range(1, 6)), *(f"z{i}" for i in range(1, 6)), "weight"}
    assert np.isfinite(frame.to_numpy()).all()
    assert np.isclose(frame.weight.sum(), expected)
    train, inference = split_train_inference(frame.assign(event_id=np.arange(len(frame))))
    assert len(train) == len(inference) == 10_000
    assert set(train.event_id).isdisjoint(inference.event_id)
    assert np.isclose(train.weight.sum(), expected)
    assert np.isclose(inference.weight.sum(), expected)

for key, value in list(globals().items()):
    if key.startswith(("NLL_value", "tmu_")):
        assert np.isfinite(np.asarray(value)).all(), (key, np.asarray(value))

if "nll_manual" in globals():
    assert np.isfinite(float(nll_manual(np.array([1.0], dtype=np.float64))))
if "model_nsbi" in globals():
    # Manual and workspace NLLs can differ by additive constants.
    manual_delta = float(nll_manual(np.array([1.2], dtype=np.float64)) - nll_manual(np.array([1.0], dtype=np.float64)))
    workspace_delta = float(model_nsbi.model(np.array([1.2])) - model_nsbi.model(np.array([1.0])))
    assert np.isclose(manual_delta, workspace_delta, atol=1e-5, rtol=1e-5), (manual_delta, workspace_delta)
    from setup_local import data_signature
    for model_dir in ["models_SigvsRef", "models_BkgvsRef"]:
        assert Path(model_dir, "model0.onnx").is_file()
        assert Path(model_dir, "training_data.sha256").read_text().strip() == data_signature()

with TemporaryDirectory() as generated:
    generate_events(output_dir=generated)
    for process in ["background", "signal"]:
        pd.testing.assert_frame_equal(
            pd.read_parquet(Path(generated, f"{process}.parquet")),
            pd.read_parquet(f"dataframes/{process}.parquet"),
        )
"""


def copy_inputs(destination):
    for path in SOURCE.iterdir():
        if path.suffix in {".py", ".ipynb", ".txt"}:
            shutil.copy2(path, destination / path.name)


def run(name, directory):
    started = time.monotonic()
    notebook = nbformat.read(directory / name, as_version=4)
    nbformat.validate(notebook)
    notebook.cells.append(nbformat.v4.new_code_cell(CHECKS))
    NotebookClient(
        notebook, timeout=600, kernel_name="python3",
        resources={"metadata": {"path": str(directory)}},
    ).execute()
    print(f"PASS {name}: {time.monotonic() - started:.1f} s", flush=True)


def main():
    with tempfile.TemporaryDirectory(prefix="mandi-sequence-") as temporary:
        directory = Path(temporary)
        copy_inputs(directory)
        for name in NOTEBOOKS:
            run(name, directory)
    with tempfile.TemporaryDirectory(prefix="mandi-fresh-fit-") as temporary:
        directory = Path(temporary)
        copy_inputs(directory)
        run(NOTEBOOKS[-1], directory)
    print("PASS sequence and fresh-runtime parameter fitting", flush=True)


if __name__ == "__main__":
    main()
