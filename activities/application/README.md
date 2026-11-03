# Lecture 2 applications — HSF-India at IIT Mandi

A small, local-runtime adaptation of the [ML4HEP-TIFR NSBI tutorial](https://github.com/iris-hep/nsbi-lhc-toolkit/tree/b1d3cb8c704ee7a54c99367cf6578e5244ac2671/workshops/ml4hep_tifr_colab) by Jay Sandesara and contributors.
The original exercise prompts, physics distributions and worked likelihood implementations are retained.
The companion [solutions](https://github.com/rafaellopesdesa/ML_Mandi_solutions/tree/main/activities/application) add discussion guidance to the existing challenges.

## Notebook order

| Order | Notebook | Colab |
| --- | --- | --- |
| 1 | [Summary statistics and likelihood fits](Exercise_1_summary_statistics.ipynb) | [Open in Colab](https://colab.research.google.com/github/rafaellopesdesa/ML_Mandi/blob/main/activities/application/Exercise_1_summary_statistics.ipynb) |
| 2.1 | [Visualise the data](Exercise_2_1_visualize_data.ipynb) | [Open in Colab](https://colab.research.google.com/github/rafaellopesdesa/ML_Mandi/blob/main/activities/application/Exercise_2_1_visualize_data.ipynb) |
| 2.2a | [Signal vs reference training](Exercise_2_2a_SigvsRef_training.ipynb) | [Open in Colab](https://colab.research.google.com/github/rafaellopesdesa/ML_Mandi/blob/main/activities/application/Exercise_2_2a_SigvsRef_training.ipynb) |
| 2.2b | [Background vs reference training](Exercise_2_2b_BkgvsRef_training.ipynb) | [Open in Colab](https://colab.research.google.com/github/rafaellopesdesa/ML_Mandi/blob/main/activities/application/Exercise_2_2b_BkgvsRef_training.ipynb) |
| 2.3 | [Parameter fitting](Exercise_2_3_parameter_fitting.ipynb) | [Open in Colab](https://colab.research.google.com/github/rafaellopesdesa/ML_Mandi/blob/main/activities/application/Exercise_2_3_parameter_fitting.ipynb) |

## Run

Open a notebook, select a **CPU runtime**, and run its setup cell first. It installs the dependencies, downloads only the required toolkit source files from a fixed upstream commit, and generates the samples on the runtime's local disk. No external event dataset or account-mounted storage is used.

For local Jupyter, clone this repository and open a notebook in this directory. The same setup cell works there. Python 3.12 is used by the automated check; setup needs internet access for dependencies and toolkit source.

Quick-run settings:

- 20,000 simulated events per process, with seed 42.
- Equal train/inference halves, with split seed 0 and weights rescaled to preserve each process's expected yield.
- At most 20,000 classifier rows in Exercise 1 and 30,000 in 2.2a/2.2b.
- One network with two 64-neuron hidden layers, batches of 512, and at most 20 epochs.
- 31 points per likelihood scan and 20 bins for training diagnostics.

The physics yields remain 1,000,000 background and 1,100 signal events. Those are **weighted expected yields**, not the number of simulated events. With this small Monte Carlo sample, fitted minima and closure plots will be noisy; these are teaching settings, not precision-analysis settings. Bins with zero expected events are omitted from the binned workspace to avoid undefined logarithms. The learned-summary histogram uses background quantiles to keep its bins populated with the small simulation sample.

## Models across notebooks

Separate Colab notebooks can have separate runtimes. Files on a runtime's local disk disappear when that runtime is deleted.

Exercise 2.3 is self-contained: it reuses matching local models or runs the small trainings from 2.2a and 2.2b automatically. The automatic training executes the training cells in those notebooks, so their settings have a single source.

To carry **your own hyperparameter changes** into a new runtime:

1. Run 2.2a and 2.2b. Their last code cells create `models_SigvsRef.zip` and `models_BkgvsRef.zip`.
2. Download the ZIPs from the Files sidebar. The final cells also contain an optional Colab download command.
3. In 2.3, upload both ZIPs with the Files sidebar **before** running the cells. In local Jupyter, put them in this directory.

Only models with a matching data signature are reused; otherwise the quick defaults are trained locally. The optional ZIPs contain model weights, scalers and a data signature, not the event samples. Only load model files you produced yourself.

## Event generation

`generate_events.py` only generates `dataframes/background.parquet` and `dataframes/signal.parquet`. Each file contains the five truth features, five reconstructed features and event weights. It has no plotting, extra pseudo-data, fold generation or unused classifiers.

After installing dependencies, it can also be run directly:

```bash
python generate_events.py --n_bkg 20000 --n_sig 20000 --seed 42
```

The notebooks' setup manages generation automatically. To use a different sample size, change `N_BKG` and `N_SIG` consistently in all notebooks and rerun setup and training. The data signature prevents older models being silently reused. Keep the train/inference split identical across 2.2a, 2.2b and 2.3.

`utils.py` contains only the shared mixture definitions, smearing parameters, analytic density helper and deterministic split. The upstream toolkit is pinned to commit `b1d3cb8c704ee7a54c99367cf6578e5244ac2671`.

## Checks and license

The GitHub Actions check runs all five notebooks on CPU, checks finite scan values, generated data and split yields, and runs 2.3 again in a fresh directory to exercise its automatic model training.
The notebooks are committed without execution outputs.

The adapted upstream material is distributed under the [MIT license](LICENSE.txt), retaining Jay Sandesara's copyright notice.
