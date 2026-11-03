# HSF-India lectures at IIT Mandi

Lecture slides and hands-on machine-learning activities for **Rafael Coelho Lopes de Sa's HSF-India lectures at IIT Mandi**.

## Lectures

| Lecture | Topics | Materials |
| --- | --- | --- |
| 1. Neural Networks and Deep Learning | Computational graphs, automatic differentiation, gradient descent, regression, classification, and likelihood-ratio estimation | [Slides](lectures/Lecture1_Neural_Networks.pptx) · [Teaching script](lectures/Lecture1_Neural_Networks_Teaching_Script.docx) |
| 2. Applications in Particle Physics | Simulation and the particle-physics likelihood, histogram-based inference, neural simulation-based inference, sufficient summaries, and likelihood ratios | [Slides](lectures/Lecture2_Applications.pptx) |

Lecture 2 currently contains the introductory 29-slide section connecting the likelihood-ratio result from Lecture 1 to particle-physics inference.

## Hands-on exercises

The Lecture 1 activities have matching JAX and PyTorch versions. They build neural networks from scratch using arrays/tensors and automatic differentiation.

| Activity | JAX | PyTorch |
| --- | --- | --- |
| Automatic differentiation, gradient descent, and regression networks | [Notebook](activities/Lecture1-NN-JAX.ipynb) · [Colab](https://colab.research.google.com/github/rafaellopesdesa/ML_Mandi/blob/main/activities/Lecture1-NN-JAX.ipynb) | [Notebook](activities/Lecture1_NN_torch.ipynb) · [Colab](https://colab.research.google.com/github/rafaellopesdesa/ML_Mandi/blob/main/activities/Lecture1_NN_torch.ipynb) |
| Binary classification and likelihood ratios | [Notebook](activities/Lecture1_Classification_JAX.ipynb) · [Colab](https://colab.research.google.com/github/rafaellopesdesa/ML_Mandi/blob/main/activities/Lecture1_Classification_JAX.ipynb) | [Notebook](activities/Lecture1_Classification_torch.ipynb) · [Colab](https://colab.research.google.com/github/rafaellopesdesa/ML_Mandi/blob/main/activities/Lecture1_Classification_torch.ipynb) |

The classification activity uses 2,000 events drawn from two overlapping two-dimensional Gaussian distributions with displaced means and unit covariance. A fixed seed gives the same dataset in the exercise and solution notebooks.

### Running in Google Colab

Open a Colab link and run the setup cell first. A CPU runtime is sufficient for these four notebooks, and the datasets are generated in the notebooks without external data downloads or a Google Drive mount.

Run the cells in order and complete the marked exercises. The student placeholders intentionally raise `NotImplementedError` until implemented. Completed versions are available in the [companion solutions repository](https://github.com/rafaellopesdesa/ML_Mandi_solutions).

## Lecture 2 application exercises

Five [application notebooks](activities/application/README.md) follow the original NSBI tutorial order: summary statistics, data visualisation, signal/reference training, background/reference training, and parameter fitting. They use small fixed-seed samples, CPU training and local runtime storage, with an Open in Colab link at the top of each notebook. The final fit can regenerate its models in a fresh runtime.

## Acknowledgments

The material builds on Michelle Kuchera's HSF-India lectures at IIT Madras and the simulation-based inference lectures given by Jay Sandesara and Rafael Coelho Lopes de Sa at TIFR. Credits for reused figures and diagrams are retained in the lecture materials.

## License

See [LICENSE](LICENSE).
