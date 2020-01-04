# easy-VQA-keras

A Keras implementation of a simple VQA architecture, using the easy-VQA dataset.
Methodolgy described in this [blog post](https://victorzhou.com/blog/easy-vqa).

## Usage

### Setup and Basic Usage

First, clone the repo and install the dependencies:

```shell
git clone https://github.com/vzhou842/easy-VQA-keras.git
cd easy-VQA-keras
pip install -r requirements.txt
```

To run the model,

```shell
python train.py
```

### Training

The training script `train.py` has two optional arguments:

```shell
python train.py [--big-model] [--use-data-dir]

Optional arguments:
  --big-model     Use the bigger model with more conv layers
  --use-data-dir  Use custom data directory, at /data
```

As described, the `--big-model` flag trains a slightly larger model, that we
used to train a 99.5% accuracy model used in the following [live demo](https://easy-vqa-demo.victorzhou.com/).

Furthermore, instead of using the default easy-VQA dataset provided by the PyPi
package, you can use your own custom-generated dataset using [this repo](https://github.com/vzhou842/easy-VQA).
After following the instructions in that repo, just copy the `/data` folder into
the root directory of this repository, so that your files look like this:

```shell
easy-VQA-keras/
├── data/
  ├── answers.txt
  ├── test/
  ├── train/
├── analyze.py
├── model.py
├── prepare_data.py
└── train.py
```

For the 99.5% accuracy model, we personally used a custom dataset generated with double the images/questions
as the official dataset (set `NUM_TRAIN` and `NUM_TEST` to 8000 and 2000,
respectively, for the `easy-VQA` repo).

### Other Files

In addition to the training script, we have three other files:
- `analyze.py`, a script we used to debug our models. Run using a model weights
  file, and produce statistics about model outputs and confusion matrices to
  analyze model errors.
- `model.py`, where the model architecture is specified
- `prepare_data.py`, which reads and processes the data, either using the
  `easy-VQA` PyPi package or a custom data directory

## Results

TODO: when finalize parameters and details about the 99.5% accuracy model
