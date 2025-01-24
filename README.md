# PeaPOD

## Setup and Requirements

To create the conda environment, run:
```bash
conda env create -f environment.yaml
```
To download the data, follow the instructions in the [P5](https://github.com/jeykigung/P5), who are the original authors of the preprocessing steps.
## Running the Code
To run the training code, run
```
python scripts/train_toys.sh
python scripts/train_beauty.sh
python scripts/train_sports.sh
```

To evaluate, run
```
python scripts/evaluate.sh
```
## Credits and Code Reference
* [POD](https://github.com/lileipisces/POD/)
* [P5](https://github.com/jeykigung/P5)
