# Naive-KNN :house:
### Description###
This is a naive implementation of the K-Nearest Neighbor Algorithm. It was done in the most simple and naive way as possible. In `my_knn` file you can find my implementation of the main functions used in the KNN Algorithm. The `main.py` file is the main file responsible for read the dataset, use the algorithm and evaluate its performance. In the directory `Data` there is a simple dataset, splitted in `train.dat` and `test.dat`, to use and try the algorithm. This dataset is a generic dataset, so it does not represent something in the real world. 

### Dependencies###
To use the code you will need `scikit-learn` library and `argparse` library installed. You can get the scikit-learn library by running the command-line bellow:

```
pip3 install -U scikit-learn
```

And you can get the `argparse` library by running:

```
pip3 install -U arparse
```

And then, we are good to go :smile:.

### Usage###
You can try my algorithm by the command-line right bellow:
```
python3 main.py train_file test_file k
```
Where:

- `train_file/test_file:` is the path to your train/test file, as an example you can use `Data/train.dat` and `Data/test.dat` as arguments. Just remember to do not invert the order while passing the parameters.

- `k:` is the n neighbors that will be used to classify your samples. 
