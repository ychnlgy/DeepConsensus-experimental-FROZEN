# DeepConsensus

Using the consensus of low and high level features for robust classification.

This is the original code used to generate all the results in our [DeepConsensus paper](https://arxiv.org/abs/1811.07266).

Please excuse the appearance of the code, we are working on [polishing it](https://github.com/ychnlgy/DeepConsensus).

## Dependencies

#### Exact versions of software
- python 3.6
- pytorch 0.4.1
- torchvision 0.2.1
- tqdm 4.26.0
- numpy 1.15.4
- scipy 1.1.0

#### Extras (to be removed in the polished version)
- matplotlib 3.0.0
- (please raise a GitHub issue if modules are missing here) 

## Examples
Change directory to ```./src/```. Then run:
```bash
./train.py modelf=sample.torch dataset=mnist64-corrupt epochs=30 mintrans=20 maxtrans=20 modelid=<modelid> download=1
```
to see performance on the MNIST-64 test set with 20 pixel translations in both x and y axes. Setting download to 1 allows the program to download the dataset into ```./data```. Replace ```<modelid>``` with:

|```<modelid>``` | Model |
|----------------|-------|
| 0              | DeepConsensus-ResNet |
| 1              | ResNet |
| 2              | DeepConsensus-CNN |
| 3              | CNN |
| 4              | _p4m_ CNN |
| 5              | DeepConsensus-small CNN |
| 6              | Small CNN |

Other interesting command line options include:

|```dataset=``` | Description |
|---------------|---------|
| mnist | 32-pixel square MNIST |
| mnist-corrupt | 32-pixel square MNIST with the option of adding pertubations |
| mnist64 | 64-pixel square MNIST |
| mnist64-corrupt | 64-pixel square MNIST with the option of adding pertubations |
| mnist64-quadrants | 64-pixel square MNIST with the quadrant of the digit also determining its class (40 classes) |
| mnist-rgb | 32-pixel square MNIST copied into 3 channels |
| mnist-rgb-corrupt | 32-pixel square MNIST copied into 3 channels with the option of adding pertubations |
| mnist-rot | 32-pixel square standard MNIST-rot dataset. Place ```.amat``` files in ```./data``` first.|
| fashion | 32-pixel square FashionMNIST |
| fashion64-corrupt | 64-pixel square FashionMNIST with the option of adding pertubations |
| svhn | 32-pixel square SVHN |
| svhn-corrupt | 32-pixel square SVHN with the option of adding pertubations |
| stl10 | 96-pixel square STL-10 |
| stl9-64stretch | 64-pixel square STL-10 with 9 classes that match classes in CIFAR10 (see cifar9-64stretch) |
| cifar10 | 32-pixel square CIFAR10 |
| cifar10-corrupt | 32-pixel square CIFAR10 with the option of adding pertubations |
| cifar1064 | 64-pixel square CIFAR10 |
| cifar1064-corrupt | 64-pixel square CIFAR10 with the option of adding pertubations |
| cifar9-64stretch | 64-pixel square CIFAR10 with 9 classes that match classes in STL-10 (see stl9-64stretch) |
| emnist | 32-pixel square EMNIST |
| emnist-corrupt | 32-pixel square EMNIST with the option of adding pertubations |
| emnist64-corrupt | 64-pixel square EMNIST with the option of adding pertubations |

Note that images which are normally smaller than 64-pixel squares are simply centered in a black 64-pixel background. Pertubation options include:

| Command line option | Description |
|---------------------|-------------|
| ```mirrorx=1```     | Horizontal flip |
| ```mirrory=1```     | Vertical flip |
| ```minmag=1.5 maxmag=1.5``` | Magnification of 1.5 |
| ```minrot=45 maxrot=45``` | Rotation of 45 degrees counterclockwise |
| ```mintrans=20 maxtrans=20``` | Translation of 20 pixels in x and y axes |
| ```minsigma=1.5 maxsigma=1.5``` | Gaussian blur with 1.5 standard deviations |
| ```mingauss=20 maxgauss=20``` | Gaussian noise addition with 20 standard deviations |

Note that to get a range of random pertubations, set the min value lower than the max value. The pertubations can also be combined. For example ```minmag=1.5 maxmag=1.5 mintrans=10 maxtrans=10``` produces test set images that are magnified 1.5 times and translated 10 pixels.

To perturb the training set, use ```corrupt_train=1``` and use the prefix ```TRAIN_``` in front of pertubation options. For example, ```corrupt_train=1 minmag=1.5 maxmag=1.5 TRAIN_mintrans=20 TRAIN_maxtrans=20``` means to train with images translated 20 pixels in both axes, but test with 1.5 magnified images (that are not translated). To perturb only the training set (while using the original test set), use ```corrupt_train=1 corrupt_test=0```.

To obtain scores and samples on DeepFool, use ```fool=1```.

There are numerous other command line options, so please email ```ychnlgy@utoronto.ca``` if you would like to see them listed here.
