# EigenLibSVM
Andrej Karpathy
1 May 2012

This is a small C++ wrapper to call libsvm if you use the Eigen matrix library.
Dependencies consist of libsvm and eigen3 library.
Current support is only for dense matrices with linear kernel svm.

## Usage

```c++
vector<int> yhat;
SVMClassifier svm;
svm.train(X, y);
svm.test(X, yhat);
```

where X is an `Eigen::MatrixXf` NxD matrix, y is an `Eigen::MatrixXf` Nx1 matrix of
labels (-1 or 1), or a `vector<int>` of labels. You can also save and load the models:

```c++
svm.saveModel("tmp_model");  
svm.loadModel("tmp_model");  
```

there is now also functionality to directly get the weights:
```c++
Eigen::MatrixXf w;
float b;
svm.getw(w, b);
Eigen::MatrixXf margin= ((X * w).array() + b).matrix(); // yhat = sign(margin)
```

See demo for details.

## Install

```
$ sudo apt-get install libsvm-dev  
$ sudo apt-get install libeigen3-dev  
$ git clone git@github.com:karpathy/EigenLibSVM.git  
$ cd EigenLibSVM  
$ mkdir build  
$ cd build  
$ cmake ..  
$ make  
$ ./svm_test  
```

where the last line will run a tiny demo that makes sure everything installed ok
(it runs almost instantly)

## License
BSD
