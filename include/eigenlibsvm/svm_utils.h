/*
Author: Andrej Karpathy (http://cs.stanford.edu/~karpathy/)
1 May 2012
BSD licence
*/

#ifndef __EIGEN_SVM_UTILS_H__
#define __EIGEN_SVM_UTILS_H__

#include <string>
#include <vector>
#include <libsvm/svm.h>
#include <eigen3/Eigen/Eigen>

using namespace std;
namespace esvm {
  
  /* 
  Trains a binary SVM using libsvm. Usage:

  vector<int> yhat;
  SVMClassifier svm;
  svm.train(X, y);
  svm.test(X, yhat);

  where X is an Eigen::MatrixXf that is NxD array. (N D-dimensional data),
  y is a vector<int> or an Eigen::MatrixXf Nx1 vector. The labels are assumed
  to be -1 and 1. This version doesn't play nice if your dataset is 
  too unbalanced.
  */
  class SVMClassifier{
    public:
      
      SVMClassifier();
      ~SVMClassifier();
      
      // functions
      void train(const Eigen::MatrixXf &X, const vector<int> &y);
      void train(const Eigen::MatrixXf &X, const Eigen::MatrixXf &y);
      
      void test(const Eigen::MatrixXf &X, vector<int> &yhat);
      
      int saveModel(const char *filename);
      void loadModel(const char *filename); 
      
      void setC(double Cnew); //default is 1.0
      
      
      //TODO: add cross validation support
      
    protected:
    
      svm_model *model_;
      svm_problem *problem_;
      svm_parameter *param_;
      svm_node *x_space_;
  };
};


#endif //__EIGEN_SVM_UTILS_H__
