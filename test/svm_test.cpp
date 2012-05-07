/*
Author: Andrej Karpathy (http://cs.stanford.edu/~karpathy/)
1 May 2012
BSD licence
*/

#include <string>
#include <vector>
#include <stdlib.h>

#include <eigen3/Eigen/Eigen>
#include <eigenlibsvm/svm_utils.h>
#include <eigenlibsvm/eigen_extensions.h>

using namespace std;
using namespace esvm;

// run from build folder of the project (see path below)
int main (int argc, char** argv) {
  
  Eigen::MatrixXf X;
  Eigen::MatrixXf y;
  eigen_extensions::loadASCII("../test/svmtestx.eig.txt", &X);
  eigen_extensions::loadASCII("../test/svmtesty.eig.txt", &y);
  
  // Classify
  cout << X.topRows(5) << endl;
  cout << y.topRows(5) << endl;
  vector<int> yhat;
  SVMClassifier svm;
  svm.train(X, y);
  svm.test(X, yhat);
  
  Eigen::MatrixXf w;
  float b;
  svm.getw(w, b);
  Eigen::MatrixXf margin= ((X * w).array() + b).matrix(); // ahh eigen...
  
  // Evaluate accuracy and print results
  int match=0;
  for(int i=0;i<yhat.size();i++) {
    if(yhat[i]==(int)y(i)) match++; else printf("WRONG! ");
    printf("y= %d, yhat= %d margin= %f\n", (int)y(i), yhat[i], margin(i));
  }
  printf("MATLAB cross-check: last margin should be around -3.107\n");
  printf("Accuracy= %f. From MATLAB cross-check, expect this to be around 0.945\n", 1.0*match/yhat.size());
  
  // Save the model
  svm.saveModel("temp.svmmodel");
  
  // Test loading model in new instance of SVMClassifier
  SVMClassifier svm2;
  svm2.loadModel("temp.svmmodel");
  yhat.clear();
  svm2.test(X, yhat);
  match=0; 
  for(int i=0;i<yhat.size();i++) if(yhat[i]==(int)y(i)) match++;
  printf("Accuracy= %f from loaded model. Should be 0.945 again.\n", 1.0*match/yhat.size());
  
  
  printf("you may want to rm the temporary file temp.svmmodel\n");
}
  
