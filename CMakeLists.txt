cmake_minimum_required (VERSION 2.6)
project (eigenlibsvm)

add_library(svmlib src/svm_utils.cpp)
include_directories("${PROJECT_SOURCE_DIR}/include")

add_executable (svm_test test/svm_test.cpp)
target_link_libraries (svm_test svmlib svm)

