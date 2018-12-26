Tensorflow C++ Examples

1) TFRecordDataset and Iterator
bazel run -c opt //tensorflow/examples/cc/tool:tfrecord_test
bazel run -c opt //tensorflow/examples/cc/tests:tfrecord_test

2) toydnn from https://github.com/theflofly/dnn_tensorflow_cpp
cp -r tensorflow/examples/cc/toydnn/data /tmp/
bazel run -c opt //tensorflow/examples/cc/toydnn:toydnn

3) Vanalli RNN
bazel run -c opt //tensorflow/examples/cc/tests:rnn_ops_test
