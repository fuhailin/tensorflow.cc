<p>Tensorflow C++ Examples</p>
<p>----------------------------------------------------------------------------------------------</p>
<p>1. TFRecordDataset and Iterator</p>
<p>bazel run -c opt //tensorflow/examples/cc/tool:tfrecord_test</p>
<p>bazel run -c opt //tensorflow/examples/cc/dataset:tfrecord_test</p>
<p>bazel run -c opt //tensorflow/examples/cc/dataset:tfrecord_test2</p>

<p>2. toydnn</p>
<p>cp -r tensorflow/examples/cc/dnn/data /tmp/</p>
<p>bazel run -c opt //tensorflow/examples/cc/dnn:toydnn</p>

<p>3. Vanilla RNN</p>
<p>bazel run -c opt //tensorflow/examples/cc/rnn:rnn_ops_test</p>

<p>4. C++ implementation of Magenta melody basic_rnn module</p>
<p>cp -r tensorflow/examples/cc/magenta/data /tmp/</p>
<p>bazel run -c opt //tensorflow/examples/cc/magenta:melody_basic_rnn</p>
