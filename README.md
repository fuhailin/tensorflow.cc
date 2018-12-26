<p>Tensorflow C++ Examples</p>
<p>----------------------------------------------------------------------------------------------</p>
<p>1. TFRecordDataset and Iterator</p>
<p>bazel run -c opt //tensorflow/examples/cc/tool:tfrecord_test</p>
<p>bazel run -c opt //tensorflow/examples/cc/tests:tfrecord_test</p>

<p>2. toydnn</p>
<p>cp -r tensorflow/examples/cc/toydnn/data /tmp/</p>
<p>bazel run -c opt //tensorflow/examples/cc/toydnn:toydnn</p>

<p>3. Vanalli RNN</p>
<p>bazel run -c opt //tensorflow/examples/cc/tests:rnn_ops_test</p>

