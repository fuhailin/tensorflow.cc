<p>Tensorflow C++ Examples</p>
<p>----------------------------------------------------------------------------------------------</p>
<p>1) TFRecordDataset and Iterator</p>
<p>bazel run -c opt //tensorflow/examples/cc/tool:tfrecord_test</p>
<p>bazel run -c opt //tensorflow/examples/cc/dataset:tfrecord_test</p>
<p>bazel run -c opt //tensorflow/examples/cc/dataset:tfrecord_test2</p>
<br/>
<p>2) toydnn</p>
<p>cp -r tensorflow/examples/cc/dnn/data /tmp/</p>
<p>bazel run -c opt //tensorflow/examples/cc/dnn:toydnn</p>
<br/>
<p>3) Vanilla RNN</p>
<p>bazel run -c opt //tensorflow/examples/cc/rnn:rnn_ops_test</p>
<p>bazel run -c opt //tensorflow/examples/cc/rnn:rnn_ops_restore</p>
<br/>
<p>4) C++ implementation of Magenta melody basic_rnn module</p>
<p>cp -r tensorflow/examples/cc/magenta/data /tmp/</p>
<p>bazel run -c opt //tensorflow/examples/cc/magenta:melody_basic_rnn</p>
<p>bazel run -c opt //tensorflow/examples/cc/magenta:melody_lstm_train</p>
<br/>
<p>5) GeneratorDataset</p>
<p>bazel run -c opt //tensorflow/examples/cc/dataset:generatordataset_test</p> 
<br/>
