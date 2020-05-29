<p>Tensorflow C++ Examples</p>
<p>Check commit history here: https://github.com/rockzhuang/tensorflow/commits?author=rockzhuang</p>
<p>----------------------------------------------------------------------------------------------</p>
<p>Firstly, run './configure'</p>
<br/>
<p>1) TFRecordDataset and Iterator</p>
<p>cp tensorflow/examples/cc/dataset/data/test1.tfrecord /tmp/</p>
<p>bazel run -c opt //tensorflow/examples/cc/dataset:tfrecord_test</p>
<p>bazel run -c opt //tensorflow/examples/cc/dataset:tfrecord_test2</p>
<br/>
<p>2) toydnn</p>
<p>cp -r tensorflow/examples/cc/dnn/data /tmp/</p>
<p>bazel run -c opt //tensorflow/examples/cc/dnn:toydnn</p>
<br/>
<p>3) RNN</p>
<p>bazel run -c opt //tensorflow/examples/cc/rnn:rnn_ops_test</p>
<p>bazel run -c opt //tensorflow/examples/cc/rnn:rnn_ops_restore</p>
<p>bazel run -c opt //tensorflow/examples/cc/rnn:rnn_ops_restore_by_clientsession</p>
<p>bazel run -c opt //tensorflow/examples/cc/rnn:lstm_ops_test</p>
<p>bazel run -c opt //tensorflow/examples/cc/rnn:lstm_multi_layers</p>
<p>bazel run -c opt //tensorflow/examples/cc/rnn:lstm_multi_layers_autograd</p>
<br/>
<p>4) Magenta melody basic_rnn module</p>
<p>cp -r tensorflow/examples/cc/magenta/data /tmp/</p>
<p>bazel run -c opt //tensorflow/examples/cc/magenta:melody_basic_rnn</p>
<p>bazel run -c opt //tensorflow/examples/cc/magenta:melody_lstm_train</p>
<p>bazel run -c opt //tensorflow/examples/cc/magenta:melody_lstm_autograd</p>
<p>bazel run -c opt //tensorflow/examples/cc/magenta:melody_cudnn_rnn</p>
<br/>
<p>5) GeneratorDataset</p>
<p>bazel run -c opt //tensorflow/examples/cc/dataset:generatordataset_test</p> 
<br/>
<p>6) CNN</p>
<p>sh tensorflow/examples/cc/cnn/mnist/data/download_data.sh</p>
<p>bazel run -c opt //tensorflow/examples/cc/cnn/mnist:mnist</p> 
<p>bazel run -c opt //tensorflow/examples/cc/cnn/mnist:mnist_rkz</p>
<br/>
<p>7) Reinforcement Learning Tuner</p>
<p>bazel run -c opt //tensorflow/examples/cc/magenta:melody_rl_tuner</p>
<br/>
<p>8) GAN</p>
<p>sh tensorflow/examples/cc/cnn/mnist/data/download_data.sh</p>
<p>bazel run -c opt //tensorflow/examples/cc/gan/dcgan:dcgan</p>
<p>bazel run -c opt //tensorflow/examples/cc/gan/dcgan:dcgan_multi_gpus</p>
