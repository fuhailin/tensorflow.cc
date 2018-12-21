load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

tf_cc_binary(
    name = "tfrecord_test",
    srcs = ["tfrecord_test.cc"],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:tensorflow",
    ],
    linkopts = ["-g"],
)

tf_cc_binary(
    name = "rnn_ops_test",
    srcs = ["rnn_ops_test.cc"],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:tensorflow",
    ],
    linkopts = ["-g"],
)
