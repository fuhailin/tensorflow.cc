load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

tf_cc_binary(
    name = "melody_basic_rnn",
    srcs = ["melody_basic_rnn.cc"],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:tensorflow",
        "//tensorflow/cc:queue_runner",
        "//tensorflow/cc:coordinator",
    ],
    linkopts = ["-g"],
)
