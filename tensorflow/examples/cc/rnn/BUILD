load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

tf_cc_binary(
    name = "rnn_ops_test",
    srcs = ["rnn_ops_test.cc"],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:tensorflow",
        "//tensorflow/cc/saved_model:loader",
        "//tensorflow/cc/tools:freeze_saved_model",
    ],
    linkopts = ["-g"],
)

tf_cc_binary(
    name = "rnn_ops_restore",
    srcs = ["rnn_ops_restore.cc"],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:tensorflow",
        "//tensorflow/cc/saved_model:loader",
        "//tensorflow/cc/tools:freeze_saved_model",
    ],
    linkopts = ["-g"],
)
