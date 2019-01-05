load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

tf_cc_binary(
    name = "toydnn",
    srcs = [
        "toydnn.cc",
        "data_set.cc",
        "data_set.h",
    ],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:gradients",
        "//tensorflow/cc:grad_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
    ],
    linkopts = ["-g"],
)
