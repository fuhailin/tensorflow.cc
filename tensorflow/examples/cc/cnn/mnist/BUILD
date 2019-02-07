load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

tf_cc_binary(
    name = "mnist",
    srcs = [
        "mnist.cc",
    ],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
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

tf_cc_binary(
    name = "mnist_rkz",
    srcs = [
        "mnist_rkz.cc",
    ],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
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
