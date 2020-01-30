load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

tf_cc_binary(
    name = "dcgan",
    srcs = [
        "dcgan.cc",
        "nn_ops_rkz.cc",
        "nn_ops_rkz.h",
        "optimizer.cc",
        "optimizer.h",
        "util.h",
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
