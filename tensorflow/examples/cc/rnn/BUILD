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

tf_cc_binary(
    name = "rnn_ops_restore_by_clientsession",
    srcs = ["rnn_ops_restore_by_clientsession.cc"],
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
    name = "lstm_ops_test",
    srcs = [
            "lstm_ops_test.cc", 
           ],
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

tf_cc_binary(
    name = "lstm_multi_layers",
    srcs = [
            "lstm_multi_layers.cc", 
           ],
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

tf_cc_binary(
    name = "lstm_multi_layers_autograd",
    srcs = [
            "lstm_multi_layers_autograd.cc", 
           ],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:tensorflow",
        "//tensorflow/cc:queue_runner",
        "//tensorflow/cc:coordinator",
        "//tensorflow/cc:gradients",
        "//tensorflow/cc:nn_grad",
    ],
    linkopts = ["-g"],
)
