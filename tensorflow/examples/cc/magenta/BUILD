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

tf_cc_binary(
    name = "melody_lstm_train",
    srcs = [
            "melody_lstm_train.cc", 
            "lstm/lstm_ops.cc",    
            "lstm/lstm_ops.h",
           ],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:tensorflow",
        "//tensorflow/cc:queue_runner",
        "//tensorflow/cc:coordinator",
        "//tensorflow/contrib/rnn:python/ops/_lstm_ops.so",
    ],
    linkopts = ["-g"],
)

tf_cc_binary(
    name = "melody_lstm_autograd",
    srcs = [
            "melody_lstm_autograd.cc", 
            "lstm/lstm_ops.cc",    
            "lstm/lstm_ops.h",
           ],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:tensorflow",
        "//tensorflow/cc:queue_runner",
        "//tensorflow/cc:coordinator",
        "//tensorflow/contrib/rnn:python/ops/_lstm_ops.so",
    ],
    linkopts = ["-g"],
)

tf_cc_binary(
    name = "melody_cudnn_rnn",
    srcs = [
            "melody_cudnn_rnn.cc", 
            "cudnn_rnn/cudnn_rnn_ops.cc",    
            "cudnn_rnn/cudnn_rnn_ops.h",
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
    name = "melody_rl_tuner",
    srcs = [
            "melody_rl_tuner.cc", 
            "rl_tuner/const.h", 
            "rl_tuner/rl_tuner.cc", 
            "rl_tuner/rl_tuner.h", 
            "rl_tuner/note_rnn.cc", 
            "rl_tuner/note_rnn.h", 
            "lstm/lstm_ops.cc",    
            "lstm/lstm_ops.h",
           ],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:tensorflow",
        "//tensorflow/cc:queue_runner",
        "//tensorflow/cc:coordinator",
        "//tensorflow/contrib/rnn:python/ops/_lstm_ops.so",
    ],
    linkopts = ["-g"],
)
