load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

tf_cc_binary(
    name = "pix2pix",
    srcs = [
        "pix2pix.cc",
        "model.cc",
        "model.h",
        "function.cc",
        "function.h",
        "const.h",
        "test.h",
    ],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:tf_ops",
        "//tensorflow/cc:gradients",
        "//tensorflow/cc:optimizer",
        "//tensorflow/cc:grad_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
    ],
    copts = ["-I/usr/include/opencv"],
    linkopts = ["-g -lopencv_shape -lopencv_stitching -lopencv_superres -lopencv_videostab -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_datasets -lopencv_dpm -lopencv_face -lopencv_freetype -lopencv_fuzzy -lopencv_hdf -lopencv_line_descriptor -lopencv_optflow -lopencv_video -lopencv_plot -lopencv_reg -lopencv_saliency -lopencv_stereo -lopencv_structured_light -lopencv_phase_unwrapping -lopencv_rgbd -lopencv_viz -lopencv_surface_matching -lopencv_text -lopencv_ximgproc -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_xobjdetect -lopencv_objdetect -lopencv_ml -lopencv_xphoto -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_photo -lopencv_imgproc -lopencv_core"],
)
