# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#
# python code for generating the test record file '/tmp/test1.tfrecord'
# Author: Rock Zhuang
# Date  : Dec 12, 2018
# 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
from tensorflow.python.lib.io import python_io
from tensorflow.python.util import compat
from tensorflow.python.lib.io import tf_record

def main(unused_argv):
  example = tf.train.Example(
             features=tf.train.Features(feature={
              "feature_0": tf.train.Feature(int64_list=tf.train.Int64List(value=[111])),
              'feature_1': tf.train.Feature(bytes_list=tf.train.BytesList(value=["1111111111"])),
            })) 

  options = tf_record.TFRecordOptions(tf_record.TFRecordCompressionType.ZLIB)
  writer = python_io.TFRecordWriter("/tmp/test1.tfrecord", options)

  writer.write(example.SerializeToString())

  example = tf.train.Example(
             features=tf.train.Features(feature={
              "feature_0": tf.train.Feature(int64_list=tf.train.Int64List(value=[222])),
              'feature_1': tf.train.Feature(bytes_list=tf.train.BytesList(value=["2222222222"])),
            })) 
  writer.write(example.SerializeToString())

  example = tf.train.Example(
             features=tf.train.Features(feature={
              "feature_0": tf.train.Feature(int64_list=tf.train.Int64List(value=[333])),
              'feature_1': tf.train.Feature(bytes_list=tf.train.BytesList(value=["3333333333"])),
            })) 
  writer.write(example.SerializeToString())

  writer.close()

  tf.compat.v1.logging.info('File /tmp/test1.tfrecord generated!')
  

if __name__ == '__main__':
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  tf.compat.v1.app.run(main=main, argv=sys.argv)
