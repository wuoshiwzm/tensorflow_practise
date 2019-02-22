import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv

##################################################################
FLAGS=tf.flags.FLAGS
#data params
tf.flags.DEFINE_string('positive_data_file','./data/rt-polaritydata/rt-polarity.pos','Data source for the positive data')
tf.flags.DEFINE_string('negative_data_file','./data/rt-polaritydata/rt-polarity.neg','Data source for the negative data')

#Eval params
tf.flags.DEFINE_integer('batch_size',64,'batch size(default 64)')
tf.flags.DEFINE_string('checkpoint_dir','./','checkpoint directory from training run')
tf.flags.DEFINE_boolean('eval_train',False,'Evaluate on all training data')

# Misc params
# allow_soft_placement允许动态分配GPU内存，log_device_placement打印出设备信息
tf.flags.DEFINE_boolean('allow_soft_placement',True,'allow device soft device placement')
tf.flags.DEFINE_boolean('log_device_placement',False,'log placement of ...')

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print('\n Parameters:')
for attr,value in sorted(FLAGS.__flags.items()):
    print('{}={}'.format(attr.upper(),value))
print("")

#load data here
if FLAGS.eval_train:
    x_raw,y_test = data_helpers.load_data_labels(
        FLAGS.positive_data_file,FLAGS.negative_data_file
    )
else:
    x_raw = ["a masterpiece four years in the making", "everything is off."]
    y_test = [1, 0]

#map data into vocabulary
vocab_path = ps.path.join(FLAGS.checkpoint_dir,'..','vocab')
vocab_processor = learn.preprocessing.VocabularyProcessor(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))
print('\nEvaluating....\n')

#Evaluation
#######################################################
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement = FLAGS.allow_soft_placement,
        log_device_placement = FLAGS.log_device_placement
    )
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        #loading data
        saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))
        saver.restore(sess,checkpoint_file)

        #get placeholders from graph by name
        input_x = graph.get_operation_by_name('input_x').outputs[0]
        # input_y = graph.get_operation_by_name('input_y').outputs[0]
        dropout_keep_prob = graph.get_operation_by_name('dropout_kep_prob').outputs[0]

        #tensors want to valuate
        predictions = graph.get_operation_by_name('output/predictions').outputs[0]

        #generat batches for one epoch
        batches = data_helpers.batch_iter(list(x_test),FLAGS.batch_size,1,shuffle=False)

        #Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions,{input_x:x_test_batch,dropout_keep_prob:1.0})
            all_predictions = np.concatenate([all_predictions,batch_predictions])

#print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print('total num of test samples:{}'.format(len(y_test)))
    print('Accuracy:{}:g'.format(correct_predictions/float(len(y_test))))

#saving to csv
predictions_human_readable = np.column_stack((np.array(x_raw),all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir,'..','prediction.csv')
print('saving evaluation to {}'.format(out_path))
with open(out_path,'w') as f:
    csv.writer(f).writerows(predictions_human_readable)

