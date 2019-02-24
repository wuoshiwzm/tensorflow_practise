#coding=utf-8
# 训练主函数
# print('etst')
# import tensorflow as tf
import tensorflow as tf
import numpy as  np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn

# Params  参数
# FLAGS = tf.flags.FLAGS


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_float('dev_sample_percetage', 1, 'percentage training data in validation loop')
# 正/负例数据
tf.flags.DEFINE_string('positive_data_fill', './data/rt-polaritydata/rt-polarity.pos', 'positive samples')
tf.flags.DEFINE_string('negative_data_fill', './data/rt-polaritydata/rt-polarity.neg', 'negative samples')
# hyperparams
# 中间隐层的维度 每一个单词转换为对应的维度  默认为128
tf.flags.DEFINE_integer('embedding', 128, 'Dimension for character')
# 多个单词一卷积 一个单词对应input的一行
tf.flags.DEFINE_string('filter_size', '3,4,5', 'filter size')
tf.flags.DEFINE_integer('num_filters', 128, 'how many Filters here')
tf.flags.DEFINE_float('drop_out', 0.5, 'Dropout')
tf.flags.DEFINE_float('L2_reg_lambda', 0.0, 'L2')  # 正则化

# 训练参数
tf.flags.DEFINE_integer('batch_size', 64, 'batch size')
tf.flags.DEFINE_integer('num_epochs', 200, 'how many total epochs')
tf.flags.DEFINE_integer('evaluate_every', 100, 'evaluate every time')  # 多少次迭代做一个可视化显示
tf.flags.DEFINE_integer('checkpoint_every', 100, 'save model every ...')  # 每个多少次保留一次模型
# Misc Parameters
tf.flags.DEFINE_boolean('allow_soft_placement',True,'Allow device soft device placement')
# tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# # Flags生效
# FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags() 弃用了


# FLAGS.flag_values_dict()
FLAGS.flag_values_dict()
print('\nParameters:')

for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), str(value.value)))


# ===============================
# 数据预处理
# 载入数据
print('loding data...')
x_text, y = data_helpers.load_data_labels(FLAGS.positive_data_fill, FLAGS.negative_data_file)

# build vocabulary 每篇文章中的词数最大的那一个
# 创建词汇表
max_document_length = max([len(x.split(' ')) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length=max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))

# shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.array(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# split train/test set  分开测试与训练集 这里是从后往前
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train,x_dev = x_shuffled[:dev_sample_index],x_shuffled[dev_sample_index:]
y_train,y_dev = y_shuffled[:dev_sample_index],x_shuffled[dev_sample_index:]
print('Vocabulary Size:{:d}'.format(len(vocab_processor.vocabulary_)))
print('Train/Dev split:{:d}/{:d}'.format(len(y_train),len(y_dev)))




# ============================
# train
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement = FLAGS.allow_soft_placement,
        log_device_placement = FLAGS.log_device_placement
    )
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length= x_train.shape[1],
            num_classes= y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes = list(map(int,FLAGS.filter_sizes.split(','))),  #不同的filter层数
            num_filters=FLAGS.num_filters,
            l2_reg_lambda= FLAGS.l2_reg_lambda
        )

        # 训练过程
        global_step = tf.Variable(0,name='global_step',trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss) #传入loss值
        train_op = optimizer.apply_gradients(grads_and_vars,global_step=global_step)

        #
        grad_summaries = []
        for g,v in grads_and_vars:
            if g is not None:
                # 在训练神经网络时，当需要查看一个张量在训练过程中值的分布情况时，
                # 可通过tf.summary.histogram()将其分布情况以直方图的形式在TensorBoard直方图仪表板上显示．
                grad_hist_summary = tf.summary.histogram('{}/grad/hist'.format(v.name),g)
                sparsity_summary = tf.summary.scalar('{}/grad/sparsity'.format(v.name),tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # output directory for models summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir,'runs',timestamp))
        print('Writing to {}\n'.format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar('loss',cnn.loss)
        acc_summary = tf.summary.scalar('accuracy',cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary,acc_summary,grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir,'summaries','dev')
        train_summary_writer = tf.summary.FileWriter(train_summary_dir,sess.graph)

        # Dev Summaries
        dev_summary_op = tf.summary.merge([loss_summary,acc_summary])
        dev_summary_dir = os.path.join(out_dir,'summaries','dev')
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir,sess.graph)

        #Checkpoint directory, we need to create it , for tensorflow assume it is already existed
        checkpoint_dir = os.path.abspath(os.path.join(out_dir,'checkpoints'))
        checkpoint_prefix = os.path.join(checkpoint_dir,'model')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(),max_to_keep=FLAGS.num_checkpoints)

        #write vocabulary
        vocab_processor.save(os.path.join(out_dir,'vocab'))

        #
        sess.run(tf.global_variables_initializer())

        # 训练阶段
        def train_step(x_batch,y_batch):

            feed_dict = {
                cnn.input_x:x_batch,
                cnn.input_y:y_batch,
                cnn.dropout_keep_prob:FLAGS.dropout_keep_prob
            }
            # 运行一堆操作
            _,step,summaries,loss,accuracy = sess.run(
                [train_op,global_step,train_summary_op,cnn.loss,cnn.accuracy],feed_dict
            )

            time_str = datetime.datetime.now().isoformat()
            print('{}:step {} , loss {:g}, acc{:g}',format(time_str,step,loss,accuracy))
            train_summary_writer.add_summary(summaries,step)

        #验证阶段
        def dev_step(x_batch,y_batch,writer=None):
            feed_dict = {
                cnn.input_x:x_batch,
                cnn.input_y:y_batch,
                cnn.dropout_keep_prob:1.0
            }

            step,summaries,loss,accuracy = sess.run(
                [global_step,dev_summary_op,cnn.loss,cnn.accuracy],feed_dict=feed_dict
            )

            time_str = datetime.datetime.now().isoformat()
            print('{}:step {}, loss{:g}, acc{:g}'.format(time_str,step,loss,accuracy))
            if writer:
                writer.add_summary(summaries,step)


        #generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train,y_train)),FLAGS.batch_size,FLAGS.num_epochs
        )
        for batch in batches:
            # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
            # list(tuple())  list包tuple
            # 利用 * 号操作符，可以将元组解压为列表
            x_batch,y_batch = zip(*batch)
            train_step(x_batch,y_batch)
            current_step = tf.train.global_step(sess,global_step)

            #写入summary
            if current_step%FLAGS.evaluate_every == 0:
                print('\nEvaluation')
                dev_step(x_dev,y_dev,writer=dev_summary_writer)
                print('')
            #saver
            if current_step%FLAGS.checkpoint_every == 0:
                path = saver.save(sess,'./',global_step=current_step)
                print('Saved model checkpoint to {}\n'.format(path))









