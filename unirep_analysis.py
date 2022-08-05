

import tensorflow as tf
import numpy as np
import os
from argparse import ArgumentParser,Action
import json
import shutil
import logging
import sys

import glob

import pudb

# Set seeds
tf.set_random_seed(42)
np.random.seed(42)

class NoArgAction(Action):
    """
    Base class for action classes that do not have arguments.
    """
    def __init__(self, *args, **kwargs):
        kwargs['nargs'] = 0
        Action.__init__(self, *args, **kwargs)

class JsonAction(NoArgAction):
    """
    Custom action class to bypass required positional arguments when printing the app's
    JSON representation.
    """
    def __call__(self, parser, namespace, values, option_string=None):
        f=open('chris_plugin_info.json')
        data = json.load(f)
        print(json.dumps(data))
        f.close()
        parser.exit()

parser = ArgumentParser(description='cli description')

parser.add_argument('--dimension','-d',
                    type        = int,
                    dest        = 'dimension',
                    default     = 64,
                    help        = "Dimension of the model")

parser.add_argument('--batch_size','-b',
                    type        = int,
                    dest        = 'batch_size',
                    default     = 12,
                    help        = "Batch size of the model")

parser.add_argument('--learning_rate','-l',
                    type        = float,
                    dest        = 'learning_rate',
                    default     = .001,
                    help        = "Learning rate of the training model")

parser.add_argument('--inputFile','-i',
                    type        = str,
                    dest        = 'inputFile',
                    default     = '',
                    help        = "Input file for the analysis")

parser.add_argument('--inputGlob','-g',
                    type        = str,
                    dest        = 'inputGlob',
                    default     = '**/*.txt',
                    help        = "Input glob to find sequence text file")

parser.add_argument('--modelWeightPath','-m',
                    type        = str,
                    dest        = 'modelWeightPath',
                    default     = '/usr/local/lib/unirep_analysis/',
                    help        = "Directory containing weight files")

parser.add_argument('--outputFile','-o',
                    type        = str,
                    dest        = 'outputFile',
                    default     = 'format.txt',
                    help        = "Output file for the analysis")


parser.add_argument('--json', action=JsonAction, dest='json',
                    default=False,
                    help='show json representation of app and exit')

parser.add_argument('--saveinputmeta',
                    action      = 'store_true',
                    dest        = 'saveinputmeta',
                    default     = False,
                    help        = "save arguments to a JSON file")

parser.add_argument('--saveoutputmeta',
                    action      = 'store_true',
                    dest        = 'saveoutputmeta',
                    default     = False,
                    help        = "save output meta data to a JSON file")

parser.add_argument(
                    type        = str,
                    dest        = 'inputdir',
                    default     = "",
                    help        = "Input path to the app")

parser.add_argument(
                    type        = str,
                    dest        = 'outputdir',
                    default     = "",
                    help        = "Output path to the app")


def main():
  """
  Define the code to be run by this plugin app.
  """


  args = parser.parse_args()

  print('Version: 0.2.8')
  for k,v in args.__dict__.items():
            print("%20s:  -->%s<--" % (k, v))



  # Set up the logger
  logger = logging.getLogger("eval")
  logger.setLevel(logging.DEBUG)
  logger.addHandler(logging.StreamHandler(stream=sys.stdout))

  logger.info("Using {}_weights".format(args.dimension))
  get_data(args)

  logger.info("Preparing data")
  prepare_data(args)

  logger.info("Formatting data")
  format_data(args)

  logger.info("Bucketting data")
  bucket_data(args)

  logger.info("Preparing model")
  prepare_model(args)

def get_data(args):

  global data_babbler
  global MODEL_WEIGHT_PATH

  # pudb.set_trace()

  if args.dimension==1900:

    # Import the mLSTM babbler model
    from src.unirep import babbler1900 as babbler
    data_babbler=babbler
    # Where model weights are stored.
    MODEL_WEIGHT_PATH = '%s/%s' % (args.modelWeightPath, "1900_weights")

  elif args.dimension==256:

    # Import the mLSTM babbler model
    from src.unirep import babbler256 as babbler
    data_babbler=babbler
    # Where model weights are stored.
    MODEL_WEIGHT_PATH = '%s/%s' % (args.modelWeightPath, "256_weights")

  else:

    # Import the mLSTM babbler model
    from src.unirep import babbler64 as babbler
    data_babbler=babbler
    # Where model weights are stored.
    MODEL_WEIGHT_PATH = '%s/%s' % (args.modelWeightPath, "64_weights")


def prepare_data(args):
  # ## Data formatting and management

  # Initialize UniRep, also referred to as the "babbler" in our code. You need to provide the batch size you will use and the path to the weight directory.

  # In[3]:
  global b


  b = data_babbler(batch_size=args.batch_size, model_path=MODEL_WEIGHT_PATH)


  # UniRep needs to receive data in the correct format, a (batch_size, max_seq_len) matrix with integer values, where the integers correspond to an amino acid label at that position, and the end of the sequence is padded with 0s until the max sequence length to form a non-ragged rectangular matrix. We provide a formatting function to translate a string of amino acids into a list of integers with the correct codex:

  # In[4]:


  seq = "MRKGEELFTGVVPILVELDGDVNGHKFSVRGEGEGDATNGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFARYPDHMKQHDFFKSAMPEGYVQERTISFKDDGTYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNFNSHNVYITADKQKNGIKANFKIRHNVEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSVLSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"


  # In[5]:


  np.array(b.format_seq(seq))


  # We also provide a function that will check your amino acid sequences don't contain any characters which will break the UniRep model.

  # In[7]:


  b.is_valid_seq(seq)

def format_data(args):
  global interim_format_path
  # You could use your own data flow as long as you ensure that the data format is obeyed. Alternatively, you can use the data flow we've implemented for UniRep training, which happens in the tensorflow graph. It reads from a file of integer sequences, shuffles them around, collects them into groups of similar length (to minimize padding waste) and pads them to the max_length. Here's how to do that:

  # First, sequences need to be saved in the correct format. Suppose we have a new-line seperated file of amino acid sequences, `seqs.txt`, and we want to format them. Note that training is currently only publicly supported for amino acid sequences less than 275 amino acids as gradient updates for sequences longer than that start to get unwieldy. If you want to train on sequences longer than this, please reach out to us.
  #
  # Sequence formatting can be done as follows:

  # In[8]:


  # Before you can train your model,
  interim_format_path = "/tmp/formatted.txt"
  input_file_path = ""
  # Determine the input sequence file either from the inputfile spec or glob pattern.
  # If multiple files found, only process the first file.
  str_glob = ""
  if len(args.inputFile):
    str_glob = '%s/**/%s' % (args.inputdir, args.inputFile)
  else:
    if len(args.inputGlob):
      str_glob = '%s/%s' % (args.inputdir, args.inputGlob)
  if len(str_glob):
    l_allHits = glob.glob(str_glob, recursive = True)
    if len(l_allHits): input_file_path = l_allHits[0]
    else:
      print("No valid input sequence text file was found!")
      sys.exit(1)

  output_file_path = os.path.join(args.outputdir,args.outputFile)
  with open(input_file_path, "r") as source:
      with open(interim_format_path, "w") as destination:
          for i,seq in enumerate(source):
              seq = seq.strip()
              if b.is_valid_seq(seq) and len(seq) < 275:
                  formatted = ",".join(map(str,b.format_seq(seq)))
                  destination.write(formatted)
                  destination.write('\n')
  shutil.copy(interim_format_path,output_file_path)


  # This is what the integer format looks like

  # In[9]:


  os.system('head -n1 {}'.format(interim_format_path))

def bucket_data(args):
  global batch
  global bucket_op
  # Notice that by default format_seq does not include the stop symbol (25) at the end of the sequence. This is the correct behavior if you are trying to train a top model, but not if you are training UniRep representations.

  # Now we can use a custom function to bucket, batch and pad sequences from `formatted.txt` (which has the correct integer codex after calling `babbler.format_seq()`). The bucketing occurs in the graph.
  #
  # What is bucketing? Specify a lower and upper bound, and interval. All sequences less than lower or greater than upper will be batched together. The interval defines the "sides" of buckets between these bounds. Don't pick a small interval for a small dataset because the function will just repeat a sequence if there are not enough to
  # fill a batch. All batches are the size you passed when initializing the babbler.
  #
  # This is also doing a few other things:
  # - Shuffling the sequences by randomly sampling from a 10000 sequence buffer
  # - Automatically padding the sequences with zeros so the returned batch is a perfect rectangle
  # - Automatically repeating the dataset

  # In[10]:


  bucket_op = b.bucket_batch_pad(interim_format_path, interval=1000) # Large interval


  # Inconveniently, this does not make it easy for a value to be associated with each sequence and not lost during shuffling. You can get around this by just prepending every integer sequence with the sequence label (eg, every sequence would be saved to the file as "{brightness value}, 24, 1, 5,..." and then you could just index out the first column after calling the `bucket_op`. Please reach out if you have questions on how to do this.

  # Now that we have the `bucket_op`, we can simply `sess.run()` it to get a correctly formatted batch

  # In[11]:


  with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      batch = sess.run(bucket_op)

  print(batch)
  print(batch.shape)

def prepare_model(args):
  global loss,top_only_step_op,x_placeholder,y_placeholder,seq_length_placeholder,initial_state_placeholder,batch_size_placeholder,all_step_op
  # You can look back and see that the batch_size we passed to __init__ is indeed 12, and the second dimension must be the longest sequence included in this batch. Now we have the data flow setup (note that as long as your batch looks like this, you don't need my flow), so we can proceed to implementing the graph. The module returns all the operations needed to feed in sequence and get out trainable representations.

  # ## Training a top model and a top model + mLSTM.

  # First, obtain all of the ops needed to output a representation

  # In[12]:


  final_hidden, x_placeholder, batch_size_placeholder, seq_length_placeholder, initial_state_placeholder = (
      b.get_rep_ops())


  # `final_hidden` should be a batch_size x rep_dim matrix.
  #
  # Lets say we want to train a basic feed-forward network as the top model, doing regression with MSE loss, and the Adam optimizer. We   can do that by:
  #
  # 1.  Defining a loss function.
  #
  # 2.  Defining an optimizer that's only optimizing variables in the top model.
  #
  # 3.  Minimizing the loss inside of a TensorFlow session

  # In[13]:


  y_placeholder = tf.placeholder(tf.float32, shape=[None,1], name="y")
  initializer = tf.contrib.layers.xavier_initializer(uniform=False)

  with tf.variable_scope("top"):
      prediction = tf.contrib.layers.fully_connected(
          final_hidden, 1, activation_fn=None,
          weights_initializer=initializer,
          biases_initializer=tf.zeros_initializer()
      )

  loss = tf.losses.mean_squared_error(y_placeholder, prediction)


  # You can specifically train the top model first by isolating variables of the "top" scope, and forcing the optimizer to only optimize these.

  # In[14]:


  learning_rate=args.learning_rate
  top_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="top")
  optimizer = tf.train.AdamOptimizer(learning_rate)
  top_only_step_op = optimizer.minimize(loss, var_list=top_variables)
  all_step_op = optimizer.minimize(loss)


  # We next need to define a function that allows us to calculate the length each sequence in the batch so that we know what index to use to obtain the right "final" hidden state

  # In[15]:


def nonpad_len(batch):
  nonzero = batch > 0
  lengths = np.sum(nonzero, axis=1)
  return lengths

def train_model(args,batch):
  nonpad_len(batch)


  # We are ready to train. As an illustration, let's learn to predict the number 42 just optimizing the top model.

  # In[16]:


  y = [[42]]*args.batch_size
  num_iters = 10
  with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      for i in range(num_iters):
          batch = sess.run(bucket_op)
          length = nonpad_len(batch)
          loss_, __, = sess.run([loss, top_only_step_op],
                  feed_dict={
                       x_placeholder: batch,
                       y_placeholder: y,
                       batch_size_placeholder: args.batch_size,
                       seq_length_placeholder:length,
                       initial_state_placeholder:b._zero_state
                  }
          )

          print("Iteration {0}: {1}".format(i, loss_))

def joint_train_model(args,batch):
  # We can also jointly train the top model and the mLSTM. Note that if using the 1900-unit (full) model, you will need a GPU with at least 16GB RAM. To see a demonstration of joint training with fewer computational resources, please run this notebook using the 64-unit model.

  # In[17]:


  y = [[42]]*args.batch_size
  num_iters = 10
  with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      for i in range(num_iters):
          batch = sess.run(bucket_op)
          length = nonpad_len(batch)
          loss_, __, = sess.run([loss, all_step_op],
                  feed_dict={
                       x_placeholder: batch,
                       y_placeholder: y,
                       batch_size_placeholder: args.batch_size,
                       seq_length_placeholder:length,
                       initial_state_placeholder:b._zero_state
                  }
          )

          print("Iteration {0}: {1}".format(i,loss_))

