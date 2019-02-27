#from data_load import get_batch_data
#x, y, num_batch = get_batch_data()

import tensorflow as tf
import numpy as np


def multihead_attention(queries, keys, num_units=None,
                        num_heads=0,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="mulithead_attention",
                        reuse=None):
  '''Applies multihead attention.

  Args:
    queries: A 3d tensor with shape of [N, T_q, C_q].
    keys: A 3d tensor with shape of [N, T_k, C_k].
    num_units: A scalar. Attention size.
    dropout_rate: A floating point number.
    is_training: Boolean. Controller of mechanism for dropout.
    causality: Boolean. If true, units that reference the future are masked.
    num_heads: An int. Number of heads.
    scope: Optional scope for `variable_scope`.
    reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.

  Returns
    A 3d tensor with shape of (N, T_q, C)
  '''
  with tf.variable_scope(scope, reuse=reuse):
    if num_units is None:
      num_units = queries.get_shape().as_list[-1]

    # Linear projection
    Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  #
    K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  #
    V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  #

    # Split and Concat
    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  #
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

    outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
    outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

    # 这里是对填充的部分进行一个mask，这些位置的attention score变为极小，我们的embedding操作中是有一个padding操作的，
    # 填充的部分其embedding都是0，加起来也是0，我们就会填充一个很小的数。
    key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))
    key_masks = tf.tile(key_masks, [num_heads, 1])
    key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])

    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)

    # 这里其实就是进行一个mask操作，不给模型看到未来的信息。
    if causality:
      diag_vals = tf.ones_like(outputs[0, :, :])
      # tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense()
      tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # fix by bincai
      masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])

      paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
      outputs = tf.where(tf.equal(masks, 0), paddings, outputs)

    outputs = tf.nn.softmax(outputs)

    # Query Mask
    query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))
    query_masks = tf.tile(query_masks, [num_heads, 1])
    query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])
    outputs *= query_masks

    # Dropout
    outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

    # Weighted sum
    outputs = tf.matmul(outputs, V_)

    # restore shape
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)

    # Residual connection
    outputs += queries

    # Normalize
    outputs = normalize(outputs)

  return outputs

def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs

EMBEDDING_DIM = 9
EMBEDDING_SIZE = 20
BATCHSIZE=2

emb=np.random.rand(EMBEDDING_SIZE, EMBEDDING_DIM)
emb[0]=np.zeros(EMBEDDING_DIM)
embedding=tf.cast(tf.convert_to_tensor(emb), dtype=tf.float32)

queries1 = tf.convert_to_tensor([[1,3,1,0,0],[1,0,0,0,0]])
queries = tf.nn.embedding_lookup(embedding, queries1)
keys = queries
queriesLen = tf.convert_to_tensor([3,1])

num_units = EMBEDDING_DIM
num_heads = 3

# Linear projection
Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (2, 5, 9) (BATCHSIZE, maxlen, hidden_units)
K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)
V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)

# Split and Concat
Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (6, 5, 3) (BATCHSIZE*num_heads, maxlen, hidden_units/num_heads)
K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # 所以：num_heads必须整除hidden_units
V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

#Q
#[[[0.8233778  0.         0.         0.         0.         0.         0.         0.07351317 0.        ]
#  [0.28679052 0.24491654 0.         0.         0.09461029 0.         0.         0.5929613  0.        ]
#  [0.8233778  0.         0.         0.         0.         0.         0.         0.07351317 0.        ]
#  [0.         0.         0.         0.         0.         0.         0.         0.         0.        ]
#  [0.         0.         0.         0.         0.         0.         0.         0.         0.        ]]
#  ...
#]
#Q_
#[[[0.8233778  0.         0.        ]
#  [0.28679052 0.24491654 0.        ]
#  [0.8233778  0.         0.        ]
#  [0.         0.         0.        ]
#  [0.         0.         0.        ]]...
# [[0.         0.         0.        ]
#  [0.         0.09461029 0.        ]
#  [0.         0.         0.        ]
#  [0.         0.         0.        ]
#  [0.         0.         0.        ]]...
# [[0.         0.07351317 0.        ]
#  [0.         0.5929613  0.        ]
#  [0.         0.07351317 0.        ]
#  [0.         0.         0.        ]
#  [0.         0.         0.        ]]...
#]
K_t=tf.transpose(K_, [0, 2, 1])    # K_t (6, 3, 5)
outputs1 = tf.matmul(Q_, K_t)      # Q_  (6, 5, 3) (BATCHSIZE*num_heads, maxlen, hidden_units/num_heads) 
outputs2 = outputs1 / (K_.get_shape().as_list()[-1] ** 0.5) # (6, 5, 5) (BATCHSIZE*num_heads, maxlen, maxlen)

# 这里是对填充的部分进行一个mask，这些位置的attention score变为极小，我们的embedding操作中是有一个padding操作的，
# 填充的部分其embedding都是0，加起来也是0，我们就会填充一个很小的数。
key_masks1 = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))
#[[1. 1. 1. 0. 0.] [1. 0. 0. 0. 0.]]
key_masks2 = tf.tile(key_masks1, [num_heads, 1])
#[[1. 1. 1. 0. 0.] [1. 0. 0. 0. 0.]
# [1. 1. 1. 0. 0.] [1. 0. 0. 0. 0.]
# [1. 1. 1. 0. 0.] [1. 0. 0. 0. 0.]]
key_masks = tf.tile(tf.expand_dims(key_masks2, 1), [1, tf.shape(queries)[1], 1])  # (6, 5, 5) (BATCHSIZE*num_heads, maxlen, maxlen)

paddings = tf.ones_like(outputs2) * (-2 ** 32 + 1)
outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs2)  # (6, 5, 5) 

#{{ if causality部分
#diag_vals = tf.ones_like(outputs[0, :, :])  # (5, 5)  (maxlen, maxlen)
## tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense()
#tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()   # (5, 5)  (maxlen, maxlen)  下三角 fix by bincai
##[[1. 0. 0. 0. 0.]
## [1. 1. 0. 0. 0.]
## [1. 1. 1. 0. 0.]
## [1. 1. 1. 1. 0.]
## [1. 1. 1. 1. 1.]]
#masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (6, 5, 5)  BATCHSIZE*num_heads个(5, 5) tile() 平铺之意，用于在同一维度上的复制
#paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
#outputs = tf.where(tf.equal(masks, 0), paddings, outputs)
#}} if causality部分

outputs1 = tf.nn.softmax(outputs)  # (6, 5, 5)  ->  (6, 5, 5)

#  #Query Mask
#  query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))
#  query_masks = tf.tile(query_masks, [num_heads, 1])
#  query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])
#  outputs *= query_masks
query_masks1 = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (2, 5)
#[[1. 1. 1. 0. 0.] [1. 0. 0. 0. 0.]]
query_masks2 = tf.tile(query_masks1, [num_heads, 1])             # (6, 5)
#[[1. 1. 1. 0. 0.] [1. 0. 0. 0. 0.]
# [1. 1. 1. 0. 0.] [1. 0. 0. 0. 0.]
# [1. 1. 1. 0. 0.] [1. 0. 0. 0. 0.]]
query_masks3 = tf.expand_dims(query_masks2, -1)                  # (6, 5, 1)
#[[[1.] [1.] [1.] [0.] [0.]] [[1.] [0.] [0.] [0.] [0.]]
# [[1.] [1.] [1.] [0.] [0.]] [[1.] [0.] [0.] [0.] [0.]]
# [[1.] [1.] [1.] [0.] [0.]] [[1.] [0.] [0.] [0.] [0.]]]
query_masks = tf.tile(query_masks3, [1, 1, tf.shape(keys)[1]])   # (6, 5, 5)
#[[[1. 1. 1. 1. 1.]  [[1. 1. 1. 1. 1.]  [[1. 1. 1. 1. 1.]  [[1. 1. 1. 1. 1.]  [[1. 1. 1. 1. 1.]  [[1. 1. 1. 1. 1.]
#  [1. 1. 1. 1. 1.]   [0. 0. 0. 0. 0.]   [1. 1. 1. 1. 1.]   [0. 0. 0. 0. 0.]   [1. 1. 1. 1. 1.]   [0. 0. 0. 0. 0.]
#  [1. 1. 1. 1. 1.]   [0. 0. 0. 0. 0.]   [1. 1. 1. 1. 1.]   [0. 0. 0. 0. 0.]   [1. 1. 1. 1. 1.]   [0. 0. 0. 0. 0.]  
#  [0. 0. 0. 0. 0.]   [0. 0. 0. 0. 0.]   [0. 0. 0. 0. 0.]   [0. 0. 0. 0. 0.]   [0. 0. 0. 0. 0.]   [0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0.]]  [0. 0. 0. 0. 0.]]  [0. 0. 0. 0. 0.]]  [0. 0. 0. 0. 0.]]  [0. 0. 0. 0. 0.]]  [0. 0. 0. 0. 0.]]]

outputs2 = outputs1 * query_masks  # (6, 5, 5)

# Weighted sum
outputs3 = tf.matmul(outputs2, V_) # (6, 5, 5)

# restore shape
outputs4 = tf.concat(tf.split(outputs3, num_heads, axis=0), axis=2)  # (2, 5, 9)

# Residual connection
outputs5 = outputs4 + queries      # (2, 5, 9)
outputs6 = normalize(outputs5)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())
  routputs, routputs5, routputs6=sess.run([outputs, outputs5, outputs6])

print('====outputs5=====')
print(outputs5.shape)
print(routputs5)
print('====outputs6=====')
print(outputs6.shape)
print(routputs6)
