from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import math
import numpy as np
import six
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops


class BertConfig(object):
    """Configuration for `BertModel`."""

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02):
        """Constructs BertConfig.
        Args:
          vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
          hidden_size: Size of the encoder layers and the pooler layer.
          num_hidden_layers: Number of hidden layers in the Transformer encoder.
          num_attention_heads: Number of attention heads for each attention layer in
            the Transformer encoder.
          intermediate_size: The size of the "intermediate" (i.e., feed-forward)
            layer in the Transformer encoder.
          hidden_act: The non-linear activation function (function or string) in the
            encoder and pooler.
          hidden_dropout_prob: The dropout probability for all fully connected
            layers in the embeddings, encoder, and pooler.
          attention_probs_dropout_prob: The dropout ratio for the attention
            probabilities.
          max_position_embeddings: The maximum sequence length that this model might
            ever be used with. Typically set this to something large just in case
            (e.g., 512 or 1024 or 2048).
          type_vocab_size: The vocabulary size of the `token_type_ids` passed into
            `BertModel`.
          initializer_range: The stdev of the truncated_normal_initializer for
            initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with tf.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))


def gelu(x):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
    input_tensor: float Tensor to perform activation.
    Returns:
    `input_tensor` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


class LayerNormalization(tf.keras.layers.Layer):


    def __init__(self, trainable=True, name=None):
        super().__init__(name=name, trainable=trainable)
        self.beta = None
        self.gamma = None

    def build(self, input_shape):
        self.beta = self.add_weight(name="beta", shape=input_shape[-1:], initializer=init_ops.zeros_initializer(),
                                    dtype=tf.float32)
        self.gamma = self.add_weight(name="gamma", shape=input_shape[-1:], initializer=init_ops.ones_initializer(),
                                     dtype=tf.float32)
        super().build(input_shape)

    def call(self, inputs, activation_fn=None, axis=-1, epsilon=10e-12):
        mean, variance = tf.nn.moments(inputs, axis, keep_dims=True)
        rdev = tf.rsqrt(variance + epsilon)
        x = (inputs - mean) * rdev
        output = x * self.gamma + self.beta
        return output

    def __call__(self, inputs, activation_fn=None, axis=-1, epsilon=10e-12):
        return super().__call__(inputs=inputs, activation_fn=activation_fn,
                                axis=axis, epsilon=epsilon)

def attention(query, key, value, multi_heads, mask=None,
              one_sided=False, attention_dropout=None):
    """
    query: A tensor of shape [batch_size, query_num, dim]
    key: A tensor of shape [batch_size, key_num, dim]
    value: A tensor of shape [batch_size, key_num, value_dim]
    query_mask: a boolean tensor of shape [batch_size, query_num]
    key_mask: a boolean tensor of shape [batch_size, key_num]
    one_sided: a boolean which determines if the attention mechanism is one sided or not
    attention_dropout: dropout for attention coefficients
    """
    batch_size, query_num, dim = get_tensor_shape(query)
    _, key_num, value_dim = get_tensor_shape(value)
    # construct mask
    if mask is None and not one_sided:
        mask = None
    else:
        if one_sided:
            q_rng = tf.range(query_num, type=tf.int32)
            q_rng = tf.expand_dims(q_rng, 1)
            k_rng = tf.range(key_num, dtype=tf.int32)
            one_sided_mask = tf.greater_equal(q_rng, k_rng)
            one_sided_mask = tf.reshape(one_sided_mask, [1, 1, query_num, key_num])
        if mask is not None:
            mask = tf.reshape(mask, [batch_size, 1, 1, key_num])
            if one_sided:
                mask = tf.logical_and(mask, one_sided_mask)
        else:
            mask = one_sided_mask
    new_dim = dim // multi_heads
    query = tf.reshape(query, [batch_size, query_num, multi_heads, new_dim])
    key = tf.reshape(key, [batch_size, key_num, multi_heads, new_dim])
    value = tf.reshape(value, [batch_size, key_num, multi_heads, value_dim // multi_heads])
    query = tf.transpose(query, [0, 2, 1, 3])
    key = tf.transpose(key, [0, 2, 3, 1])
    value = tf.transpose(value, [0, 2, 1, 3])
    coefficients = tf.matmul(query, key) / tf.sqrt(float(new_dim))
    if mask is not None:
        mask = tf.cast(mask, tf.float32)
        coefficients = coefficients * mask + (1-mask) * -100
    coefficients = tf.nn.softmax(coefficients, -1)
    coefficients = dropout(coefficients, attention_dropout)
    result = tf.matmul(coefficients, value)
    result = tf.transpose(result, [0, 2, 1, 3])
    result = tf.reshape(result, [batch_size, query_num, value_dim])
    return result

class AttentionLayer(tf.keras.layers.Layer):

    def __init__(self, num_attention_heads=1, size_per_head=512, query_act=None,
                 initializer_range=0.02,
                 attention_probs_dropout_prob=0.0, value_act=None,
                 key_act=None,
                 trainable=True,
                 name=None):
        super().__init__(name=name)
        # `query_layer` = [B*F, N*H]
        self.attention_size = num_attention_heads * size_per_head
        self.query_layer = tf.keras.layers.Dense(
            num_attention_heads * size_per_head,
            activation=query_act,
            name="query",
            kernel_initializer=create_initializer(initializer_range)
        )
        # `key_layer` = [B*T, N*H]
        self.key_layer = tf.keras.layers.Dense(
            num_attention_heads * size_per_head,
            activation=key_act,
            name="key",
            kernel_initializer=create_initializer(initializer_range),
        )
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # `value_layer` = [B*T, N*H]
        self.value_layer = tf.keras.layers.Dense(
            num_attention_heads * size_per_head,
            activation=value_act,
            name="value",
            kernel_initializer=create_initializer(initializer_range)
        )
        self.size_per_head = size_per_head
        self.num_attention_heads = num_attention_heads
        self.trainable = trainable

    def call(self, inputs, mask=None, attention_probs_dropout_prob=None, training=False):
        """
        inputs: a tensor of shape [batch_size, seq_length, dim]
        mask: a boolean tensor of shape [batch_size, seq_length]
        attention_probs_dropout_prob: dropout use for attention mechanism
        """
        query = self.query_layer(inputs)
        key = self.key_layer(inputs)
        value = self.value_layer(inputs)
        if training:
            if attention_probs_dropout_prob is None:
                attention_dropout = self.attention_probs_dropout_prob
            else:
                attention_dropout = attention_probs_dropout_prob
        else:
            attention_dropout = None
        result = attention(query, key, value, self.num_attention_heads,
                           mask=mask, attention_dropout=attention_dropout)
        return result

    def __call__(self, inputs, mask=None, attention_probs_dropout_prob=None, training=False):
        return super().__call__(
            inputs=inputs,
            mask=mask,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            training=training
        )

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, num_attention_heads=12, hidden_size=768, initializer_range=0.02,
                 attention_probs_dropout_prob=0.1, hidden_dropout_prob=None,
                 intermediate_act_fn=gelu,
                 intermediate_size=3072,
                 name=None):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        attention_head_size = hidden_size // num_attention_heads
        self.attention_layer = AttentionLayer(
            num_attention_heads=num_attention_heads,
            size_per_head=attention_head_size,
            initializer_range=initializer_range,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            name="attention/self"
        )
        self.attention_dense_layer = tf.keras.layers.Dense(
            hidden_size,
            kernel_initializer=create_initializer(initializer_range),
            name="attention/output/dense"
        )
        self.attention_layer_norm = LayerNormalization(name="attention/output/LayerNorm")
        self.intermediate_dense_layer = tf.keras.layers.Dense(
                intermediate_size,
                activation=intermediate_act_fn,
                kernel_initializer=create_initializer(initializer_range),
                name="intermediate/dense"
            )
        self.output_dense_layer = tf.keras.layers.Dense(
                hidden_size,
                kernel_initializer=create_initializer(initializer_range),
                name="output/dense"
            )
        self.output_layer_norm = LayerNormalization(name="output/LayerNorm")

    def call(self, inputs, mask=None, attention_probs_dropout_prob=None,
             hidden_dropout_prob=None,
             training=False):
        if training:
            if attention_probs_dropout_prob is None:
                attention_probs_dropout_prob = self.attention_probs_dropout_prob
            if hidden_dropout_prob is None:
                hidden_dropout_prob = self.hidden_dropout_prob
        if mask is None:
            mask = tf.ones(shape=get_tensor_shape(inputs)[:-1], dtype=tf.bool)
        attention_output = self.attention_layer(inputs, mask, attention_probs_dropout_prob, training)
        attention_output = tf.boolean_mask(attention_output, mask)
        inputs = tf.boolean_mask(inputs, mask)
        attention_output = self.attention_dense_layer(attention_output)
        attention_output = dropout(attention_output, hidden_dropout_prob)
        attention_output = self.attention_layer_norm(attention_output + inputs)
        intermediate_output = self.intermediate_dense_layer(attention_output)
        layer_output = self.output_dense_layer(intermediate_output)
        layer_output = dropout(layer_output, hidden_dropout_prob)
        layer_output = self.output_layer_norm(layer_output + attention_output)
        indices = tf.where(mask)
        shape = get_tensor_shape(mask) + get_tensor_shape(layer_output)[-1:]
        layer_output = tf.scatter_nd(indices, layer_output, shape)
        return layer_output

    def __call__(self, inputs, mask=None,
                 attention_probs_dropout_prob=None, hidden_dropout_prob=None,
                 training=False):
        return super().__call__(inputs=inputs, mask=mask,
                                attention_probs_dropout_prob=attention_probs_dropout_prob,
                                hidden_dropout_prob=hidden_dropout_prob,
                                training=training)


class TransformerModel(tf.keras.Model):
    def __init__(self, num_attention_heads=12, num_hidden_layers=12, hidden_size=768, initializer_range=0.02,
                 attention_probs_dropout_prob=0.1, hidden_dropout_prob=None,
                 intermediate_act_fn=gelu,
                 intermediate_size=3072, trainable=True, name=None):
        super().__init__(name=name)
        self.trainable = trainable
        self.num_hidden_layers = num_hidden_layers
        self.blocks = []
        for layer_idx in range(num_hidden_layers):
            block = TransformerBlock(
                num_attention_heads=num_attention_heads,
                hidden_size=hidden_size,
                initializer_range=initializer_range,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                hidden_dropout_prob=hidden_dropout_prob,
                intermediate_act_fn=intermediate_act_fn,
                intermediate_size=intermediate_size,
                name="layer_%d" % layer_idx
            )
            self.blocks.append(block)

    def call(self, inputs, mask=None,
             output_blocks=None,
             attention_probs_dropout_prob=None,
             hidden_dropout_prob=None,
             training=False):
        """
        Args:
            inputs: a tensor of shape [batch_size, seq_length, dim]
            mask: a boolean tensor of shape [batch_size, seq_length]
            output_layers: a list. will returns the output of transfromer blocks in that list
        Returns:
            if output_blocks is None returns the last output as a tensor, else it will returns a
            dictionary: {block_index: layer_output}

        """

        if output_blocks is None:
            max_block = self.num_hidden_layers - 1
            flag = False
        else:
            flag = True
            _output_blocks = []
            for x in output_blocks:
                if x >= 0:
                    _output_blocks.append(x)
                else:
                    _output_blocks.append(self.num_hidden_layers + x)
            max_block = max(_output_blocks)

        if flag:
            outputs = {}
        output = inputs
        for idx in range(max_block + 1):
            output = self.blocks[idx](
                inputs=output,
                mask=mask,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                hidden_dropout_prob=hidden_dropout_prob,
                training=training
            )
            if flag:
                if idx in _output_blocks:
                    outputs[idx] = output
        if flag:
            return outputs
        else:
            return output

    def __call__(self, inputs, mask=None,
             output_blocks=None,
             attention_probs_dropout_prob=None,
             hidden_dropout_prob=None,
             training=False):
        return super().__call__(
            inputs=inputs, mask=mask,
            output_blocks=output_blocks,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            hidden_dropout_prob=hidden_dropout_prob,
            training=training
        )


class Embedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_size=128,
                 use_token_type=True,
                 token_type_vocab_size=2,
                 use_positional_embedding=True,
                 max_positional_embedding=512,
                 initializer_range=0.02,
                 name=None,
                 word_embedding_name=None,
                 token_type_embedding_name=None,
                 positional_embedding_name=None,
                 trainable=True
                 ):
        super().__init__(name=name, trainable=trainable)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.use_token_type = use_token_type
        self.token_type_vocab_size = token_type_vocab_size
        self.use_positional_embedding = use_positional_embedding
        self.max_positional_embedding = max_positional_embedding
        self.initializer_range = initializer_range
        self.word_embedding_name = word_embedding_name
        if self.word_embedding_name is None:
            self.word_embedding_name = "word_embeddings"
        self.token_type_embedding_name = token_type_embedding_name
        if self.token_type_embedding_name is None:
            self.token_type_embedding_name = "token_type_embeddings"
        self.positional_embedding_name = positional_embedding_name
        if self.positional_embedding_name is None:
            self.positional_embedding_name = "positional_embeddings"
        self.layer_norm = LayerNormalization(name="LayerNorm")

    def build(self, input_shape):
        self.embedding_table = self.add_weight(name=self.word_embedding_name,
                                               shape=[self.vocab_size, self.embedding_size],
                                               initializer=create_initializer(self.initializer_range),
                                               dtype=tf.float32)
        if self.use_token_type:
            self.token_type_table = self.add_weight(name=self.token_type_embedding_name,
                                                    shape=[self.token_type_vocab_size, self.embedding_size],
                                                    initializer=create_initializer(self.initializer_range),
                                                    dtype=tf.float32)
        if self.use_positional_embedding:
            self.full_position_embeddings = self.add_weight(name=self.positional_embedding_name,
                                                            shape=[self.max_positional_embedding, self.embedding_size],
                                                            initializer=create_initializer(self.initializer_range),
                                                            dtype=tf.float32)
        super().build(input_shape)

    def call(self, inputs, token_type_ids=None, use_one_hot_embedding=True):
        shape = get_tensor_shape(inputs)
        if use_one_hot_embedding:
            ids = tf.reshape(inputs, [shape[0] * shape[1]])
            one_hot_ids = tf.one_hot(ids, self.vocab_size)
            embedding = tf.matmul(one_hot_ids, self.embedding_table)
            embedding = tf.reshape(embedding, [shape[0], shape[1], self.embedding_size])
        else:
            embedding = tf.gather(self.embedding_table, inputs)
        if self.use_token_type:
            if token_type_ids is None:
                raise ValueError("`token_type_ids` must be specified if"
                                 "`use_token_type` is True.")
            token_ids = tf.reshape(token_type_ids, [shape[0] * shape[1]])
            one_hot_tokens = tf.one_hot(token_ids, self.token_type_vocab_size)
            token_embedding = tf.matmul(one_hot_tokens, self.token_type_table)
            token_embedding = tf.reshape(token_embedding, [shape[0], shape[1], self.embedding_size])
            embedding = embedding + token_embedding
        if self.use_positional_embedding:
            positional_embedding = self.full_position_embeddings[0:shape[1]]
            embedding = embedding + positional_embedding
        embedding = self.layer_norm(embedding)
        return embedding

    def __call__(self, inputs, token_type_ids=None, use_one_hot_embedding=True):
        return super().__call__(inputs=inputs, token_type_ids=token_type_ids,
                                use_one_hot_embedding=use_one_hot_embedding)

class BertModel(tf.keras.models.Model):

    def __init__(self, config, name=None, trainable=True):
        super().__init__(name=name)
        config = copy.deepcopy(config)
        self.config = config
        self.embedding = Embedding(
            vocab_size=config.vocab_size,
            embedding_size=config.hidden_size,
            use_token_type=True,
            token_type_vocab_size=config.type_vocab_size,
            use_positional_embedding=True,
            max_positional_embedding=config.max_position_embeddings,
            initializer_range=config.initializer_range,
            name="embedding",
            word_embedding_name="word_embedding",
            token_type_embedding_name="token_type_embedding",
            positional_embedding_name="positional_embedding"
        )
        self.transformer = TransformerModel(hidden_size=config.hidden_size,
                                            num_hidden_layers=config.num_hidden_layers,
                                            num_attention_heads=config.num_attention_heads,
                                            intermediate_size=config.intermediate_size,
                                            intermediate_act_fn=get_activation(config.hidden_act),
                                            hidden_dropout_prob=config.hidden_dropout_prob,
                                            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                                            initializer_range=config.initializer_range,
                                            name="encoder"
                                            )
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. We assume that this has been pre-trained
        self.dense_pooler = tf.keras.layers.Dense(
            config.hidden_size,
            activation=tf.tanh,
            kernel_initializer=create_initializer(config.initializer_range),
            name="pooler/dense")
        self.trainable = trainable
        self.dropout = config.hidden_dropout_prob

    def call(self, inputs, mask=None, token_type_ids=None, training=False,
             pooling=False, output_blocks=None, drop_out=None, attention_dropout=None,
             use_one_hot_embedding=True):
        if output_blocks is not None and pooling:
            raise ValueError("If you want to specify output_block, then you can't use pooling option")
        if token_type_ids is None:
            token_type_ids = tf.zeros_like(inputs, dtype=tf.int32)
        embedding = self.embedding(inputs, token_type_ids, use_one_hot_embedding)
        results = self.transformer(embedding, mask,
                                   output_blocks=output_blocks,
                                   attention_probs_dropout_prob=attention_dropout,
                                   hidden_dropout_prob=drop_out,
                                   training=training)
        if output_blocks is not None:
            return results
        elif not pooling:
            return results
        else:
            cls_token = results[:, 0, :]
            if training:
                if drop_out is None:
                    drop_out = self.config.hidden_dropout_prob
                cls_token = dropout(cls_token, drop_out)
            pool = self.dense_pooler(cls_token)
            return pool

    def __call__(self, inputs, mask=None, token_type_ids=None, training=False,
             pooling=False, output_blocks=None, drop_out=None, attention_dropout=None,
             use_one_hot_embedding=True):
        return super().__call__(
            inputs=inputs,
            mask=mask,
            token_type_ids=token_type_ids,
            training=training,
            pooling=pooling,
            output_blocks=output_blocks,
            drop_out=drop_out,
            attention_dropout=attention_dropout,
            use_one_hot_embedding=use_one_hot_embedding
        )


def dropout(input_tensor, dropout_prob):
    """Perform dropout.
    Args:
      input_tensor: float Tensor.
      dropout_prob: Python float. The probability of dropping out a value (NOT of
        *keeping* a dimension as in `tf.nn.dropout`).
    Returns:
      A version of `input_tensor` with dropout applied.
    """
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
    return output


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)


def get_activation(activation_string):
    """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.
    Args:
      activation_string: String name of the activation function.
    Returns:
      A Python function corresponding to the activation function. If
      `activation_string` is None, empty, or "linear", this will return None.
      If `activation_string` is not a string, it will return `activation_string`.
    Raises:
      ValueError: The `activation_string` does not correspond to a known
        activation.
    """

    # We assume that anything that"s not a string is already an activation
    # function, so we just return it.
    if not isinstance(activation_string, six.string_types):
        return activation_string

    if not activation_string:
        return None

    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return tf.nn.relu
    elif act == "gelu":
        return gelu
    elif act == "tanh":
        return tf.tanh
    else:
        raise ValueError("Unsupported activation: %s" % act)


def get_tensor_shape(input_tensor):
    input_tensor = ops.convert_to_tensor(input_tensor)
    if tf.executing_eagerly():
        return input_tensor.shape.as_list()
    static_shape = input_tensor.shape.as_list()
    dynamic_shape = tf.shape(input_tensor)
    if static_shape is None:
        return dynamic_shape
    else:
        shape = []
        dynamic_shape = tf.unstack(dynamic_shape)
        for st, dyn in zip(static_shape, dynamic_shape):
            if st is None:
                shape.append(dyn)
            else:
                shape.append(st)
    return shape
    
