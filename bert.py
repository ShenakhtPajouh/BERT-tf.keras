from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import math
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


class LayerNormalization(tf.keras.layers.Layer):

    def build(self, input_shape):
        self.beta = self.add_weight(name="beta", shape=input_shape[-1:], initializer=init_ops.zeros_initializer(),
                                    dtype=tf.float32)
        self.gamma = self.add_weight(name="gamma", shape=input_shape[-1:], initializer=init_ops.ones_initializer(),
                                     dtype=tf.float32)
        super().build(input_shape)

    def __init__(self, trainable=True, name=None):
        super().__init__(name=name, trainable=trainable)
        self.beta = None
        self.gamma = None

    def call(self, inputs, activation_fn=None, begin_norm_axis=1, begin_params_axis=-1):
        inputs = ops.convert_to_tensor(inputs)
        inputs_shape = inputs.shape
        inputs_rank = inputs_shape.ndims
        if inputs_rank is None:
            raise ValueError('Inputs %s has undefined rank.' % inputs.name)
        if begin_norm_axis < 0:
            begin_norm_axis = inputs_rank + begin_norm_axis
        if begin_params_axis >= inputs_rank or begin_norm_axis >= inputs_rank:
            raise ValueError('begin_params_axis (%d) and begin_norm_axis (%d) '
                             'must be < rank(inputs) (%d)' %
                             (begin_params_axis, begin_norm_axis, inputs_rank))
        params_shape = inputs_shape[begin_params_axis:]
        if not params_shape.is_fully_defined():
            raise ValueError(
                'Inputs %s: shape(inputs)[%s:] is not fully defined: %s' %
                (inputs.name, begin_params_axis, inputs_shape))
        # Calculate the moments on the last axis (layer activations).
        norm_axes = list(range(begin_norm_axis, inputs_rank))
        mean, variance = tf.nn.moments(inputs, norm_axes, keep_dims=True)
        # Compute layer normalization using the batch_normalization function.
        variance_epsilon = 1e-12
        outputs = tf.nn.batch_normalization(
            inputs,
            mean,
            variance,
            offset=self.beta,
            scale=self.gamma,
            variance_epsilon=variance_epsilon)
        outputs.set_shape(inputs_shape)
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs

    def __call__(self, inputs, activation_fn=None, begin_norm_axis=1, begin_params_axis=-1, **kwargs):
        return super().__call__(inputs=inputs, activation_fn=activation_fn, begin_norm_axis=begin_norm_axis,
                                begin_params_axis=begin_params_axis, **kwargs)


class AttentionLayer(tf.keras.Model):

    def __init__(self, num_attention_heads=1, size_per_head=512, query_act=None,
                 initializer_range=0.02,
                 attention_probs_dropout_prob=0.0, value_act=None,
                 key_act=None,
                 trainable=True,
                 name=None):
        super().__init__(name=name)
        # `query_layer` = [B*F, N*H]
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

    def call(self, inputs, attention_probs_dropout_prob=0.0, attention_mask=None, do_return_2d_tensor=False,
             batch_size=None,
             from_seq_length=None,
             to_seq_length=None, training=False):
        def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                                 seq_length, width):
            output_tensor = tf.reshape(
                input_tensor, [batch_size, seq_length, num_attention_heads, width])

            output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
            return output_tensor

        from_shape = get_shape_list(inputs[0], expected_rank=[2, 3])
        to_shape = get_shape_list(inputs[1], expected_rank=[2, 3])
        if len(from_shape) != len(to_shape):
            raise ValueError(
                "The rank of `from_tensor` must match the rank of `to_tensor`.")
        if len(from_shape) == 3:
            batch_size = from_shape[0]
            from_seq_length = from_shape[1]
            to_seq_length = to_shape[1]
        elif len(from_shape) == 2:
            if batch_size is None or from_seq_length is None or to_seq_length is None:
                raise ValueError(
                    "When passing in rank 2 tensors to attention_layer, the values "
                    "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                    "must all be specified.")
        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   F = `from_tensor` sequence length
        #   T = `to_tensor` sequence length
        #   N = `num_attention_heads`
        #   H = `size_per_head`

        from_tensor_2d = reshape_to_matrix(inputs[0])
        to_tensor_2d = reshape_to_matrix(inputs[1])

        query_layer = self.query_layer(from_tensor_2d)
        key_layer = self.key_layer(to_tensor_2d)
        value_layer = self.value_layer(to_tensor_2d)

        # `query_layer` = [B, N, F, H]
        query_layer = transpose_for_scores(query_layer, batch_size,
                                           self.num_attention_heads, from_seq_length,
                                           self.size_per_head)

        # `key_layer` = [B, N, T, H]
        key_layer = transpose_for_scores(key_layer, batch_size, self.num_attention_heads,
                                         to_seq_length, self.size_per_head)

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        # `attention_scores` = [B, N, F, T]
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        attention_scores = tf.multiply(attention_scores,
                                       1.0 / math.sqrt(float(self.size_per_head)))

        if attention_mask is not None:
            # `attention_mask` = [B, 1, F, T]
            attention_mask = tf.expand_dims(attention_mask, axis=[1])

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_scores += adder

        # Normalize the attention scores to probabilities.
        # `attention_probs` = [B, N, F, T]
        attention_probs = tf.nn.softmax(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        if training:
            attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

        # `value_layer` = [B, T, N, H]
        value_layer = tf.reshape(
            value_layer,
            [batch_size, to_seq_length, self.num_attention_heads, self.size_per_head])

        # `value_layer` = [B, N, T, H]
        value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

        # `context_layer` = [B, N, F, H]
        context_layer = tf.matmul(attention_probs, value_layer)

        # `context_layer` = [B, F, N, H]
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

        if do_return_2d_tensor:
            # `context_layer` = [B*F, N*H]
            context_layer = tf.reshape(
                context_layer,
                [batch_size * from_seq_length, self.num_attention_heads * self.size_per_head])
        else:
            # `context_layer` = [B, F, N*H]
            context_layer = tf.reshape(
                context_layer,
                [batch_size, from_seq_length, self.num_attention_heads * self.size_per_head])

        return context_layer

    def __call__(self, inputs, attention_probs_dropout_prob=0.0, attention_mask=None,
                 do_return_2d_tensor=False,
                 batch_size=None,
                 from_seq_length=None,
                 to_seq_length=None, training=False, **kwargs):
        return super().__call__(inputs, attention_probs_dropout_prob=attention_probs_dropout_prob,
                                attention_mask=attention_mask,
                                do_return_2d_tensor=do_return_2d_tensor,
                                batch_size=batch_size,
                                from_seq_length=from_seq_length,
                                to_seq_length=to_seq_length, training=training, **kwargs)


def gelu(input_tensor):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
    input_tensor: float Tensor to perform activation.
    Returns:
    `input_tensor` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
    return input_tensor * cdf


class TransformerModel(tf.keras.Model):

    def __init__(self, num_attention_heads=12, num_hidden_layers=12, hidden_size=768, initializer_range=0.02,
                 attention_probs_dropout_prob=0.1, hidden_dropout_prob=None, intermediate_act_fn=gelu,
                 intermediate_size=3072, trainable=True, name=None):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.num_hidden_size = num_hidden_layers
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        attention_head_size = int(hidden_size / num_attention_heads)
        self.attention_layers = []
        self.attention_dense_layers = []
        self.intermediate_dense_layers = []
        self.attention_layer_norms = []
        self.output_dense_layers = []
        self.output_layer_norms = []
        for layer_idx in range(num_hidden_layers):
            attention_head = AttentionLayer(num_attention_heads=num_attention_heads,
                                            size_per_head=attention_head_size,
                                            initializer_range=initializer_range,
                                            attention_probs_dropout_prob=attention_probs_dropout_prob,
                                            name=("layer_%d" % layer_idx) + "/attention/self"
                                            )
            self.attention_layers.append(attention_head)
            dense_layer = tf.keras.layers.Dense(
                hidden_size,
                kernel_initializer=create_initializer(initializer_range),
                name=("layer_%d" % layer_idx) + "/attention/output/Dense"
            )
            self.attention_dense_layers.append(dense_layer)
            norm_layer = LayerNormalization(name=("layer_%d" % layer_idx) + "/attention/output/LayerNorm")
            self.attention_layer_norms.append(norm_layer)
            intermediate_dense = tf.keras.layers.Dense(
                intermediate_size,
                activation=intermediate_act_fn,
                kernel_initializer=create_initializer(initializer_range),
                name=("layer_%d" % layer_idx) + "/intermediate/Dense"
            )
            self.intermediate_dense_layers.append(intermediate_dense)
            output_dense_layer = tf.keras.layers.Dense(
                hidden_size,
                kernel_initializer=create_initializer(initializer_range),
                name=("layer_%d" % layer_idx) + "/output/Dense"
            )
            self.output_dense_layers.append(output_dense_layer)
            norm_layer = LayerNormalization(name=("layer_%d" % layer_idx) + "/output/LayerNorm")
            self.output_layer_norms.append(norm_layer)
        self.trainable = trainable

    def call(self, inputs, attention_mask=None, attention_probs_dropout_prob=None, hidden_dropout_prob=None,
             do_return_all_layers=False):
        if hidden_dropout_prob is None:
            hidden_dropout_prob = self.hidden_dropout_prob
        if attention_probs_dropout_prob is None:
            attention_probs_dropout_prob = self.attention_probs_dropout_prob
        input_shape = get_shape_list(inputs, expected_rank=3)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        input_width = input_shape[2]
        # The Transformer performs sum residuals on all layers so the input needs
        # to be the same as the hidden size.
        if input_width != self.hidden_size:
            raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                             (input_width, self.hidden_size))
        # The Transformer performs sum residuals on all layers so the input needs
        # to be the same as the hidden size.
        if input_width != self.hidden_size:
            raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                             (input_width, self.hidden_size))
        # We keep the representation as a 2D tensor to avoid re-shaping it back and
        # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
        # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
        # help the optimizer.
        prev_output = reshape_to_matrix(inputs)

        all_layer_outputs = []
        for i in range(self.num_hidden_size):
            layer_input = prev_output
            attention_heads = []
            attention_head = self.attention_layers[i]((layer_input, layer_input),
                                                      batch_size=batch_size,
                                                      from_seq_length=seq_length,
                                                      to_seq_length=seq_length,
                                                      do_return_2d_tensor=True,
                                                      attention_mask=attention_mask,
                                                      attention_probs_dropout_prob=attention_probs_dropout_prob)
            attention_heads.append(attention_head)
            attention_output = None
            if len(attention_heads) == 1:
                attention_output = attention_heads[0]
            else:
                # In the case where we have other sequences, we just concatenate
                # them to the self-attention head before the projection.
                attention_output = tf.concat(attention_heads, axis=-1)

            # Run a linear projection of `hidden_size` then add a residual
            # with `layer_input`.
            attention_output = self.attention_dense_layers[i](attention_output)
            attention_output = dropout(attention_output, self.hidden_dropout_prob)
            attention_output = self.attention_layer_norms[i](inputs=attention_output + layer_input, begin_norm_axis=-1,
                                                             begin_params_axis=-1)
            intermediate_output = self.intermediate_dense_layers[i](attention_output)
            layer_output = self.output_dense_layers[i](intermediate_output)
            layer_output = dropout(layer_output, hidden_dropout_prob)
            layer_output = self.output_layer_norms[i](layer_output + attention_output)
            prev_output = layer_output
            all_layer_outputs.append(layer_output)

        if do_return_all_layers:
            final_outputs = []
            for layer_output in all_layer_outputs:
                final_output = reshape_from_matrix(layer_output, input_shape)
                final_outputs.append(final_output)
            return final_outputs
        else:
            final_output = reshape_from_matrix(prev_output, input_shape)
            return final_output

    def __call__(self, inputs, attention_mask=None, attention_probs_dropout_prob=None, hidden_dropout_prob=None,
                 do_return_all_layers=False, **kwargs):
        return super().__call__(inputs=inputs, attention_mask=attention_mask,
                                attention_probs_dropout_prob=attention_probs_dropout_prob,
                                hidden_dropout_prob=hidden_dropout_prob, do_return_all_layers=do_return_all_layers,
                                **kwargs)


class EmbeddingLookup(tf.keras.layers.Layer):

    def build(self, input_shape):
        self.embedding_table = self.add_weight(name=self.word_embedding_name,
                                               shape=[self.vocab_size, self.embedding_size],
                                               initializer=create_initializer(self.initializer_range),
                                               dtype=tf.float32)
        super().build(input_shape)

    def __init__(self, vocab_size, embedding_size=128, word_embedding_name="word_embeddings", initializer_range=0.02,
                 trainable=True, name=None):
        super().__init__(name=name, trainable=trainable)
        self.embedding_table = None
        self.word_embedding_name = word_embedding_name
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.initializer_range = initializer_range

    def call(self, inputs, use_one_hot_embeddings=False):
        if inputs.shape.ndims == 2:
            inputs = tf.expand_dims(inputs, axis=[-1])
        if use_one_hot_embeddings:
            flat_input_ids = tf.reshape(inputs, [-1])
            one_hot_input_ids = tf.one_hot(flat_input_ids, depth=self.vocab_size)
            output = tf.matmul(one_hot_input_ids, self.embedding_table)
        else:
            output = tf.nn.embedding_lookup(self.embedding_table, inputs)
        input_shape = get_shape_list(inputs)
        output = tf.reshape(output,
                            input_shape[0:-1] + [input_shape[-1] * self.embedding_size])
        return output, self.embedding_table

    def __call__(self, inputs, use_one_hot_embeddings=False, **kwargs):
        return super().__call__(inputs=inputs, use_one_hot_embeddings=use_one_hot_embeddings, **kwargs)


class EmbeddingPostprocessor(tf.keras.layers.Layer):

    def __init__(self, initializer_range=0.02, token_type_vocab_size=16,
                 token_type_embedding_name="token_type_embeddings", position_embedding_name="position_embeddings",
                 max_position_embeddings=512, use_token_type=False, use_position_embeddings=True, dropout_prob=0.1,
                 trainable=True,
                 name=None):
        super().__init__(name=name, trainable=trainable)
        self.token_type_table = None
        self.full_position_embeddings = None
        self.max_position_embeddings = max_position_embeddings
        self.dropout_prob = dropout_prob
        self.use_token_type = use_token_type
        self.use_position_embeddings = use_position_embeddings
        self.max_position_embedding = max_position_embeddings
        self.token_type_embedding_name = token_type_embedding_name
        self.position_embedding_name = position_embedding_name
        self.token_type_vocab_size = token_type_vocab_size
        self.initializer_range = initializer_range

    def build(self, input_shape):
        if self.use_token_type:
            self.token_type_table = self.add_weight(name=self.token_type_embedding_name,
                                                    shape=[self.token_type_vocab_size, input_shape[2].value],
                                                    initializer=create_initializer(self.initializer_range),
                                                    dtype=tf.float32)
        if self.use_position_embeddings:
            assert_op = tf.assert_less_equal(input_shape[1].value, self.max_position_embeddings)
            with tf.control_dependencies([assert_op]):
                self.full_position_embeddings = self.add_weight(name=self.position_embedding_name,
                                                                shape=[self.max_position_embedding,
                                                                       input_shape[2].value],
                                                                initializer=create_initializer(self.initializer_range),
                                                                dtype=tf.float32)
        super().build(input_shape)

    def call(self, inputs, token_type_ids=None, dropout_prob=None):
        input_shape = get_shape_list(inputs, expected_rank=3)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        width = input_shape[2]

        output = inputs

        if self.use_token_type:
            if token_type_ids is None:
                raise ValueError("`token_type_ids` must be specified if"
                                 "`use_token_type` is True.")
            # This vocab will be small so we always do one-hot here, since it is always
            # faster for a small vocabulary.
            flat_token_type_ids = tf.reshape(token_type_ids, [-1])
            one_hot_ids = tf.one_hot(flat_token_type_ids, depth=self.token_type_vocab_size)
            token_type_embeddings = tf.matmul(one_hot_ids, self.token_type_table)
            token_type_embeddings = tf.reshape(token_type_embeddings,
                                               [batch_size, seq_length, width])
            output += token_type_embeddings

        if self.use_position_embeddings:
            # Since the position embedding table is a learned variable, we create it
            # using a (long) sequence length `max_position_embeddings`. The actual
            # sequence length might be shorter than this, for faster training of
            # tasks that do not have long sequences.
            #
            # So `full_position_embeddings` is effectively an embedding table
            # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
            # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
            # perform a slice.
            position_embeddings = tf.slice(self.full_position_embeddings, [0, 0],
                                           [seq_length, -1])
            num_dims = len(output.shape.as_list())

            # Only the last two dimensions are relevant (`seq_length` and `width`), so
            # we broadcast among the first dimensions, which is typically just
            # the batch size.
            position_broadcast_shape = []
            for _ in range(num_dims - 2):
                position_broadcast_shape.append(1)
            position_broadcast_shape.extend([seq_length, width])
            position_embeddings = tf.reshape(position_embeddings,
                                             position_broadcast_shape)
            output += position_embeddings
        return output

    def __call__(self, inputs, token_type_ids=None,
                 dropout_prob=None, **kwargs):
        return super().__call__(inputs=inputs, token_type_ids=token_type_ids, dropout_prob=dropout_prob, **kwargs)


class Embeddings(tf.keras.Model):

    def __init__(self, config, name=None, trainable=True):
        super().__init__(name=name)
        self.config = config
        # Perform embedding lookup on the word ids.
        self.embedding_lookup = EmbeddingLookup(
            vocab_size=config.vocab_size,
            embedding_size=config.hidden_size,
            initializer_range=config.initializer_range,
            word_embedding_name="word_embeddings",
            name="zz",
            trainable=trainable)
        self.embedding_postprocessor = EmbeddingPostprocessor(use_token_type=True,
                                                              token_type_vocab_size=config.type_vocab_size,
                                                              token_type_embedding_name="token_type_embeddings",
                                                              use_position_embeddings=True,
                                                              position_embedding_name="position_embeddings",
                                                              initializer_range=config.initializer_range,
                                                              max_position_embeddings=config.max_position_embeddings,
                                                              dropout_prob=config.hidden_dropout_prob, name="z"
                                                              )

        self.layer_norm = LayerNormalization(name="LayerNorm")
        self.trainable = trainable

    def call(self, inputs, token_type_ids=None, use_one_hot_embeddings=True, hidden_dropout_prob=None, training=None,
             mask=None):
        input_shape = get_shape_list(inputs, expected_rank=2)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        if token_type_ids is None:
            token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)
        # Perform embedding lookup on the word ids.
        (embedding_output, embedding_table) = self.embedding_lookup(inputs,
                                                                    use_one_hot_embeddings=use_one_hot_embeddings)
        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        embedding_output = self.embedding_postprocessor(embedding_output, token_type_ids=token_type_ids)
        embedding_output = self.layer_norm(embedding_output, begin_norm_axis=-1, begin_params_axis=-1)
        if hidden_dropout_prob is None:
            hidden_dropout_prob = self.config.hidden_dropout_prob
        embedding_output = dropout(embedding_output, hidden_dropout_prob)

        return embedding_output

    def __call__(self, inputs, token_type_ids=None, use_one_hot_embeddings=True, hidden_dropout_prob=None, **kwargs):
        return super().__call__(inputs=inputs, token_type_ids=token_type_ids,
                                use_one_hot_embeddings=use_one_hot_embeddings, hidden_dropout_prob=hidden_dropout_prob)


class BertModel(tf.keras.models.Model):

    def __init__(self, config, name=None, trainable=True):
        super().__init__(name=name)
        self.config = config
        config = copy.deepcopy(config)
        self.embedding = Embeddings(config)
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
            name="pooler/Dense")
        self.trainable = trainable

    def call(self, inputs, training=False, input_mask=None, token_type_ids=None, use_one_hot_embeddings=True,
             hidden_dropout_prob=None,
             attention_probs_dropout_prob=None, pooled=False, get_all_encoder_layers=False):
        if not training:
            self.config.hidden_dropout_prob = 0.0
            self.config.attention_probs_dropout_prob = 0.0
        input_shape = get_shape_list(inputs, expected_rank=2)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

        embedding_output = self.embedding(inputs=inputs, token_type_ids=token_type_ids,
                                          use_one_hot_embeddings=use_one_hot_embeddings,
                                          hidden_dropout_prob=hidden_dropout_prob)

        attention_mask = create_attention_mask_from_input_mask(
            inputs, input_mask)

        all_encoder_layers = self.transformer(embedding_output, attention_mask=attention_mask,
                                              attention_probs_dropout_prob=attention_probs_dropout_prob,
                                              hidden_dropout_prob=hidden_dropout_prob,
                                              do_return_all_layers=True)
        sequence_output = all_encoder_layers[-1]
        first_token_tensor = tf.squeeze(sequence_output[:, 0:1, :], axis=1)
        pooled_output = self.dense_pooler(first_token_tensor)
        if pooled:
            return pooled_output
        elif get_all_encoder_layers:
            return all_encoder_layers
        else:
            return sequence_output

    def __call__(self, inputs, training=True, input_mask=None, token_type_ids=None, use_one_hot_embeddings=True,
                 hidden_dropout_prob=None,
                 attention_probs_dropout_prob=None, pooled=False, get_all_encoder_layers=False, **kwargs):
        return super().__call__(inputs=inputs, training=training, input_mask=input_mask,
                                token_type_ids=token_type_ids,
                                use_one_hot_embeddings=use_one_hot_embeddings, hidden_dropout_prob=hidden_dropout_prob,
                                attention_probs_dropout_prob=attention_probs_dropout_prob, pooled=pooled,
                                get_all_encoder_layers=get_all_encoder_layers, **kwargs)


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.
    Args:
      tensor: A tf.Tensor object to find the shape of.
      expected_rank: (optional) int. The expected rank of `tensor`. If this is
        specified and the `tensor` has a different rank, and exception will be
        thrown.
      name: Optional name of the tensor for the error message.
    Returns:
      A list of dimensions of the shape of tensor. All static dimensions will
      be returned as python integers, and dynamic dimensions will be returned
      as tf.Tensor scalars.
    """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size=128,
                     initializer_range=0.02,
                     word_embedding_name="word_embeddings",
                     use_one_hot_embeddings=False):
    """Looks up words embeddings for id tensor.
    Args:
      input_ids: int32 Tensor of shape [batch_size, seq_length] containing word
        ids.
      vocab_size: int. Size of the embedding vocabulary.
      embedding_size: int. Width of the word embeddings.
      initializer_range: float. Embedding initialization range.
      word_embedding_name: string. Name of the embedding table.
      use_one_hot_embeddings: bool. If True, use one-hot method for word
        embeddings. If False, use `tf.nn.embedding_lookup()`. One hot is better
        for TPUs.
    Returns:
      float Tensor of shape [batch_size, seq_length, embedding_size].
    """
    # This function assumes that the input is of shape [batch_size, seq_length,
    # num_inputs].
    #
    # If the input is a 2D tensor of shape [batch_size, seq_length], we
    # reshape to [batch_size, seq_length, 1].
    if input_ids.shape.ndims == 2:
        input_ids = tf.expand_dims(input_ids, axis=[-1])

    embedding_table = tf.get_variable(
        name=word_embedding_name,
        shape=[vocab_size, embedding_size],
        initializer=create_initializer(initializer_range))

    if use_one_hot_embeddings:
        flat_input_ids = tf.reshape(input_ids, [-1])
        one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
        output = tf.matmul(one_hot_input_ids, embedding_table)
    else:
        output = tf.nn.embedding_lookup(embedding_table, input_ids)

    input_shape = get_shape_list(input_ids)

    output = tf.reshape(output,
                        input_shape[0:-1] + [input_shape[-1] * embedding_size])
    return output, embedding_table


def reshape_to_matrix(input_tensor):
    """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
    ndims = input_tensor.shape.ndims
    if ndims < 2:
        raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                         input_tensor.shape)
    if ndims == 2:
        return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor


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


def reshape_from_matrix(output_tensor, orig_shape_list):
    """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
    if len(orig_shape_list) == 2:
        return output_tensor

    output_shape = get_shape_list(output_tensor)

    orig_dims = orig_shape_list[0:-1]
    width = output_shape[-1]

    return tf.reshape(output_tensor, orig_dims + [width])


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.
    Args:
      tensor: A tf.Tensor to check the rank of.
      expected_rank: Python integer or list of integers, expected rank.
      name: Optional name of the tensor for the error message.
    Raises:
      ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


def create_attention_mask_from_input_mask(from_tensor, to_mask):
    """Create 3D attention mask from a 2D tensor mask.
    Args:
      from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
      to_mask: int32 Tensor of shape [batch_size, to_seq_length].
    Returns:
      float Tensor of shape [batch_size, from_seq_length, to_seq_length].
    """
    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]

    to_shape = get_shape_list(to_mask, expected_rank=2)
    to_seq_length = to_shape[1]

    to_mask = tf.cast(
        tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

    # We don't assume that `from_tensor` is a mask (although it could be). We
    # don't actually care if we attend *from* padding tokens (only *to* padding)
    # tokens so we create a tensor of all ones.
    #
    # `broadcast_ones` = [batch_size, from_seq_length, 1]
    broadcast_ones = tf.ones(
        shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

    # Here we broadcast along two dimensions to create the mask.
    mask = broadcast_ones * to_mask

    return mask


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
