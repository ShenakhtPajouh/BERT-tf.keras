import re

import tensorflow as tf
import bert
import original_bert


def builder(config_path, ckpt_path, name=None, session=None):
    config = bert.BertConfig.from_json_file(config_path)
    graph = tf.Graph()
    with graph.as_default():
        inputs = tf.placeholder(shape=[None, None], dtype=tf.int32)
        mask = tf.placeholder(shape=[None, None], dtype=tf.int32)
        model = original_bert.BertModel(config=config, is_training=False, input_ids=inputs, input_mask=mask,
                                        use_one_hot_embeddings=False)

    conf = tf.ConfigProto(device_count={'GPU': 0})
    sess = tf.Session(config=conf, graph=graph)

    with graph.as_default():
        saver = tf.train.Saver()
        saver.restore(sess, ckpt_path)
        gb = tf.global_variables()
        official_bert_variables = sess.run(gb)

    def _f():
        model_2 = bert.BertModel(config=config, trainable=True, name=name)
        inputs = tf.placeholder(shape=[None, None], dtype=tf.int32)
        mask = tf.placeholder(shape=[None, None], dtype=tf.int32)
        y = model_2(inputs, input_mask=mask)
        assigns = []
        variables = model_2.variables
        transformer_variables = sorted(zip((var.name.lower() for var in variables), variables), key=lambda t: t[0])
        off_bert_pairs = sorted(zip((var.name.lower() for var in gb), official_bert_variables), key=lambda t: t[0])
        for i in range(len(transformer_variables)):
            assigns.append(tf.assign(transformer_variables[i][1], off_bert_pairs[i][1]))
        return model_2, assigns

    if tf.executing_eagerly() and session is None:
        model_2, _ = _f()
    else:
        if session is None:
            session = tf.get_default_session()
        with session.graph.as_default():
            model_2, assigns = _f()
        _ = session.run(assigns)
    return model_2


def custom_builder(config_path, ckpt_path,
                   corresponding_blocks={1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12},
                   name=None, session=None):
    """Create and use from arbitrary sequence of BERTs blocks
    corresponding_blocks: a dictionary of len of new model's length, for each entry key corresponds to new model and
     value to checkpoint model. values must be in the range of num_hidden_layers of checkpoint model
     example:
     1.
     {1:1,2:2,3,3}
     2.
     {1:1,2:3,3:3,4:7}
    """

    config = bert.BertConfig.from_json_file(config_path)
    num_blocks = config.num_hidden_layers
    graph = tf.Graph()
    with graph.as_default():
        inputs = tf.placeholder(shape=[None, None], dtype=tf.int32)
        mask = tf.placeholder(shape=[None, None], dtype=tf.int32)
        model = original_bert.BertModel(config=config, is_training=False, input_ids=inputs, input_mask=mask,
                                        use_one_hot_embeddings=False)

    conf = tf.ConfigProto(device_count={'GPU': 0})
    sess = tf.Session(config=conf, graph=graph)

    with graph.as_default():
        saver = tf.train.Saver()
        saver.restore(sess, ckpt_path)
        gb = tf.global_variables()
        official_bert_variables = sess.run(gb)

    def _f():
        config.num_hidden_layers = len(corresponding_blocks)
        model_2 = bert.BertModel(config=config, trainable=True, name=name)
        inputs = tf.placeholder(shape=[None, None], dtype=tf.int32)
        mask = tf.placeholder(shape=[None, None], dtype=tf.int32)
        y = model_2(inputs, input_mask=mask)
        assigns = []
        variables = model_2.variables

        def atoi(text):
            return int(text) if text.isdigit() else text

        transformer_variables = sorted(zip((var.name.lower() for var in variables), variables),
                                       key=lambda t: [atoi(c) for c in re.split(r'(\d+)', t[0])])
        off_bert_pairs = sorted(zip((var.name.lower() for var in gb), official_bert_variables),
                                key=lambda t: [atoi(c) for c in re.split(r'(\d+)', t[0])])

        embedding_variables = 5
        layer_variables = 16
        pooling_variables = 2

        off_bert_pairs_by_block = [off_bert_pairs[0:pooling_variables]]
        for j in range(num_blocks):
            off_bert_pairs_by_block += [off_bert_pairs[
                                        pooling_variables + j * layer_variables:pooling_variables + (
                                                j + 1) * layer_variables]]
        off_bert_pairs_by_block += [off_bert_pairs[-embedding_variables:]]

        transformer_variables_by_block = [transformer_variables[0:pooling_variables]]
        for j in range(len(corresponding_blocks)):
            transformer_variables_by_block += [transformer_variables[
                                               pooling_variables + j * layer_variables:pooling_variables + (
                                                       j + 1) * layer_variables]]
        transformer_variables_by_block += [transformer_variables[-embedding_variables:]]

        for j in range(len(corresponding_blocks) + 2):
            if j == 0:
                for k in range(pooling_variables):
                    assigns.append(tf.assign(transformer_variables_by_block[j][k][1],
                                             off_bert_pairs_by_block[j][k][1]))
            elif j == len(corresponding_blocks) + 2 - 1:
                for k in range(embedding_variables):
                    assigns.append(tf.assign(transformer_variables_by_block[j][k][1],
                                             off_bert_pairs_by_block[j][k][1]))
            else:
                for k in range(layer_variables):
                    assigns.append(tf.assign(transformer_variables_by_block[j][k][1],
                                             off_bert_pairs_by_block[corresponding_blocks[j]][k][1]))

        return model_2, assigns

    if tf.executing_eagerly() and session is None:
        model_2, _ = _f()
    else:
        if session is None:
            session = tf.get_default_session()
        with session.graph.as_default():
            model_2, assigns = _f()
        _ = session.run(assigns)
    return model_2

