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
