import tensorflow as tf


def builder(session, config_path, ckpt_path, seq_length):
    with open(config_path) as f:
        conf = json.loads(f.read())
    config = modeling.BertConfig(**conf)
    graph = tf.Graph()
    with graph.as_default():
        inputs = tf.placeholder(shape=[None, None], dtype=tf.int32)
        mask = tf.placeholder(shape=[None, None], dtype=tf.int32)
        model = modeling.BertModel(config=config, is_training=False, input_ids=inputs, input_mask=mask,
                                   use_one_hot_embeddings=False)

    conf = tf.ConfigProto(device_count={'GPU': 0})
    conf.gpu_options.allow_growth = True
    sess = tf.Session(config=conf, graph=graph)

    with graph.as_default():
        saver = tf.train.Saver()
        saver.restore(sess, ckpt_path)
        gb = tf.global_variables()
        official_bert_variables = sess.run(gb)

    with session.graph.as_default():
        model_2 = bert.BertModel(config=config, is_training=False)
        assigns = []
        variables = model_2.variables
        transformer_variables = sorted(zip((var.name.lower() for var in variables), variables), key=lambda t: t[0])
        off_bert_pairs = sorted(zip((var.name.lower() for var in gb), official_bert_variables), key=lambda t: t[0])
        for i in range(len(transformer_variables)):
            assign.append(tf.assign(transformer_variables[i][1], off_bert_pairs[i][1]))
        session.run(assigns)
    return model_2
