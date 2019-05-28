import os
import tensorflow as tf
import bert
import original_bert as modeling
import argparse


def build(bert_config_path, bert_checkpoint_path, session=None, name=None):
    if name is None:
        name = "bert"
    config = bert.BertConfig.from_json_file(bert_config_path)
    conf = tf.ConfigProto(device_count={'GPU': 0})
    graph = tf.Graph()
    with graph.as_default():
        x = tf.ones(shape=(1, 1), dtype=tf.int32)
        original_bert = modeling.BertModel(config, is_training=False, input_ids=x,
                                           use_one_hot_embeddings=False, scope="bert")
        saver = tf.train.Saver()
        sess = tf.Session(config=conf, graph=graph)
        saver.restore(sess=sess, save_path=bert_checkpoint_path)
        original_weights = {v.name: v for v in tf.global_variables()}
    orig_weights = sorted(list(original_weights))
    original_weights = [original_weights[v] for v in orig_weights]
    original_weights = sess.run(original_weights)
    sess.close()
    eager = session is None and tf.executing_eagerly()
    def _build():
        x = tf.ones(shape=(1, 1), dtype=tf.int32)
        model = bert.BertModel(config=config, name=name)
        y = model(x, training=False, pooling=True, use_one_hot_embedding=False)
        weights = {v.name: v for v in model.weights}
        _weights = sorted(list(weights))
        weights = [weights[v] for v in _weights]
        assigns = [u.assign(v) for u, v in zip(weights, original_weights)]
        return model, assigns
    if eager:
        model, _ = _build()
    else:
        if session is None:
            try:
                session = tf.get_default_session()
            except:
                raise Exception("No session is given and there is no default session")
        with session.graph.as_default():
            model, assigns = _build()
        session.run(assigns)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="config file path", required=True)
    parser.add_argument("--checkpoint", help="checkpoint file path", required=True)
    parser.add_argument("--target", help="target h5 file path", required=True)
    args = parser.parse_args()
    conf = tf.ConfigProto(device_count={'GPU': 0})
    tf.enable_eager_execution(config=conf)
    model = build(args.config, args.checkpoint, name="bert")
    model.save_weights(args.target, save_format="h5")






