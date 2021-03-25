import cv2
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

if __name__ == '__main__':
    with tf.Graph().as_default():
        with tf.Session() as session:
            tf.keras.backend.set_session(session)
            tf.keras.backend.set_learning_phase(0)
            model = tf.keras.models.load_model('model.h5', compile=False)

            from tensorflow.python.framework.graph_util import convert_variables_to_constants

            graph = session.graph
            with graph.as_default():
                freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(None or []))
                output_names = [out.op.name for out in model.outputs]
                output_names = output_names or []
                output_names += [v.op.name for v in tf.global_variables()]
                input_graph_def = graph.as_graph_def()
                for node in input_graph_def.node:
                    node.device = ""
                frozen_graph = convert_variables_to_constants(session, input_graph_def, output_names, freeze_var_names)

                tf.io.write_graph(
                    graph_or_graph_def=frozen_graph,
                    logdir=".",
                    name="model.pb",
                    as_text=False
                )

                tf.io.write_graph(
                    graph_or_graph_def=frozen_graph,
                    logdir=".",
                    name="model.pbtxt",
                    as_text=True
                )

                net = cv2.dnn.readNet(r'model.pb')
                for layer_name in net.getLayerNames():
                    print(layer_name)
