from hyperlpr_py3 import finemapping_vertical
from hyperlpr_py3 import e2emodel
from hyperlpr_py3 import finemapping_vertical
from hyperlpr_py3 import finemapping_vertical
import tensorflow as tf

model = finemapping_vertical.getModel()
model.load_weights("./model/model12.h5")
tf.saved_model.save(model, "./lpr_model/finemapping_vertical")
pred_model = e2emodel.construct_model("./model/ocr_plate_all_w_rnn_2.h5")
tf.saved_model.save(model, "./lpr_model/ocr_plate_all_w_rnn_2")
