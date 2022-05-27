from main import parse_opt
from utils.load_rul_data import train_data_rul, train_label_rul, test_data_rul, test_label_rul

opt = parse_opt()
def Predict(data, model):
  if opt.condition_train:
    train_label = to_onehot(train_label)
    test_label  = to_onehot(test_label)

  if model == 'dnn':
    data = data.reshape(len(data), int(opt.input_shape*2))
    network = dnn_model(opt)
  if model == 'cnn_1d':
    network = cnn_1d_model(opt, training=None)
  if opt.model == 'resnet_cnn_2d':
    inputs = Input(shape=[128, 128, 2])
    output = resnet_152(opt)(inputs, training=None)
    network = Model(inputs, output)
  if opt.model == 'cnn_2d':
    network = cnn_2d_model(opt, [128, 128, 2])
  if opt.model == 'autoencoder':
    network = autoencoder_model(train_data)
  if opt.model == 'lstm':
    network = lstm_model(opt)
  
  if opt.load_weight:
    if os.path.exists(os.path.join(opt.save_dir, opt.model)):
      print(f'\nLoad weight: {os.path.join(opt.save_dir, opt.model)}\n')
      network.load_weights(os.path.join(opt.save_dir, opt.model))
      
  y_pred = network.predict(data)
  return y_pred


