from model.autoencoder import autoencoder_model
from model.cnn import cnn_1d_model, cnn_2d_model
from model.dnn import dnn_model
from utils.load_data import train_data, train_label, test_data, test_label
from utils.tools import recall_m, precision_m, f1_m, to_onehot
import argparse
import numpy as np
import os

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--input_shape', default=2559, type=int)
    parser.add_argument('--num_classes', default=3, type=str, help='condition 1, condition 2, condition 3')
    parser.add_argument('--model', default='dnn', type=str, help='dnn, cnn_1d, cnn_2d, autoencoder')
    parser.add_argument('--save_dir', default='/content/drive/Shareddrives/newpro112233/company/PRONOSTIA', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

def main(opt, train_data, train_label, test_data, test_label):
  train_label = to_onehot(train_label)
  test_label  = to_onehot(test_label)

  if opt.model == 'dnn':
    train_data = np.squeeze(train_data)
    test_data  = np.squeeze(test_data)
    network = dnn_model(opt)
  if opt.model == 'cnn_1d':
    train_data = np.squeeze(train_data)
    test_data  = np.squeeze(test_data)
    network = cnn_1d_model(opt)
  if opt.model == 'cnn_2d':
    train_data = np.squeeze(train_data)
    test_data  = np.squeeze(test_data)
    network = cnn_1d_model(opt, [128, 128, 1])
  if opt.model == 'autoencoder':
    network = autoencoder_model(train_data)
  
  network.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['acc', f1_m, precision_m, recall_m]) # loss='mse'
  network.summary()
  history = network.fit(train_data, train_label,
                      epochs     = opt.epochs,
                      batch_size = opt.batch_size,
                      validation_data = (test_data, test_label))
  _, test_acc,  test_f1_m,  test_precision_m,  test_recall_m  = network.evaluate(test_data, test_label, verbose=0)
  print(f'\n Score in test set: \n Accuracy: {test_acc}, F1: {test_f1_m}, Precision: {test_precision_m}, recall: {test_recall_m}' )
  network.save(os.path.join(opt.save_dir, opt.model))

if __name__ == '__main__':
  opt = parse_opt()
  main(opt, train_data, train_label, test_data, test_label)
