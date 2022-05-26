from main import parse_opt
from utils.load_rul_data import train_data_rul, train_label_rul, test_data_rul, test_label_rul

opt = parse_opt()
main(opt, train_data_rul, train_label_rul, test_data_rul, test_label_rul)
