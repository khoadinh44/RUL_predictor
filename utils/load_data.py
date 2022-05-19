import os
import numpy as np
import pandas as pd

def load_RUL_data(data_dir):
    vibra_data = []
    for filename in os.listdir(data_dir):
        if filename[:3] == 'acc':
            dataset = pd.read_csv(os.path.join(data_dir, filename), sep=',')  # (20479, 4)
            dataset = np.array(dataset.iloc[:, 4])  # 4 for x axis, 5 for y axis
            dataset = dataset.reshape(1, -1)
            if vibra_data == []:
              vibra_data = dataset
            else:
              vibra_data = np.concatenate((vibra_data, dataset))
    # 'Hour', 'Minute', 'Second', 'microsecond', 'Horiz', 'vert'
    vibra_data = np.array(vibra_data)
    vibra_data = np.expand_dims(vibra_data, axis=1)
    return vibra_data

# Train data-------------------------------------------------------------------------
Bearing1_1_path = '/content/drive/Shareddrives/newpro112233/company/PRONOSTIA/data/Training_set/Learning_set/Bearing1_1'
Bearing1_2_path = '/content/drive/Shareddrives/newpro112233/company/PRONOSTIA/data/Training_set/Learning_set/Bearing1_2'
Bearing2_1_path = '/content/drive/Shareddrives/newpro112233/company/PRONOSTIA/data/Training_set/Learning_set/Bearing2_1'
Bearing2_2_path = '/content/drive/Shareddrives/newpro112233/company/PRONOSTIA/data/Training_set/Learning_set/Bearing2_2'
Bearing3_1_path = '/content/drive/Shareddrives/newpro112233/company/PRONOSTIA/data/Training_set/Learning_set/Bearing3_1'
Bearing3_2_path = '/content/drive/Shareddrives/newpro112233/company/PRONOSTIA/data/Training_set/Learning_set/Bearing3_2'
print('\n Training data'+'-'*100)
Bearing3_2_data = load_RUL_data(Bearing3_2_path)
Bearing3_1_data = load_RUL_data(Bearing3_1_path)
Bearing2_2_data = load_RUL_data(Bearing2_2_path)
Bearing2_1_data = load_RUL_data(Bearing2_1_path)
Bearing1_2_data = load_RUL_data(Bearing1_2_path)
Bearing1_1_data = load_RUL_data(Bearing1_1_path)

Conditions_1_train = np.concatenate((Bearing1_1_data, Bearing1_2_data))
Conditions_2_train = np.concatenate((Bearing2_1_data, Bearing2_2_data))
Conditions_3_train = np.concatenate((Bearing3_1_data, Bearing3_2_data))
train_data = np.concatenate((Conditions_1_train, Conditions_2_train, Conditions_3_train))
print(f'\n Shape of train data: {train_data.shape}')


# Test data---------------------------------------------------------------------------
Bearing1_3_path = '/content/drive/Shareddrives/newpro112233/company/PRONOSTIA/data/Test_set/Test_set/Bearing1_3'
Bearing1_4_path = '/content/drive/Shareddrives/newpro112233/company/PRONOSTIA/data/Test_set/Test_set/Bearing1_4'
Bearing1_5_path = '/content/drive/Shareddrives/newpro112233/company/PRONOSTIA/data/Test_set/Test_set/Bearing1_5'
Bearing1_6_path = '/content/drive/Shareddrives/newpro112233/company/PRONOSTIA/data/Test_set/Test_set/Bearing1_6'
Bearing1_7_path = '/content/drive/Shareddrives/newpro112233/company/PRONOSTIA/data/Test_set/Test_set/Bearing1_7'
Bearing2_3_path = '/content/drive/Shareddrives/newpro112233/company/PRONOSTIA/data/Test_set/Test_set/Bearing2_3'
Bearing2_4_path = '/content/drive/Shareddrives/newpro112233/company/PRONOSTIA/data/Test_set/Test_set/Bearing2_4'
Bearing2_5_path = '/content/drive/Shareddrives/newpro112233/company/PRONOSTIA/data/Test_set/Test_set/Bearing2_5'
Bearing2_6_path = '/content/drive/Shareddrives/newpro112233/company/PRONOSTIA/data/Test_set/Test_set/Bearing2_6'
Bearing2_7_path = '/content/drive/Shareddrives/newpro112233/company/PRONOSTIA/data/Test_set/Test_set/Bearing2_7'
Bearing3_3_path = '/content/drive/Shareddrives/newpro112233/company/PRONOSTIA/data/Test_set/Test_set/Bearing3_3'
print('\n Test data'+'-'*100)
Bearing1_3_data = load_RUL_data(Bearing1_3_path)
Bearing1_4_data = load_RUL_data(Bearing1_4_path)
Bearing1_5_data = load_RUL_data(Bearing1_5_path)
Bearing1_6_data = load_RUL_data(Bearing1_6_path)
Bearing1_7_data = load_RUL_data(Bearing1_7_path)
Bearing2_3_data = load_RUL_data(Bearing2_3_path)
Bearing2_4_data = load_RUL_data(Bearing2_4_path)
Bearing2_5_data = load_RUL_data(Bearing2_5_path)
Bearing2_6_data = load_RUL_data(Bearing2_6_path)
Bearing2_7_data = load_RUL_data(Bearing2_7_path)
Bearing3_3_data = load_RUL_data(Bearing3_3_path)

Conditions_1_test = np.concatenate((Bearing1_3_data, Bearing1_4_data, Bearing1_5_data, Bearing1_6_data, Bearing1_7_data))
Conditions_2_test = np.concatenate((Bearing2_3_data, Bearing2_4_data, Bearing2_5_data, Bearing2_6_data, Bearing2_7_data))
Conditions_3_test = Bearing3_3_data
test_data = np.concatenate((Conditions_1_test, Conditions_2_test, Conditions_3_test))
print(f'\n Shape of test data: {test_data.shape}')
