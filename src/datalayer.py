from os import listdir
from os.path import isfile, join
import numpy as np

import util.fileutil as fu

class stockdata(object):
    """Stock Data
    """

    def __init__(self, datapath, suffix=".csv"):
        """constructor
        create like and dislike data
        Args:
            datapath: data path.
                'like' and 'dislike' directories should be under the path.
                'like' directory contains stock data
                'dislike' directory contains stock data 
            suffix: data file extension. default is ".csv"
        """
        self.input_data = {
            "like": self.createData(datapath + "/like", suffix, 0),
            "dislike": self.createData(datapath + "/dislike", suffix, 1)
        }

    def getCrossValidationInput(self, num_groups, group):
        """getCrossValidationInpu
        Args:
            num_groups: seperate entire data to num_groups
            group: group number. Start from 0 to num)groups-1
        Return:
            dictionary with train and test data
        """
        train_data = []
        test_data = []
        for i in range(num_groups):
            for key in self.input_data:
                if(i != group):
                    train_data = np.append(train_data, self.input_data[key][i::num_groups])
                else:
                    test_data = np.append(test_data, self.input_data[key][i::num_groups])
        train_data = np.array(np.random.permutation(train_data))
        test_data = np.array(np.random.permutation(test_data))
        return {"train":train_data, "test":test_data}

    def createData(self, path, suffix, label):
        """createData
        Args:
            path: directory path
            suffix: data file suffix
            label: 0 => [0, 1], 1 => [1, 0]
        Return:
            list suffled data with label
            example_data_structure)
            [dictionary{features:data, label:label}, dictionary{features:data, label:label}, ...]
        """
        data_set = self.readData(path, suffix)
        data_set = self.parseData(data_set)
        data_set = self.addLabel(data_set, label)
        data_set = np.random.permutation(data_set)
        return data_set

    def readData(self, path, suffix):
        """ readData
        Args:
            path: directory path
            suffix: data file suffix
        Return:
            data set
        """
        files = fu.getAllFileNames(path, suffix)
        return np.array(fu.readDataFiles(files, ","))

    def parseData(self, data_set):
        """parseData
        skip header and convert string value to float
        Args:
            data_set: three dimensional numpy array
        Return:
            modified data_set without header
        """
        result = []
        for data in data_set:
            # convert data to numpy array
            data = np.array(data)

            # remove header, date, adj close then convert string to float then flip
            result.append(np.flipud(data[1:,1:-1].astype(np.float)))
        return np.array(result)

    def addLabel(self, data_set, label):
        """ addLabel
        add label on data
        Args:
            data_set: three dimensional numpy array
            label: 0 => [0, 1], 1 => [1, 0]
        Return:
            data_set with label
        """
        result = []
        for data in data_set:
            data_dictionary = {
                "features": data,
                "label": [0,1] if label == 0 else [1,0]
            }
            result.append(data_dictionary)
        return np.array(result)

