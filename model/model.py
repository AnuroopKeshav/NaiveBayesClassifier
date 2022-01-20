import numpy as np
import scipy.stats as sp

class NaiveBayesClassifier:

    # Grouping entries based on different discrete resultant values
    def groupClasses(self) -> None:
        self.classes = {}

        for entry in range(len(self.resultant_vector_train)):
            if self.resultant_vector_train[entry] not in self.classes:
                self.classes[self.resultant_vector_train[entry]] = []

            self.classes[self.resultant_vector_train[entry]].append(self.feature_matrix_train[entry])

    # Calculating the mean and standard deviation of all the classes
    # Used for the Gaussian Probability Density funtion
    def defineClassMeanAndStd(self) -> None:
        self.class_mean_std = {}
        for output_class, feature_class in self.classes.items():
            self.class_mean_std[output_class] = [(np.mean(class_attr), np.std(class_attr)) for class_attr in zip(*feature_class)]

    # Funtion that iteratively finds the product of probabilities for all the classes and returns the max
    def findMaxProba(self, entry) -> np.float64:
        proba_dict = {}

        for output, class_info in self.class_mean_std.items():
            probability = 1

            for attr in range(len(class_info)):
                probability *= sp.norm(class_info[attr][0], class_info[attr][1]).pdf(entry[attr])
            
            proba_dict[output] = probability

        return max(zip(proba_dict.values(), proba_dict.keys()))[1]

    # Fitting the model with the data
    def fit(self, feature_matrix: np.array, resultant_vector: np.array) -> None:
        self.feature_matrix_train = feature_matrix
        self.resultant_vector_train = resultant_vector

        self.groupClasses()
        self.defineClassMeanAndStd()

    # Predicting for test_data
    def predict(self, feature_matrix_test: np.array) -> np.array:
        predictions = [self.findMaxProba(feature_matrix_test[entry]) for entry in range(len(feature_matrix_test))]

        return np.array(predictions)