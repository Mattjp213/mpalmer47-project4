
import numpy

class DataUtils(object):

    @staticmethod
    def get_results_directory_name():
        return "Results"

    @staticmethod
    def load_feature_labels_from_file(file_name):

        file = open(file_name + ".csv")

        feature_labels = []

        for line in file:
            feature_labels = line.split(",")
            break

        return feature_labels

    @staticmethod
    def load_data_to_nd_array(file_name):

        file = open(file_name + ".csv")

        number_of_lines = 0
        number_of_columns = 0
        in_header = True

        for line in file:
            if in_header:
                number_of_columns = len(line.split(","))
                in_header = False
            else:
                number_of_lines +=1

        file.close()

        data_from_file = numpy.ndarray(shape=(number_of_lines, number_of_columns))

        file = open(file_name + ".csv")

        in_header = True
        string_value = 1
        string_values_dict = {}
        row_num = 0

        for line in file:
            if in_header:
                in_header = False
            else:
                line_parts = line.split(",")
                for i in range(len(line_parts)):
                    try:
                        value = float(line_parts[i])
                    except ValueError:
                        string_in_line = line_parts[i].strip()
                        if string_in_line in string_values_dict:
                            value = string_values_dict[string_in_line]
                        else:
                            value = string_value
                            string_values_dict[string_in_line] = value
                            string_value += 1
                    data_from_file[row_num, i] = value
                row_num += 1

        file.close()

        return data_from_file

    @staticmethod
    def pull_fold_out_of_data_set(dataset: numpy.ndarray, fold_start_index : int, fold_end_index: int):

        if fold_start_index < 0:
            fold_start_index = 0

        if fold_end_index >= len(dataset) - 1:
            fold_end_index = (len(dataset) - 1)

        fold = dataset[fold_start_index:fold_end_index + 1, :]

        rows_in_remainder = []

        if fold_start_index > 0:
            for i in range(fold_start_index):
                rows_in_remainder.append(i)

        if fold_end_index < len(dataset) - 1:
            for i in range(fold_end_index + 1, len(dataset)):
                rows_in_remainder.append(i)

        remainder = dataset[rows_in_remainder, :]

        return fold, remainder

    @staticmethod
    def get_k_folds_training_and_test_sets(dataset: numpy.ndarray, k: int):

        if k < 1:
            k = 1
        elif k > 10:
            k = 10

        number_of_points_per_test_fold = int(len(dataset)/k)

        data_folds = []

        for i in range(k):
            new_fold = []
            test_fold_start_index = i * number_of_points_per_test_fold
            if i < k - 1:
                test_fold_end_index = (test_fold_start_index + number_of_points_per_test_fold) - 1
            else:
                test_fold_end_index = len(dataset) - 1
            test_fold, training_fold = DataUtils.pull_fold_out_of_data_set(dataset, test_fold_start_index, test_fold_end_index)
            new_fold.append(training_fold)
            new_fold.append(test_fold)
            data_folds.append(new_fold)

        return data_folds

    @staticmethod
    def write_error_per_training_set_size(output_file_name, errors, training_set_sizes):

        output_file = open(output_file_name, "w")

        output_file.write("TrainingSetPercentSize,Error\n")

        for i in range(len(errors)):
            output_file.write(str(training_set_sizes[i]) + "," + str(errors[i]) + "\n")

        output_file.close()

        print("File written: " + output_file_name)

    @staticmethod
    def write_convergence_diffs(output_file_name, convergence_diffs):

        output_file = open(output_file_name, "w")

        output_file.write("ConvergenceDiffs\n")

        for i in range(len(convergence_diffs)):
            output_file.write(str(convergence_diffs[i]) + "\n")

        output_file.close()

    @staticmethod
    def file_already_exists(file_name):

        try:
            existing_file = open(file_name, "r")
            existing_file.close()

        except FileNotFoundError:
            return False

        return True








