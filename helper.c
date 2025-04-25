#include "helper.h"

// returns the product of a scalar and a vector
float* vec_multiply(float scalar, float* vec, int size)
{
	float* res = (float*)malloc(size * sizeof(float));

	for (int i = 0; i < size; ++i) {
		res[i] = scalar * vec[i];
	}
	return res;
}

// returns the vector difference: (vec1 - vec2)
float* vec_diff(float* vec1, float* vec2, int size)
{
	float* diff = (float*)malloc(size * sizeof(float));

	for (int i = 0; i < size; ++i) {
		diff[i] = vec1[i] - vec2[i];
	}
	return diff;
}

// returns dot/inner product of two vectors
float vec_inner(float* vec1, float* vec2, int size)
{
	float res = 0.0f;

	for (int i = 0; i < size; ++i) {
		res += vec1[i] * vec2[i];
	}
	return res;
}

// returns outer product of two vectors, i.e., (vec1 vec2^T)
float** vec_outer(float* vec1, float* vec2, int vec1_size, int vec2_size)
{
	float** res = (float**)malloc(vec1_size * sizeof(float*));

	for (int i = 0; i < vec1_size; ++i) {
		float* vals = (float*)malloc(vec2_size * sizeof(float));

		for (int j = 0; j < vec2_size; ++j) {
			vals[j] = vec1[i] * vec2[j];
		}
		res[i] = vals;
	}
	return res;
}

// returns the hadamard product of two vectors
float* vec_hadamard(float* vec1, float* vec2, int size)
{
	float* res = (float*)malloc(size * sizeof(float));

	for (int i = 0; i < size; ++i) {
		res[i] = vec1[i] * vec2[i];
	}
	return res;

}

// returns the matrix obtained by the difference of two matrices: i.e., (M1 - M2)
float** mat_diff(float** matrix1, float** matrix2, int rows, int cols)
{
	float** res = (float**)malloc(rows * sizeof(float*));

	for (int i = 0; i < rows; ++i) {
		float* vals = (float*)malloc(cols * sizeof(float));

		for (int j = 0; j < cols; ++j) {
			vals[j] = matrix1[i][j] - matrix2[i][j];
		}
		res[i] = vals;
	}
	return res;
}

// returns the product of a scalar and a matrix
float** mat_multiply(float scalar, float** matrix, int rows, int cols)
{
	float** res = (float**)malloc(rows * sizeof(float*));

	for (int i = 0; i < rows; ++i) {
		float* vals = (float*)malloc(cols * sizeof(float));

		for (int j = 0; j < cols; ++j) {
			vals[j] = scalar * matrix[i][j];
		}
		res[i] = vals;
	}
	return res;
}


// performs matrix multiplication of a matrix with a vector, and returns the resulting vector
// if multiply_transposed_matrix is TRUE, returns (M^T v), else returns (M v)
float* matmul(float** matrix, float* vec, int vec_size, int output_vec_size, Bool multiply_transposed_matrix)
{
	float* output_vec = (float*)malloc(output_vec_size * sizeof(float));
	memset(output_vec, 0.0f, output_vec_size * sizeof(float));

	for (int i = 0; i < output_vec_size; ++i) {
		for (int j = 0; j < vec_size; ++j) {
			float m = multiply_transposed_matrix ? matrix[j][i] : matrix[i][j];
			output_vec[i] += m * vec[j];
		}
	}
	return output_vec;
}

// clears the memory allocated to the matrix
void free_mat(float** matrix, int rows)
{
	for (int i = 0; i < rows; ++i) {
		free(matrix[i]);	
	}
	free(matrix);
}

// the rectified linear activation function
float ReLU(float val)
{
	return (val > 0.0f) ? val : 0.0f;
}

// returns the vector obtained by the derivative of ReLU acting on a layer
float* differentiated_relu(float* vec, int size)
{
	float* diff_activated_layer = (float*)malloc(size * sizeof(float));

	for (int i = 0; i < size; ++i) {
		diff_activated_layer[i] = vec[i] > 0 ? 1.0f : 0.0f;
	}
	return diff_activated_layer;
}

// returns the gradient of the loss function, according to the particular type of loss
float* loss_gradient(float* predictions, float* actual, int num_nodes, LOSS_FUNC_TYPE loss_function_type)
{
	switch (loss_function_type) {
		case LF_MSE: return mse_loss_gradient(predictions, actual, num_nodes);
		default: break;
	}
	return (float*)0;	// NULL
}

// mean square error loss function
float mse_loss(float* predictions, float* actual, int num_nodes)
{
	float res = 0.0f;

	for (int i = 0; i < num_nodes; ++i) {
		float curr_term = (actual[i] - predictions[i]) * (actual[i] - predictions[i]);
		curr_term /= num_nodes;
		res += curr_term;
	}
	return res;
}

// gradient of the mean square error loss function
float* mse_loss_gradient(float* predictions, float* actual, int num_nodes)
{
	return vec_diff(predictions, actual, num_nodes);
}

// reads a .csv file, and returns the dataset object for it
// each row in the file is treated as a single vector
DataSet* get_dataset_from_csv(const char* file_path)
{
	FILE* file = fopen(file_path, "r");
	assert(!!file);

	float** data = NULL;
	int row_count = 0;
	int col_count = 0;
	char line[MAX_CSV_LINE_LENGTH];
	Bool is_first_row = TRUE;

	while (fgets(line, sizeof(line), file)) {
		char* line_copy = _strdup(line);
		char* context = NULL;
		char* token = STRTOK(line_copy, ",", &context);

		// On first row, count columns
		if (is_first_row) {
			while (token) {
				col_count++;
				token = STRTOK(NULL, ",", &context);
			}
			rewind(file);
			free(line_copy);
			is_first_row = FALSE;
			continue;
		}

		float* row_data = (float*)malloc(col_count * sizeof(float));
		fgets(line, sizeof(line), file);
		line_copy = _strdup(line);
		token = STRTOK(line_copy, ",", &context);

		for (int j = 0; j < col_count; j++) {
			row_data[j] = token ? atof(token) : 0.0f;
			token = STRTOK(NULL, ",", &context);
		}

		float** temp = (float**)realloc(data, (row_count + 1) * sizeof(float*));
		data = temp;
		data[row_count++] = row_data;
		free(line_copy);
	}
	fclose(file);

	DataSet* dataset = (DataSet*)malloc(sizeof(DataSet));
	dataset->dataset = data;
	dataset->set_size = row_count;
	dataset->vector_size = col_count;
	dataset->normalized_dataset = NULL;
	dataset->mean_features = NULL;
	dataset->std_deviation_features = NULL;
	normalize_dataset(dataset);
	return dataset;
}

// sets the normalized set for a given dataset (as per Z-score normalization)
// set_size: the number of items in the dataset
// vector_size: the size of each vector in the set
void normalize_dataset(DataSet* dataset)
{
	float** normalized_dataset = (float**)malloc(dataset->set_size * sizeof(float*));

	float* mean_features = (float*)malloc(dataset->vector_size * sizeof(float));
	float* std_deviation_features = (float*)malloc(dataset->vector_size * sizeof(float));

	for (int i = 0; i < dataset->set_size; ++i) {
		normalized_dataset[i] = (float*)malloc(dataset->vector_size * sizeof(float));
	}

	// calculate means
	for (int j = 0; j < dataset->vector_size; ++j) {
		mean_features[j] = 0.0f;
		for (int i = 0; i < dataset->set_size; ++i) {
			mean_features[j] += dataset->dataset[i][j];
		}
		mean_features[j] /= dataset->set_size;
	}

	// calculate std deviations
	for (int j = 0; j < dataset->vector_size; ++j) {
		std_deviation_features[j] = 0.0f;
		for (int i = 0; i < dataset->set_size; ++i) {
			std_deviation_features[j] += (dataset->dataset[i][j] - mean_features[j]) * (dataset->dataset[i][j] - mean_features[j]);
		}
		std_deviation_features[j] = sqrtf(std_deviation_features[j] / dataset->set_size);
	}

	// get normalized dataset values
	for (int i = 0; i < dataset->set_size; ++i) {
		for (int j = 0; j < dataset->vector_size; ++j) {
			normalized_dataset[i][j] = (dataset->dataset[i][j] - mean_features[j]) / std_deviation_features[j];
		}
	}
	dataset->normalized_dataset = normalized_dataset;
	dataset->mean_features = mean_features;
	dataset->std_deviation_features = std_deviation_features;
}

float* get_denormalized_vector(float* normalized_vector, DataSet* dataset)
{
	float* denorm_vec = (float*)malloc(dataset->vector_size * sizeof(float));

	for (int i = 0; i < dataset->vector_size; ++i) {
		denorm_vec[i] = dataset->std_deviation_features[i] * normalized_vector[i] + dataset->mean_features[i];
	}
	return denorm_vec;
}