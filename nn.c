#include "nn.h"

NN_Layer* create_nn_layer(int size)
{
	NN_Layer* layer = (NN_Layer*)malloc(sizeof(NN_Layer));
	
	float* node_vals = (float*)malloc(size * sizeof(float));
	memset(node_vals, 0.0f, size * sizeof(float));

	float* activated_node_vals = (float*)malloc(size * sizeof(float));
	memset(activated_node_vals, 0.0f, size * sizeof(float));
	
	layer->node_vals = node_vals;
	layer->activated_node_vals = activated_node_vals;
	layer->num_nodes = size;
	layer->next = NULL;
	layer->prev = NULL;
	layer->weights = NULL;
	layer->bias = NULL;
	layer->activation_function = NULL;
	layer->af_type = AF_NONE;
	return layer;
}

void free_nn_layer(NN_Layer* layer, int weight_size)
{
	if (layer->weights != NULL) {
		for (int i = 0; i < weight_size; ++i) {
			free(layer->weights[i]);
		}
		free(layer->weights);
	}
	free(layer->node_vals);
	free(layer->bias);
	free(layer);
}

float** create_weight_matrix(int rows, int cols)
{
	float** weights = (float**)malloc(rows * sizeof(float*));
	
	for (int i = 0; i < rows; ++i) {
		weights[i] = (float*)malloc(cols * sizeof(float));
		for (int j = 0; j < cols; ++j) {
			weights[i][j] = 0.2 * (float)rand() / RAND_MAX;		// random float between 0.0 and 0.2 
		}
	}
	return weights;
}

float* create_bias_vector(int size)
{
	float* bias = (float*)malloc(size * sizeof(float));
	float random_number = 0.2 * (float)rand() / RAND_MAX;		// random float between 0.0 and 0.2 
	memset(bias, random_number, size * sizeof(float));
	return bias;
}

float (*get_activation_function(ACTIVATION_FUNC_TYPE af_type))(float)
{
	switch (af_type) {
		case AF_RELU: return &ReLU;
		default: break;
	}
	return NULL;
}

// creates a neural network with given number of input nodes, output nodes, hidden layers, and learning rate.
// pass an integer array into num_hidden_layer_nodes, providing the number of nodes to be created for each hidden layer.
// pass an array activation_function_types, providing the types of activation functions to be assigned to each hidden layer, AND the output layer.
// pass the type of loss function in loss_type.
NN* create_nn(const int num_input_nodes, const int num_output_nodes, const int num_hidden_layers, const int* num_hidden_layer_nodes, const ACTIVATION_FUNC_TYPE* activation_functions, const LOSS_FUNC_TYPE loss_type, const float learning_rate)
{
	// creating the layers
	NN_Layer* input_layer = create_nn_layer(num_input_nodes);
	NN_Layer* output_layer = create_nn_layer(num_output_nodes);
	NN_Layer** hidden_layers = (NN_Layer**)malloc(num_hidden_layers * sizeof(NN_Layer*));

	for (int i = 0; i < num_hidden_layers; ++i) {
		hidden_layers[i] = create_nn_layer(num_hidden_layer_nodes[i]);
		
		if (activation_functions != NULL && activation_functions[i] != NULL) {
			hidden_layers[i]->activation_function = get_activation_function(activation_functions[i]);
			hidden_layers[i]->af_type = activation_functions[i];
		}

		if (i == 0) {
			input_layer->next = hidden_layers[i];
			hidden_layers[i]->prev = input_layer;
		}
		else {
			hidden_layers[i - 1]->next = hidden_layers[i];
			hidden_layers[i]->prev = hidden_layers[i - 1];
		}
	}
	hidden_layers[num_hidden_layers - 1]->next = output_layer;
	output_layer->prev = hidden_layers[num_hidden_layers - 1];
	output_layer->activation_function = get_activation_function(activation_functions[num_hidden_layers]);
	output_layer->af_type = activation_functions[num_hidden_layers];

	// setting weights and biases
	NN_Layer* prev_layer = input_layer;
	NN_Layer* curr_layer = prev_layer->next;
	
	while (curr_layer != NULL) {
		curr_layer->bias = create_bias_vector(curr_layer->num_nodes);
		curr_layer->weights = create_weight_matrix(prev_layer->num_nodes, curr_layer->num_nodes);
		
		prev_layer = curr_layer;
		curr_layer = curr_layer->next;
	}

	NN* nn = (NN*)malloc(sizeof(NN));
	nn->input_layer = input_layer;
	nn->output_layer = output_layer;
	nn->hidden_layers = hidden_layers;
	nn->num_hidden_layers = num_hidden_layers;
	nn->loss_type = loss_type;
	nn->learning_rate = learning_rate;
	return nn;
}

// clears the NN (and all objects contained in it) from memory
void free_nn(NN* nn)
{
	int prev_layer_weight_size = nn->input_layer->num_nodes;
	free_nn_layer(nn->input_layer, NULL);

	for (int i = 0; i < nn->num_hidden_layers; ++i) {
		int next_weight_size = nn->hidden_layers[i]->num_nodes;
		free_nn_layer(nn->hidden_layers[i], prev_layer_weight_size);
		prev_layer_weight_size = next_weight_size;
	}
	free(nn->hidden_layers);
	free_nn_layer(nn->output_layer, prev_layer_weight_size);
	free(nn);
}

void print_layer(NN_Layer* layer, Bool print_weights_and_bias)
{
	for (int i = 0; i < layer->num_nodes; ++i) {
		if (print_weights_and_bias) printf("Node %d | %.2f (Bias: %.2f) |", i + 1, layer->node_vals[i], layer->bias[i]);
		else printf("Node %d | %.2f |", i + 1, layer->node_vals[i]);

		NN_Layer* prev_layer = layer->prev;
		if (prev_layer != NULL && print_weights_and_bias) {
			for (int j = 0; j < prev_layer->num_nodes; ++j) {
				printf(" %.2f ", layer->weights[j][i]);
			}
		}
		printf("\n");
	}
	printf("\n");
}

// to print the values of all the nodes of the NN, layer by layer (on the console)
void print_nn(NN* nn)
{
	printf("--- INPUTS ---\n");
	print_layer(nn->input_layer, FALSE);
	printf("\n");
	
	for (int i = 0; i < nn->num_hidden_layers; ++i) {
		printf("--- HIDDEN LAYER (%d) ---\n", i + 1);
		print_layer(nn->hidden_layers[i], TRUE);
	}
	printf("\n");
	printf("--- OUTPUTS ---\n");
	print_layer(nn->output_layer, TRUE);
}

// forward propagate the NN: update the values of all the layers
void forward_propagate_nn(NN* nn)
{
	for (NN_Layer* curr_layer = nn->input_layer; curr_layer->next != NULL; curr_layer = curr_layer->next) {
		NN_Layer* next_layer = curr_layer->next;
		float* x = (curr_layer->af_type != AF_NONE) ? curr_layer->activated_node_vals : curr_layer->node_vals;
		float* bias = next_layer->bias;

		float* y = matmul(next_layer->weights, x, curr_layer->num_nodes, next_layer->num_nodes, TRUE);

		for (int i = 0; i < next_layer->num_nodes; ++i) {
			y[i] += bias[i];
		}
		free(next_layer->node_vals);
		next_layer->node_vals = y;
		next_layer->activated_node_vals = get_activated_layer(next_layer);
	}
}

float* get_activated_layer(NN_Layer* layer)
{
	float* activated_layer = (float*)malloc(layer->num_nodes * sizeof(float));

	for (int i = 0; i < layer->num_nodes; ++i) {
		activated_layer[i] = (layer->af_type != AF_NONE) ? layer->activation_function(layer->node_vals[i]) : layer->node_vals[i];
	}
	return activated_layer;
}

// returns the vector obtained by the derivative of the activation function acting on a layer
float* get_differentiated_activated_layer(NN_Layer* layer)
{
	switch (layer->af_type) {
		case AF_RELU: return differentiated_relu(layer->node_vals, layer->num_nodes);
		default: break;
	}
	return NULL;
}

// backward propagate the NN: update the weights for all the layers through gradient descent
void back_propagate_nn(NN* nn, float* actual_output)
{
	NN_Layer* curr_layer = nn->output_layer;
	float* loss = NULL;

	while (curr_layer != NULL && curr_layer->prev != NULL) {
		int curr_size = curr_layer->num_nodes;

		// calculate the loss for that layer
		float *diff = NULL;
		if (curr_layer == nn->output_layer) {
			diff = loss_gradient(curr_layer->node_vals, actual_output, curr_size, nn->loss_type);
		}
		else {
			NN_Layer* next_layer = curr_layer->next;
			diff = matmul(next_layer->weights, loss, next_layer->num_nodes, curr_size, FALSE);
			free(loss);
		}
		float* curr_activated = get_differentiated_activated_layer(curr_layer);
		loss = vec_hadamard(diff, curr_activated, curr_size);
		
		free(diff);
		free(curr_activated);
		
		// calculate the weights and bias gradients
		NN_Layer* prev_layer = curr_layer->prev;
		int prev_size = prev_layer->num_nodes;
		float* prev_activated = (prev_layer != nn->input_layer) ? get_differentiated_activated_layer(prev_layer) : prev_layer->node_vals;

		float** dLdW = vec_outer(prev_activated, loss, prev_size, curr_size);
		float** dW = mat_multiply(nn->learning_rate, dLdW, prev_size, curr_size);
		float* db = vec_multiply(nn->learning_rate, loss, curr_size);

		free_mat(dLdW, prev_size);
		if (prev_layer != nn->input_layer) free(prev_activated);

		// update the weights and bias of the layer
		float** old_weights = curr_layer->weights;
		float* old_bias = curr_layer->bias;
		curr_layer->weights = mat_diff(old_weights, dW, prev_size, curr_size);
		curr_layer->bias = vec_diff(old_bias, db, curr_size);

		free(db);
		free(old_bias);
		free_mat(old_weights, prev_size);
		free_mat(dW, prev_size);
		
		curr_layer = prev_layer;
	}
	free(loss);
}

// to set the input vector of the NN
void set_nn_inputs(NN* nn, float* inputs)
{
	float* nn_inputs = nn->input_layer->node_vals;
	for (int i = 0; i < nn->input_layer->num_nodes; ++i) {
		nn_inputs[i] = inputs[i];
	}
}

// to set the node values and activation values to zero for all layers of the NN
void clear_nn_values(NN* nn)
{
	for (NN_Layer* curr_layer = nn->input_layer; curr_layer != NULL; curr_layer = curr_layer->next) {
		for (int i = 0; i < curr_layer->num_nodes; ++i) {
			curr_layer->node_vals[i] = 0.0f;
			curr_layer->activated_node_vals[i] = 0.0f;
		}
	}
}

// returns the prediction made by the NN for a given input vector
// if target_dataset is passed (i.e., not NULL), then denormalized predictions are returned
float* predict(NN* nn, float* inputs, DataSet* target_dataset, Bool print_prediction)
{
	set_nn_inputs(nn, inputs);
	forward_propagate_nn(nn);
	
	float* nn_outputs = nn->output_layer->node_vals;
	float* prediction = (float*)malloc(nn->output_layer->num_nodes * sizeof(float));

	for (int i = 0; i < nn->output_layer->num_nodes; ++i) {
		prediction[i] = nn_outputs[i];
	}

	float* denormalized_prediction = NULL;
	if (target_dataset != NULL) {
		denormalized_prediction = get_denormalized_vector(prediction, target_dataset);
		free(prediction);
	}

	if (print_prediction) {
		printf("Prediction: ");
		for (int i = 0; i < nn->output_layer->num_nodes; ++i) {
			printf("%.2f ", (target_dataset != NULL) ? denormalized_prediction[i] : prediction[i]);
		}
	}
	return (target_dataset != NULL) ? denormalized_prediction : prediction;
}

// train the NN to predict a given set of input-target pairs, for a given number of epochs each
// print_nn_details: if TRUE, then prints the values, weights and bias for each layer of the NN at the end of each set item
void train(NN* nn, float** input_set, float** target_set, int set_size, int epochs, Bool print_nn_details)
{
	printf("Training neural network for %d epochs, against %d input-target pairs\n", epochs, set_size);
	for (int set_num = 0; set_num < set_size; ++set_num) {
		printf("Processing... (%d / %d)", set_num + 1, set_size);

		float* inputs = input_set[set_num];
		float* target = target_set[set_num];
		
		set_nn_inputs(nn, inputs);

		for (int epoch_num = 1; epoch_num <= epochs; ++epoch_num)
		{
			forward_propagate_nn(nn);
			back_propagate_nn(nn, target);
		}
		if (print_nn_details) {
			printf("\n");
			print_nn(nn);
			printf("\n");
		}
		printf("\r");
	}
	printf("\n--- Complete ---\n\n");
}

// returns a test result containing the set of all prediction vectors for each input vector in the input dataset, and the accuracy vector (giving the accuracy for each output feature)
// if train_target_dataset is passed (i.e., not NULL), the predictions are denormalized in accordance with the mean and std deviation in the target dataset object used during training
NN_TestResult* test(NN* nn, float** test_input_dataset, float** test_target_dataset, int set_size, DataSet* train_target_dataset)
{
	float** prediction_set = (float**)malloc(set_size * sizeof(float*));

	for (int i = 0; i < set_size; ++i) {
		float* prediction = predict(nn, test_input_dataset[i], train_target_dataset, FALSE);
		prediction_set[i] = prediction;
	}

	// calculate and print accuracy vector
	float* accuracy_vector = (float*)malloc(nn->output_layer->num_nodes * sizeof(float));

	for (int j = 0; j < nn->output_layer->num_nodes; ++j) {
		accuracy_vector[j] = 0.0f;
		for (int i = 0; i < set_size; ++i) {
			accuracy_vector[j] += abs(prediction_set[i][j] - test_target_dataset[i][j]) / test_target_dataset[i][j];
		}
		accuracy_vector[j] /= set_size;
	}

	NN_TestResult* test_result = (NN_TestResult*)malloc(sizeof(NN_TestResult));
	test_result->prediction_dataset = prediction_set;
	test_result->accuracy_vector = accuracy_vector;
	test_result->set_size = set_size;
	test_result->vector_size = nn->output_layer->num_nodes;
	return test_result;
}

void print_nn_test_result(NN_TestResult* test_result, Bool print_predictions)
{
	if (print_predictions) {
		printf("--- PREDICTIONS ---\n\n");
		
		for (int i = 0; i < test_result->set_size; ++i) {
			printf("%d | ", i + 1);
			for (int j = 0; j < test_result->vector_size; ++j) {
				printf("%.2f ", test_result->prediction_dataset[i][j]);
			}
			printf("\n");
		}
		printf("\n");
	}

	printf("--- ACCURACY ---\n");
	for (int i = 0; i < test_result->vector_size; ++i) {
		printf("%.2f ", test_result->accuracy_vector[i]);
	}
}

int main()
{
	srand(time(NULL));

	// initialize the neural network
	int num_input_nodes = 5;
	int num_output_nodes = 1;
	int num_hidden_layer_nodes[2] = {10, 10};
	int num_hidden_layers = sizeof(num_hidden_layer_nodes) / sizeof(num_hidden_layer_nodes[0]);
	ACTIVATION_FUNC_TYPE activation_functions[3] = { AF_RELU, AF_RELU, AF_RELU };
	LOSS_FUNC_TYPE loss_type = LF_MSE;
	float learning_rate = 1e-3;

	NN* nn = create_nn(num_input_nodes, num_output_nodes, num_hidden_layers, num_hidden_layer_nodes, activation_functions, loss_type, learning_rate);
	
	// prepare the training data
	DataSet* input_dataset = get_dataset_from_csv("datasets/input_dataset.csv");
	DataSet* target_dataset = get_dataset_from_csv("datasets/target_dataset.csv");

	assert(
		input_dataset->set_size == target_dataset->set_size &&
		input_dataset->vector_size >= num_input_nodes &&
		target_dataset->vector_size >= num_output_nodes
	);

	// train the neural network
	int set_size = input_dataset->set_size;
	int epochs = 10000;

	train(nn, input_dataset->normalized_dataset, target_dataset->normalized_dataset, set_size, epochs, FALSE);
	
	// prepare the testing data
	DataSet* test_input_dataset = get_dataset_from_csv("datasets/test_input_dataset.csv");
	DataSet* test_target_dataset = get_dataset_from_csv("datasets/test_target_dataset.csv");

	assert(
		test_input_dataset->set_size == test_target_dataset->set_size &&
		test_input_dataset->vector_size >= num_input_nodes &&
		test_target_dataset->vector_size >= num_output_nodes
	);

	// test the neural network
	int test_set_size = test_input_dataset->set_size;
	NN_TestResult* test_result = test(nn, test_input_dataset->normalized_dataset, test_target_dataset->normalized_dataset, test_set_size, target_dataset);
	print_nn_test_result(test_result, TRUE);

	free_nn(nn);
	return 0;
}
