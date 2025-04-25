#pragma once

#ifndef NN_H
#define NN_H

#include "helper.h"

typedef enum { AF_NONE, AF_RELU } ACTIVATION_FUNC_TYPE;

struct NN_Layer {
	float* node_vals;
	float* activated_node_vals;
	int num_nodes;
	float** weights;
	float* bias;
	float (*activation_function)(float val);
	ACTIVATION_FUNC_TYPE af_type;
	struct NN_Layer* next;
	struct NN_Layer* prev;
};
typedef struct NN_Layer NN_Layer;

typedef struct {
	NN_Layer* input_layer;
	NN_Layer** hidden_layers;
	NN_Layer* output_layer;
	int num_hidden_layers;
	LOSS_FUNC_TYPE loss_type;
	float learning_rate;
} NN;

typedef struct {
	float** prediction_dataset;
	float* accuracy_vector;
	int set_size;
	int vector_size;
} NN_TestResult;

NN_Layer* create_nn_layer(int size);
void free_nn_layer(NN_Layer* layer, int weight_size);
float* create_bias_vector(int size);
float** create_weight_matrix(int rows, int cols);
float (*get_activation_function(ACTIVATION_FUNC_TYPE af_type))(float);
NN* create_nn(const int num_input_nodes, const int num_output_nodes, const int num_hidden_layers, const int* num_hidden_layer_nodes, const ACTIVATION_FUNC_TYPE* activation_functions, const LOSS_FUNC_TYPE loss_type, const float learning_rate);
void free_nn(NN* nn);
void print_layer(NN_Layer* layer, Bool print_weights_and_bias);
void print_nn(NN* nn);
void forward_propagate_nn(NN* nn);
float* get_activated_layer(NN_Layer* layer);
void back_propagate_nn(NN* nn, float* actual_output);
float* get_differentiated_activated_layer(NN_Layer* layer);
void set_nn_inputs(NN* nn, float* inputs);
void clear_nn_values(NN* nn);
float* predict(NN* nn, float* inputs, DataSet* target_dataset, Bool print_prediction);
void train(NN* nn, float** input_set, float** target_set, int set_size, int epochs, Bool print_nn_details);
NN_TestResult* test(NN* nn, float** test_input_dataset, float** test_target_dataset, int set_size, DataSet* train_target_dataset);
void print_nn_test_result(NN_TestResult* test_result, Bool print_predictions);

#endif
