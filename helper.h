#pragma once

#ifndef HELPER_H
#define HELPER_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <time.h>
#include <math.h>

#ifdef _MSC_VER
#define STRTOK strtok_s
#else
#define STRTOK strtok_r
#endif

#define MAX_CSV_LINE_LENGTH 1000

typedef struct {
	float** dataset;
	float** normalized_dataset;
	float* mean_features;
	float* std_deviation_features;
	int set_size;
	int vector_size;
} DataSet;

typedef enum { FALSE, TRUE } Bool;
typedef enum { LF_MSE } LOSS_FUNC_TYPE;

float* vec_multiply(float scalar, float* vec, int size);
float* vec_diff(float* vec1, float* vec2, int size);
float vec_inner(float* vec1, float* vec2, int size);
float** vec_outer(float* vec1, float* vec2, int vec1_size, int vec2_size);
float* vec_hadamard(float* vec1, float* vec2, int size);
float** mat_diff(float** matrix1, float** matrix2, int rows, int cols);
float* matmul(float** matrix, float* vec, int vec_size, int output_vec_size, Bool multiply_transposed_matrix);
float** mat_multiply(float scalar, float** matrix, int rows, int cols);
void free_mat(float** matrix, int rows);
float ReLU(float val);
float* loss_gradient(float* predictions, float* actual, int num_nodes, LOSS_FUNC_TYPE loss_function_type);
float mse_loss(float* predictions, float* actual, int num_nodes);
float* mse_loss_gradient(float* predictions, float* actual, int num_nodes);
float* differentiated_relu(float* vec, int size);
DataSet* get_dataset_from_csv(const char* file_path);
void normalize_dataset(DataSet* dataset);
float* get_denormalized_vector(float* normalized_vector, DataSet* dataset);

#endif
