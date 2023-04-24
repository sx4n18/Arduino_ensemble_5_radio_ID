/* 

   This is the input handler header file, this defines all the needed 
   function header for handling the input. 
   
   This will do the log transformation and quantisation and feed the 
   results into the input tensor.


*/

#ifndef ENSEMBLE_MODEL_5_LOG_PLUS_1_FOR_RADIO_ID_UART_2_TENSOR_H_
#define ENSEMBLE_MODEL_5_LOG_PLUS_1_FOR_RADIO_ID_UART_2_TENSOR_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "string.h"

void pass_the_number_in(TfLiteTensor* input_tensor, double* log_processed_value_lst);

void bin_ratio_subtraction_n_quantise(unsigned int length_of_list, uint8_t num_of_net, double* preprocessed_value_lst, float scale, float zero_point, int8_t* data_int8_ptr);

#endif