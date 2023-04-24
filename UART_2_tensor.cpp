/* 
  This cpp file defines the input handler functions in detail.

*/

#include "UART_2_tensor.h"

void pass_the_number_in(TfLiteTensor* input_tensor, double* log_processed_value_lst)
{
  uint8_t net_index[5] = {5, 8, 11, 17, 19};
  float scale = input_tensor->params.scale;
  float zero_point = input_tensor->params.zero_point;
  unsigned int slicing_start = 0;
  for(uint8_t index=0; index<5; index++)
  {
    unsigned int length_of_list = 1023 - net_index[index];
    bin_ratio_subtraction_n_quantise(length_of_list, net_index[index], log_processed_value_lst, scale, zero_point, input_tensor->data.int8+slicing_start);
    slicing_start += length_of_list;
  }
  if (slicing_start != 5055)
  {
    MicroPrintf("Allocation of input failed");
    return;
  }
}

void bin_ratio_subtraction_n_quantise(unsigned int length_of_list, uint8_t num_of_net, double* preprocessed_value_lst, float scale, float zero_point, int8_t* data_int8_ptr)
{
  double temp_subtraction;

  for (unsigned int index = 0; index< length_of_list; index++)
  {
    temp_subtraction = preprocessed_value_lst[index+num_of_net+1] - preprocessed_value_lst[index];
    data_int8_ptr[index] = temp_subtraction/scale + zero_point;
  }


}