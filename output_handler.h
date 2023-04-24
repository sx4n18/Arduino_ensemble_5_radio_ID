/* 

   This is the output handler header file, this defines all the needed 
   function header for handling the output. 
   
   This basically just does the major hard vote.


*/

#ifndef ENSEMBLE_MODEL_5_LOG_PLUS_1_FOR_RADIO_ID_OUTPUT_HANDLER_H_
#define ENSEMBLE_MODEL_5_LOG_PLUS_1_FOR_RADIO_ID_OUTPUT_HANDLER_H_

#include "tensorflow/lite/c/common.h"

void decision_collection(TfLiteTensor * output_tensor, uint8_t* vote_lst);

void Major_hardvote(uint8_t* vote_lst, uint8_t* nominees);




#endif