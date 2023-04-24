/* 

   This is the output handler cpp file, this defines all the needed 
   function detail for handling the output. 
   
   This simply just does the majority hardvote.


*/

#include "output_handler.h"


// This function just does a major hard vote and fill the result of the final inference
// to the vote list. Which is the list of size 5.
void decision_collection(TfLiteTensor * output_tensor, uint8_t* vote_lst){
   for(uint8_t num_nets = 0; num_nets<5; num_nets++)
   {
     uint8_t infer_this_net = 0;
     int8_t temp_output_value;
     int8_t max_value = 0;

     for (uint8_t index_output = 0; index_output<18; index_output++)
     {
       temp_output_value = output_tensor->data.int8[18*num_nets+index_output];
       if (temp_output_value > max_value)
       {
         infer_this_net = index_output;
         max_value = temp_output_value;
       }

     }
     vote_lst[num_nets] = infer_this_net;
   }

}

void Major_hardvote(uint8_t* vote_lst, uint8_t* nominees){
  uint8_t count[18] = 
  {0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0};
  uint8_t temp_value;
  uint8_t max_cnt = 0;
  uint8_t winner = 0;
  
  // loop through to get count for each inference
  for(uint8_t index=0; index<5; index++)
  {
    temp_value = vote_lst[index];
    count[temp_value]++;
  }

  // loop through to get the the biggest count (winner count)
  for(uint8_t index_2 = 0; index_2<18; index_2++)
  {
    if(count[index_2] > max_cnt)
    {
      max_cnt = count[index_2];
      winner = index_2;
    }
  }
  
  // log down the winner and its count
  nominees[0] = winner; 
  nominees[1] = max_cnt;
  
  // eliminate the winner and its count from the original array
  count[winner] = 0;
  winner = 0;
  max_cnt = 0;

  // loop through again to the max count, aka the second winner
  for(uint8_t index_3 = 0; index_3<18; index_3++)
  {
    if(count[index_3] > max_cnt)
    {
      max_cnt = count[index_3];
      winner = index_3;
    }
  }
  // log down the winner and its count
  nominees[2] = winner;
  nominees[3] = max_cnt;

}



