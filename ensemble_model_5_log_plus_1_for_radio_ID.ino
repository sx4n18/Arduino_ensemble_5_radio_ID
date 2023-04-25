
#include <math.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <TensorFlowLite.h>

#include "whittle_ensemble_5.h"
#include "output_handler.h"
#include "UART_2_tensor.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define SCREEN_WIDTH   128 // OLED display width, in pixels
#define SCREEN_HEIGHT  64  // OLED display height, in pixels

#define OLED_RESET -1
#define SCREEN_ADDRESS 0x3C //I2C address for this oled screen

// instantiate a class for screen display
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);


// Globals, used for compatibility with Arduino-style sketches.
namespace {
const tflite::Model* ensemble_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 1024 * 14;
// Keep aligned to 16 bytes for CMSIS
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// global variables definition
double log_1_plus_received_input[1024];
char rc;
byte voting_list[5];
char* radio_label[] = {"Am241", "Ba133", "BGD", "Co57", "Co60", "Cs137", "DU", "EU152", "Ga67", "HEU",
"I131", "Ir192", "Np237", "Ra226", "Tc99m", "Th232", "Tl201", "WGPu"};
byte nominees[4];


void setup() {
  // put your setup code here, to run once:
  tflite::InitializeTarget();

  // Serial initialisation
  Serial.begin(9600);
  
  // SSD1306_SWITCHCAPVCC = generate display voltage from 3.3V internally
  // OLED display check, this would normally pass... so I removed the Serial
  // connection prerequisite for this statement.
  if(!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)){
    Serial.println(F("SSD1306 allocation failed"));
    for (;;);
  }
  
  ////////////////////////////////////////////////////////////////
  //this is needed, since adafruit logo is in the display buffer

  display_on_oled(0,0, 1, 0, "Loading model...");

  delay(500);
  ////////////////////////////////////////////////////////////////


  // Map the ensemble model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  ensemble_model = tflite::GetModel(whittled_ensemble_5_tflite);
  if (ensemble_model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        ensemble_model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }
  
  // This pulls in all the operation implementations we need.
  // For our case, only 6 operators were needed, this could 
  // be observed by using netron app
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroMutableOpResolver<6> ensemble_resolver;
  ensemble_resolver.AddStridedSlice();
  ensemble_resolver.AddSoftmax();
  ensemble_resolver.AddFullyConnected();
  ensemble_resolver.AddRelu();
  ensemble_resolver.AddQuantize();
  ensemble_resolver.AddConcatenation();

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      ensemble_model, ensemble_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
  
  ////////////////////////////////////////////////////////////////
  //  OLED display

  display_on_oled(0,0, 1, 1, "Initialising...");

  delay(500);
  ////////////////////////////////////////////////////////////////

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }
  
  while (!Serial)
  {}
  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);


  ////////////////////////////////////////////////////////////////
  //  OLED display

  display_on_oled(0,0, 1, 1, "Initialisation done");
  delay(1000);
  ////////////////////////////////////////////////////////////////
  Serial.println(F("Waiting for input"));

  if ((input->dims->size != 2) || (input->dims->data[0] != 1) ||
      (input->dims->data[1] != 5055) || (input->type != kTfLiteInt8)) 
  {
    MicroPrintf("Bad input tensor parameters in model");
    return;
  }
  
  display.clearDisplay();

}

void loop() {
  // put your main code here, to run repeatedly:
  // applying what I did in the old implementation to here.

  //display_on_oled(0, 20, 2, 0, "Waiting 4 'C' ");
  //display_on_oled_without_flushing(60, 0, 1, "Waiting 4 '");
  display.setTextColor(SSD1306_BLACK,SSD1306_WHITE);
  display.setCursor(60,0);
  display.print("Ready to be");
  display.setCursor(60, 8);
  display.print("gin");
  display.display();
  rc = Serial.read();
  while(rc != 'C')
    {
      rc = Serial.read();
    }
  
  //display_on_oled(0, 20, 2, 0, "Data loading...");
  //display_on_oled_without_flushing(60, 0, 1, "Data loading...");
  Serial.println("Y");
  display.setCursor(60,0);
  display.print("Data loadin");
  display.setCursor(60, 8);
  display.print("g...");
  display.display();
  for(int i = 0; i< 1024; i++)
    {
      //Receive all the input numbers
       while (Serial.available () == 0)
         { }
       double temp_integer = Serial.parseInt();
       Serial.read();// ditch the extra char
       log_1_plus_received_input[i] = log(temp_integer+1);
      //float temp_num = Serial.parseFloat();
      //char  temp_char = Serial.read(); //ditch the extra zero
         // Quantize the input from floating-point to integer
        // input->data.
        // int8_t x_quantized = temp_num / input->params.scale + input->params.zero_point;
         // Place the quantized input in the model's input tensor
        // input->data.int8[i] = x_quantized;

    }

  //display_on_oled(0, 20, 2, 0, "Data received");
  display.setCursor(60,0);
  display.print("Inferring..");
  display.display();

  // pass the received preprocessed data into the input tensor
  pass_the_number_in(input, log_1_plus_received_input);

  // after passing in the log-ed and subtracted value into the tensor,
  // invoke the interpreter

  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Invoke failed" );
    return;
  }
  
  //display_on_oled(10,20, 2, 0, "Inferencing...");

  // getting the hard major vote of the output tensor and put it in the voting list

  decision_collection(output, voting_list);


  Major_hardvote(voting_list, nominees);
  Serial.println(nominees[0]);
  display_histogram(nominees);
  delay(1500);

  


}

void display_on_oled(int8_t x, int8_t y, byte text_size, bool inverse, char* to_display_content)
{
  display.clearDisplay();
  display.setTextSize(text_size);
  if (!inverse){
    display.setTextColor(SSD1306_WHITE);
  }
  else
  {
    display.setTextColor(SSD1306_BLACK,SSD1306_WHITE);
  }
  display.setCursor(x, y);
  display.println(to_display_content);
  display.display();
}

void display_on_oled_without_flushing(int8_t x, int8_t y, byte text_size, char* content_2_display)
{
  display.setTextSize(text_size);
  display.setCursor(x, y);
  display.println(content_2_display);
  display.display();
}

void display_histogram(byte* winner_n_count)
{
  display.clearDisplay();
  display.fillRect(5, 12, 20, winner_n_count[1]*5, SSD1306_WHITE);
  display.fillRect(30, 12, 20, winner_n_count[3]*5, SSD1306_WHITE);
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(12, 0);
  display.print(winner_n_count[0]);
  display.setCursor(35, 0);
  display.print(winner_n_count[2]);
  display.setCursor(12, 45);
  display.print(winner_n_count[1]);
  display.setCursor(35, 45);
  display.print(winner_n_count[3]);
  display.setCursor(20, 55);
  display.print(F("Result is:"));
  display.println(radio_label[winner_n_count[0]]);
  display.display();
}
