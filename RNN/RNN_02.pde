/* 17/01/2021
 * Debugar a rede neural recorrente
 * 16/01/2021
*/

AI ai;
Neural_Network RNN;

int[] present_layer = { 1, 2, 1};
int[] past_layer    = { 5, 3, 0};

void setup() {
  //size(400, 300);
  fullScreen();
  RNN = new Neural_Network(present_layer, past_layer);
  ai = new AI(RNN);
  thread("learning");
}

void draw() {
  background(0);
  stroke(255);
  line(0, height/2, width, height/2);
  int n_show = 2;
  if(mousePressed) {
    for(int i = 0; i < n_show; i++)
      ai.showIntegrativeFuncion(i);
    ai.showFuncion();
    ai.showBestIntegrativeFuncion();
  } else {
    for(int i = 0; i < n_show; i++)
      ai.showFuncion(i);
    ai.showDerivativeFuncion();
    ai.showBestFuncion();
  }
}

void learning() {
  while(true) {
    ai.novaGeracaoPSO();
    ai.printFitness();
  }
}