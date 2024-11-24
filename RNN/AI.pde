public class AI {
  private Neural_Network rnn;
  private PSO pso;
  private int tam = 100;
  private int numP = 500;
  private int fut = 3;
  private int geracaoAtual = 0;
  private float dx = 0.05;
  private float[] d_vet;
  private float[] i_vet;
  private float[] c_vet;
  private float[] best_vet;
  private float[] best_i_vet;
  private float best_fitness = -9999.9;
  private Matrix[] weightsAndBias = new Matrix[tam];
  
  AI(Neural_Network dl) {
    rnn = dl;
    pso = new PSO();
    pso.setSize(tam);
    pso.setCoord(dl.getVetValuesLength());
    
    best_vet = new float[numP];
    best_i_vet = new float[numP];
    d_vet = derivativeFunction().clone();
    c_vet = function().clone();
    i_vet = integrativeFunction(d_vet).clone();
    startRNN();
  }
  
  private void startRNN() {
    for(int i = 0; i < tam; i++) {
      this.weightsAndBias[i] = new Matrix(rnn.getVetValuesLength(), 1);
      weightsAndBias[i].randomFillNormal(0.2);
      for(int j = 0; j < rnn.getVetValuesLength(); j++) pso.pop[i].pose[0].coord[j] = weightsAndBias[i].get(j, 0);
    }
  }
  
  private void updateBest() {
    rnn.restart();
    rnn.updateValues(weightsAndBias[pso.bestPoseInd]);
    Matrix ent = rnn.getEntrance();
    Matrix saida;
    
    for(int j = 0; j < fut; j++) {
      ent.set(0, 0, 0);
      rnn.feedforward();
      saida = rnn.getOutput();
      best_vet[j] = saida.get(0, 0);
    }
    
    for(int j = fut; j < numP; j++) {
      ent.set(0, 0, d_vet[j-fut]);
      rnn.feedforward();
      saida = rnn.getOutput();
      best_vet[j] = saida.get(0, 0);
    }
    best_i_vet = integrativeFunction(best_vet);
    best_fitness = pso.pop[pso.bestPoseInd].fitness;
  }
  
  public float[] getPopArray(int ind) {
    if(ind >= numP) return null;
    float[] _vet = new float[numP];
    rnn.restart();
    rnn.updateValues(weightsAndBias[ind]);
    Matrix ent = rnn.getEntrance();
    Matrix saida;
    
    for(int j = 0; j < fut; j++) {
      ent.set(0, 0, 0);
      rnn.feedforward();
      saida = rnn.getOutput();
      _vet[j] = saida.get(0, 0);
    }
    
    for(int j = fut; j < numP; j++) {
      ent.set(0, 0, d_vet[j-fut]);
      rnn.feedforward();
      saida = rnn.getOutput();
      _vet[j] = saida.get(0, 0);
    }
    return _vet;
  }
  
  public void networkTest() {
    for(int i = 0; i < tam; i++) testarRede(i);
  }
  
  public void testarRede(int i) {
    rnn.restart();
    rnn.updateValues(weightsAndBias[i]);
    Matrix ent = rnn.getEntrance();
    Matrix saida;
    float sum = 0, val;
    
    for(int j = 0; j < fut; j++) {
      ent.set(0, 0, 0);
      rnn.feedforward();
      saida = rnn.getOutput();
      val = saida.get(0, 0);
      val = d_vet[j] - val;
      sum += val*val;
    }
    
    for(int j = fut; j < numP; j++) {
      ent.set(0, 0, d_vet[j - fut]);
      rnn.feedforward();
      saida = rnn.getOutput();
      val = saida.get(0, 0);
      val = d_vet[j] - val;
      sum += val*val;
    }
    pso.pop[i].fitness = -sum;
  }
  
  private void psoVetValues() {
    for(int i = 0; i < tam; i++) {
      for(int j = 0; j < rnn.getVetValuesLength(); j++) weightsAndBias[i].mat[j][0] = pso.pop[i].pose[0].coord[j];
    }
  }
  
  public void novaGeracaoPSO() {
    networkTest();
    pso.iteration();
    updateBest();
    psoVetValues();
    geracaoAtual++;
  }
  
  public void printFitness() {
    println(best_fitness);
  }
  
  public float function(float x) {
    return 4*noise(x/6.0) - 2;//sin(x);
  }
  
  public float[] function() {
    float[] res = new float[numP];
    float vl = 0;
    for(int i = 0; i < res.length; i++) {
      res[i] = function(vl);
      vl += dx;
    }
    return res;
  }
  
  public float[] derivativeFunction() {
    float[] res = new float[numP];
    float vl = 0;
    for(int i = 0; i < res.length; i++) {
      res[i] = (function(vl + dx) - function(vl)) / dx;
      vl += dx;
    }
    return res;
  }
  
  public float[] integrativeFunction(float[] vet) {
    float[] res = new float[numP];
    float vl = c_vet[0]/dx;
    res[0] = c_vet[0];
    for(int i = 1; i < res.length; i++) {
      res[i] = vl*dx;
      vl += vet[i];
    }
    return res;
  }
  
  public void showFuncion() {
    showFuncion(c_vet);
  }
  
  public void showBestFuncion() {
    showFuncion(color(255, 255, 0), best_vet);
  }
  
  public void showBestIntegrativeFuncion() {
    showFuncion(color(255, 255, 0), best_i_vet);
  }
  
  public void showDerivativeFuncion() {
    showFuncion(color(0, 255, 0), d_vet);
  }
  
  public void showIntegrativeFuncion() {
    showFuncion(color(0, 0, 255), i_vet);
  }
  
  public void showFuncion(int ind) {
    showFuncion(color(127 - 3*ind, 127 - 3*ind, 0), getPopArray(ind));
  }
  
  public void showFuncion(float[] fn) {
    showFuncion(color(255, 0, 0), fn);
  }
  
  public void showFuncion(color c, float[] fn) {
    float dx = width / (float) fn.length;
    float max = 1.5;
    float dy = height / (2.0 * max);
    float x = 0, y = 0, x1 = 0, y1 = 0;
    stroke(c);
    strokeWeight(width/400.0 + 1);
    for(int i = 1; i < fn.length; i++) {
      x1 = x + dx;
      y1 = height/2 - fn[i]*dy;
      y = height/2 - fn[i-1]*dy;
      line(x, y, x1, y1);
      x += dx;
    }
  }
  
  public void showIntegrativeFuncion(int ind) {
    showFuncion(color(127 - 3*ind, 127 - 3*ind, 0), integrativeFunction(getPopArray(ind)));
  }
  
}
