public class Robot {
  private float px = 0;
  private float py = 0;
  private float th = 0; // theta
  private int time_micro = 0;
  private float L = 0.1;
  private float vel_x = 0;
  private float vel_l = 0;
  private float prop = 1; // Proporção pixels/metro
  
  Robot(float x, float y, float t, float p) {
    px = x;
    py = y;
    th = t;
    prop = p;
  }
  
  
}
