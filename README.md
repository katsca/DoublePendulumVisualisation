# My Matplotlib Project

This project gives visualization of a double pendulum with two stiff uniform rods whose masses and lengths are variable. The acceleration due to gravity is also variable but there is no resitance (ideal conditions).

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/katsca/DoublePendulumVisualisation.git
   cd DoublePendulumVisualisation
   
2. Create a virtual Environment 
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows

3. Install Dependencies
   ```bash
   pip install -r requirements.txt
   
## Usage

Run the main Script
   ```bash
   python doublePendulum.py
   ```

The flags you can add onto the end<br>
--h:
   * Help
   
--length1:
   * The length of rod 1 in m
   * Must be a positive float
   * Default: 1.0
   
--length2:
   * The length of rod 2 in m<br>
   * Must be a positive float
   * Default: 1.0
   
--mass1:
   * The mass of rod 1 in kg
   * Must be a positive float
   * Default: 1.0
   
--mass2:
   * The mass of rod 2 in kg
   * Must be a positive float
   * Default: 1.0 

--gravity:
   * The acceleration due to gravity in m/s^2
   * Must be a positive float
   * Default: 9.8
   
--theta1:
   * The initial angle of rod 1 to the verticle in rad
   * Must be a positive float between -pi and pi
   * Default: Some random float between -pi and pi

--theta2:
   * The initial angle of rod 2 to the verticle in rad
   * Must be a positive float between -pi and pi
   * Default: Some random float between -pi and pi

--t_samples:
   * The number of samples taken over a set period of time
   * Must be a positive int
   * Default: 1000

--t_length:
   * The length of time to run for in s
   * Must be a positive float
   * Default: 20.0

--t_speed:
   * The time between each frame of the animation in s
   * Must be a positive float
   * Default: 0.1

## EXAMPLE IMAGE OF PROGRAMME WORKING
![alt text](https://github.com/katsca/DoublePendulumVisualisation/blob/main/ExampleImage.png?raw=true)


## Workings to get the equations
Can be found in Double_Pendulum_Workings.pdf




