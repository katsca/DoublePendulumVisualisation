from matplotlib import pyplot as plt
import unicodedata
import numpy as np
import random
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
import argparse

def ELE1(theta1, dtheta1, theta2, dtheta2):
    denominator1 = m2*l2/4 + 1/12 * m2*l2
    denominator2 = (m1*l1/4+m2*l1+1/12*m1*l1) - (1/2*m2*l1*np.cos(theta1-theta2))*(m2*l2*1/2*np.cos(theta1-theta2)/denominator1)
    expression1 = -m2*l2*1/2*dtheta2*np.sin(theta1-theta2) - m1*g*np.sin(theta1)/1/2 - m2*g*np.sin(theta1)
    expression2 = -m2*l2*1/2*np.cos(theta1-theta2) * (1/2*m2*l1*dtheta1*np.sin(theta1-theta2) - m2*g*1/2*np.sin(theta2)) / denominator1
    expression3 = expression1 + expression2
    return (expression3/denominator2)    


def ELE2(theta1, dtheta1, theta2, dtheta2):
    denominator1 = 1/2*m2*l2*np.cos(theta1-theta2)
    expression1 = (m1*l1*1/4+m2*l1+1/12*m1*l1) * (-m2*l2*1/4-1/12*m2*l2) / denominator1
    denominator2 = m2*l2*1/2*np.cos(theta1-theta2) + expression1
    expression2 = (m1*l1*1/4+m2*l1+1/12*m1*l1) * (1/2*m2*l1*dtheta1*np.sin(theta1-theta2) - m2*g*1/2*np.sin(theta2)) / denominator1 + m2*l2*dtheta2*np.sin(theta1-theta2) + m1*g*np.sin(theta1)/2 + m2*g*np.sin(theta1)
    return -expression2 / denominator2
    
def EOM(t, y):
    theta1, dtheta1, theta2, dtheta2 = y
    ddtheta1 = ELE1(theta1, dtheta1, theta2, dtheta2)
    ddtheta2 = ELE2(theta1, dtheta1, theta2, dtheta2)
    return [dtheta1, ddtheta1, dtheta2, ddtheta2]

def positive_float(value):
    try:
        f = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a valid float")
    if f <= 0:
        raise argparse.ArgumentTypeError(f"{value} must be positive")
    return f
    
def valid_float_angle(value):
    try:
        f = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a valid float")
    if f > np.pi or f < -np.pi:
        pi_text = unicodedata.lookup("GREEK SMALL LETTER PI")
        raise argparse.ArgumentTypeError(f"{value} must be between -{pi_text}, {pi_text}")
    return f


def positive_int(value):
    try:
        i = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a valid int")
    if i <= 0:
        raise argparse.ArgumentTypeError(f"{value} must be positive")
    return i



#Randomly generate the initial theta1 and theta2 between [-pi, pi]
theta1_init = random.uniform(-np.pi, np.pi)
theta2_init = random.uniform(-np.pi, np.pi)

#Input arguments
parser = argparse.ArgumentParser(description="Double Pendulum visualisation no Damping")
parser.add_argument("--length1", type=positive_float, default=1.0,
                    help="Length of rod 2 in m  (default: 1)")
parser.add_argument("--length2", type=positive_float, default=1.0,
                    help="Length of rod 2 in m (default: 1)")
parser.add_argument("--mass1", type=positive_float, default=1.0,
                    help="Mass of rod 1 in kg (default: 1)")
parser.add_argument("--mass2", type=positive_float, default=1.0,
                    help="Mass of rod 2 in kg (default: 1)")
parser.add_argument("--gravity", type=positive_float, default=9.8,
                    help="Acceleration of gravity in m/s^2 (default: 9.8)")
parser.add_argument("--theta1", type=valid_float_angle, default=theta1_init,
                    help="Start angle of theta 1 in rad (default: random)")
parser.add_argument("--theta2", type=valid_float_angle, default=theta2_init,
                    help="Start angle of theta 2 in rad (default: random)")
parser.add_argument("--t_samples", type=positive_int, default=1000,
                    help="Number of samples taken (default: 1000)")
parser.add_argument("--t_length", type=positive_float, default=20,
                    help="Length of time to run for in s (default: 20)")

parser.add_argument("--t_speed", type=positive_float, default=0.1,
                    help="The time between each frame in the animation in s (default: 0.1)")
args = parser.parse_args()

l1 = args.length1
l2 = args.length2
m1 = args.mass1
m2 = args.mass2
g = args.gravity
theta1_init = args.theta1
theta2_init = args.theta2
T = args.t_length
t_spacing = args.t_samples
t_interval = args.t_speed
max_axis_length = l1+l2

t_eval = np.linspace(0,T,1000)
y0 = [theta1_init, 0.0, theta2_init, 0.0]

#solve
sol = solve_ivp(EOM, [0,T], y0, t_eval=t_eval, method="RK45")
print("SOLVED EOM")

#Now going to plot over time 
t = sol.t # time values in array 1D
theta1 = sol.y[0] #theta1 values in array 1D
theta2 = sol.y[2] # theta2 values in array 1D

#ROD 1 - define the start and various end points in the array
x1_start, y1_start = 0, 0
x1_end = l1 * np.sin(theta1)
y1_end = -l1 * np.cos(theta1)

#ROD 2- define the start and various end points in the array
x2_start = x1_end
y2_start = y1_end
x2_end = x2_start + l2 * np.sin(theta2)
y2_end = y2_start - l2 * np.cos(theta2)

#CM ROD 1 AND 2 - define the centre of mass positions for each time
x1_cm = np.array(l1 * np.sin(theta1) * 1/2)
y1_cm = np.array(-l1*np.cos(theta1) * 1/2)
x2_cm = np.array(2*x1_cm + l2*np.sin(theta2)*1/2)
y2_cm = np.array(2*y1_cm - l2*np.cos(theta2)*1/2)

#set up figure
fig, ax = plt.subplots(figsize=(5,5))
#Axis liimits
ax.set_xlim(-max_axis_length, max_axis_length)
ax.set_ylim(-max_axis_length, max_axis_length)
ax.set_xlabel("X POSITION (m)")
ax.set_ylabel("Y POSITION (m)")
ax.set_aspect('equal')

#Define the various plots we will have
#Lines to represent the rods

line1, = ax.plot([], [], 'o-', lw=3, label="Rod 1", zorder=5)
line2, = ax.plot([], [], 'o-', lw=3, label="Rod 2", zorder=5)

#Position of centre of mass
scatter1 = ax.scatter([], [], c='blue', s=30, label='CoM Rod 1', zorder=6)
scatter2 = ax.scatter([], [], c='red', s=30, label='CoM Rod 2', zorder=6)
# Trails as thin lines
trail1, = ax.plot([], [], c='blue', lw=1, alpha=0.6, label="CoM Rod 1 Path", zorder=2)
trail2, = ax.plot([], [], c='red', lw=1, alpha=0.6, label="CoM Rod 2 Path", zorder=2)
#Pivot position
scatter3 = ax.scatter([0], [0], c="black", s=30, zorder=10)

#Text to tell us initial conditions and current time
cond_text = f"Initial conditions: θ1={theta1[0]:.2f}rad, θ2={theta2[0]:.2f}rad"
cond_text_2 = f"Length 1: {l1}m, Mass 1 : {m1}kg & Length 2: {l2}m, Mass 2: {m2}kg"
text_handle = ax.text(0, max_axis_length+max_axis_length*0.275, cond_text, ha='center', va='top', fontsize=12, transform=ax.transData)
text_handle_2 = ax.text(0, max_axis_length+max_axis_length*0.175, cond_text_2, ha='center', va='top', fontsize=12, transform=ax.transData)
time_text = ax.text(0, max_axis_length+max_axis_length*0.075, '', ha='center', va='top', fontsize=12, transform=ax.transData)

#Initialize our various lines
def init():
    line1.set_data([],[])
    line2.set_data([],[])
    scatter1.set_offsets(np.empty((0, 2)))
    scatter2.set_offsets(np.empty((0, 2)))
    trail1.set_data([], [])
    trail2.set_data([], [])
    return line1, line2, scatter1, scatter2, trail1, trail2


def update(i): 
    #Rod 1
    x1_end_i = l1 * np.sin(theta1[i])
    y1_end_i = -l1*np.cos(theta1[i])
    line1.set_data([0,x1_end_i], [0, y1_end_i])
    
    #Rod 2
    x2_start_i = x1_end_i
    y2_start_i = y1_end_i
    x2_end_i = x2_start_i + l2*np.sin(theta2[i])
    y2_end_i = y2_start_i - l2*np.cos(theta2[i])
    line2.set_data([x2_start_i, x2_end_i], [y2_start_i, y2_end_i])
    
    #Scatter 1 and (CoM positions)
    scatter1.set_offsets([[x1_cm[i], y1_cm[i]]])
    scatter2.set_offsets([[x2_cm[i], y2_cm[i]]])
    
    #Update trails of CoM
    trail1.set_data(x1_cm[:i+1], y1_cm[:i+1])
    trail2.set_data(x2_cm[:i+1], y2_cm[:i+1])
    
    #Update time
    time_text.set_text(f"t = {t[i]:.2f} s")
    
    return line1, line2, scatter1, scatter2, trail1, trail2, time_text


    
ani = FuncAnimation(fig, update, frames=len(t), init_func=init,
                    interval=t_interval, blit=False)
plt.legend(loc="upper right")
plt.show()