import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import argparse
from datetime import datetime


class JointDynamics:
    def __init__(self, inertia, kp, kd, spring_k, dt=0.001):
        """
        Initialize the dynamic system
        
        Args:
            inertia: moment of inertia of the joint
            kp: proportional gain for PD controller
            kd: derivative gain for PD controller
            spring_k: spring constant for torsional spring
            dt: time step for simulation
        """
        self.inertia = inertia
        self.kp = kp
        self.kd = kd
        self.spring_k = spring_k
        self.dt = dt
        
        # State variables
        self.angle = 0.0
        self.velocity = 0.0
        self.acceleration = 0.0
        
    def reset(self):
        """Reset the state of the system"""
        self.angle = 0.0
        self.velocity = 0.0
        self.acceleration = 0.0
        
    def step(self, target_angle):
        """
        Simulate one time step of the system
        
        Args:
            target_angle: target angle for the joint
            
        Returns:
            angle, velocity, acceleration
        """
        # Calculate torques
        error = target_angle - self.angle
        
        # PD controller torque
        pd_torque = self.kp * error - self.kd * self.velocity
        
        # Spring torque (negative because it opposes displacement from 0)
        spring_torque = -self.spring_k * self.angle
        
        # Total torque
        total_torque = pd_torque + spring_torque
        
        # Calculate acceleration (τ = I*α)
        self.acceleration = total_torque / self.inertia
        
        # Update velocity and angle using Euler integration
        self.velocity += self.acceleration * self.dt
        self.angle += self.velocity * self.dt
        
        return self.angle, self.velocity, self.acceleration


def generate_sine_target(duration, dt, frequency=0.5, amplitude=1.0):
    """
    Generate sine wave target
    
    Args:
        duration: duration in seconds
        dt: time step
        frequency: sine wave frequency in Hz
        amplitude: sine wave amplitude in radians
        
    Returns:
        time and target arrays
    """
    time = np.arange(0, duration, dt)
    target = amplitude * np.sin(2 * np.pi * frequency * time)
    return time, target


def generate_data(duration=10.0, dt=0.001, 
                 inertia=0.1, kp=5.0, kd=0.5, spring_k=1.0,
                 sine_freq=0.5, sine_amp=1.0,
                 noise_stddev=0.01, radius_to_imu=0.1):
    """
    Generate synthetic data for a rotating joint
    
    Args:
        duration: simulation duration in seconds
        dt: time step for simulation
        inertia: moment of inertia
        kp: proportional gain
        kd: derivative gain
        spring_k: spring constant
        sine_freq: frequency of sine wave target
        sine_amp: amplitude of sine wave target
        noise_stddev: standard deviation of noise for measurements
        
    Returns:
        DataFrame with time series data, where time is in seconds from start
    """
    # Initialize system
    joint = JointDynamics(inertia, kp, kd, spring_k, dt)
    
    # Generate target angles
    time, targets = generate_sine_target(duration, dt, sine_freq, sine_amp)
    num_steps = len(time)
    
    # Initialize arrays for data collection
    angles = np.zeros(num_steps)
    angular_velocities = np.zeros(num_steps)
    angular_accelerations = np.zeros(num_steps)
    
    # Run simulation
    for i in range(num_steps):
        angles[i], angular_velocities[i], angular_accelerations[i] = joint.step(targets[i])
    
    # Create dataframe
    data = {
        'time': time,
        'target_angle': targets,
        'angle': angles,
        'ang_vel_x': angular_velocities + np.random.normal(0, noise_stddev, num_steps),
        'ang_vel_y': np.zeros(num_steps),
        'ang_vel_z': np.zeros(num_steps),
        'lin_acc_x': angular_accelerations * radius_to_imu + np.random.normal(0, noise_stddev, num_steps),
        'lin_acc_y': np.zeros(num_steps),
        'lin_acc_z': np.zeros(num_steps)
    }
    
    return pd.DataFrame(data)


def plot_data(df, output_path=None):
    """Create plots of the generated data"""
    plt.figure(figsize=(12, 8))
    
    # angle plot
    plt.subplot(3, 1, 1)
    plt.plot(df['time'], df['target_angle'], 'r-', label='Target')
    plt.plot(df['time'], df['angle'], 'b-', label='Actual')
    plt.ylabel('angle (rad)')
    plt.legend()
    plt.grid(True)
    
    # Velocity plot
    plt.subplot(3, 1, 2)
    plt.plot(df['time'], df['ang_vel_x'], 'g-')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.grid(True)
    
    # Acceleration plot
    plt.subplot(3, 1, 3)
    plt.plot(df['time'], df['lin_acc_x'], 'm-')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Acceleration (rad/s²)')
    plt.grid(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic data for a rotating joint')
    parser.add_argument('--duration', type=float, default=10.0, help='Simulation duration in seconds')
    parser.add_argument('--dt', type=float, default=0.001, help='Time step for simulation')
    parser.add_argument('--inertia', type=float, default=0.1, help='Moment of inertia')
    parser.add_argument('--kp', type=float, default=5.0, help='Proportional gain')
    parser.add_argument('--kd', type=float, default=0.5, help='Derivative gain')
    parser.add_argument('--spring_k', type=float, default=1.0, help='Spring constant')
    parser.add_argument('--sine_freq', type=float, default=0.5, help='Frequency of sine wave target')
    parser.add_argument('--sine_amp', type=float, default=1.0, help='Amplitude of sine wave target')
    parser.add_argument('--noise', type=float, default=0.01, help='Measurement noise standard deviation')
    parser.add_argument('--no_plot', action='store_true', help='Disable plotting')
    parser.add_argument('--radius_to_imu', type=float, default=0.1, help='Radius to IMU from the joint axis')
    args = parser.parse_args()
    
    # Generate data
    print(f"Generating synthetic data with duration={args.duration}s...")
    df = generate_data(
        duration=args.duration,
        dt=args.dt,
        inertia=args.inertia,
        kp=args.kp,
        kd=args.kd,
        spring_k=args.spring_k,
        sine_freq=args.sine_freq,
        sine_amp=args.sine_amp,
        noise_stddev=args.noise,
        radius_to_imu=args.radius_to_imu
    )
    
    # Create output directory if it doesn't exist
    os.makedirs('data/synthetic', exist_ok=True)
    
    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save to CSV
    csv_path = f"data/synthetic/joint_data_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")
    
    # Plot if requested
    if not args.no_plot:
        plot_path = f"data/plots/joint_data_{timestamp}.png"
        os.makedirs('data/plots', exist_ok=True)
        plot_data(df, plot_path)
        print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
