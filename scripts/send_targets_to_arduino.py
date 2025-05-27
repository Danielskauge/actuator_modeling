import serial
import time
import math
import numpy as np
from typing import Literal, NamedTuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
plt.ion()

@dataclass
class SineParams:
    amp_range: tuple[float, float]
    freq_range: tuple[float, float]
    max_segment_time: float # in seconds

@dataclass
class UniformParams:
    target_range: tuple[float, float]
    delta_t_range: tuple[float, float]


def send_joint_target_angle(target_angle_rad: float):
    """
    Given a list of joint positions (e.g., [45, 90, 135]),
    converts them into a comma-separated string and sends over serial.
    """
    # Convert each numerical value to a string and join with commas.
    
    command = str(target_angle_rad)
    # Append a newline to indicate end-of-command
    ser.write((command + "\n").encode('utf-8'))
    print("Sent:", command)
    
class SineGenerator:
    def __init__(self, params: SineParams, update_freq: float):
        self.params = params
        assert self.params.amp_range[1] <= np.pi, "Amplitude range must be less than or equal to pi"
        assert self.params.amp_range[0] > 0, "Amplitude range must be greater than 0"
        self.update_freq = update_freq
        self.reset()
    
    def reset(self):
        self.freq = np.random.uniform(*self.params.freq_range)
        self.amp = np.random.uniform(*self.params.amp_range)
        self.current_sample = 0
        period = 1.0 / self.freq
        num_periods = max(int(self.params.max_segment_time / period), 1) # number of periods in the segment, rounded down, at least 1
        self.samples_in_segment = int(num_periods * period * self.update_freq) # number of samples in the segment
    
    def next_target(self) -> float:
        if self.current_sample >= self.samples_in_segment:
            self.reset()
        
        t = self.current_sample / self.update_freq
        target = self.amp * np.sin(2 * np.pi * self.freq * t)
        self.current_sample += 1
        return target

class UniformGenerator:
    def __init__(self, params: UniformParams, update_freq: float):
        self.params = params
        self.update_freq = update_freq
        self.reset()
    
    def reset(self):
        self.target = np.random.uniform(*self.params.target_range)
        hold_time = np.random.uniform(*self.params.delta_t_range)
        self.samples_to_hold = int(hold_time * self.update_freq)
        self.current_sample = 0
    
    def next_target(self) -> float:
        if self.current_sample >= self.samples_to_hold:
            self.reset()
        self.current_sample += 1
        return self.target

def run_target_generation(
    mode: Literal["sin", "uniform"], 
    duration: float, 
    update_freq: float, 
    params: SineParams | UniformParams,
    log_targets: bool = False,
    send_targets: bool = True
) -> list[float] | None:

    if log_targets:
        targets_history = []
    generator = SineGenerator(params, update_freq) if mode == "sin" else UniformGenerator(params, update_freq)
    period = 1.0 / update_freq
    end_time = time.time() + duration
    
    while time.time() < end_time:
        start_time = time.time()
        target = generator.next_target()
        
        if send_targets:
            send_joint_target_angle(target)
        
        if log_targets:
            targets_history.append(target)
            
        elapsed = time.time() - start_time
        if elapsed < period:
            time.sleep(period - elapsed)
            
    if log_targets:
        return targets_history
    else:
        return None
    
    
def plot_targets(targets_history: list[float], update_freq: float):
    time = np.arange(len(targets_history)) / update_freq  # Assuming 120Hz update frequency
    plt.figure(figsize=(10, 6))
    plt.plot(time, targets_history, 'b-', linewidth=2)
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Target Angle (rad)')
    plt.title('Target Angle vs Time')
    plt.savefig("targets.png", dpi=300, bbox_inches='tight')
    plt.close()

# Example usage:
if __name__ == "__main__":
    duration = 20
    update_freq = 120
    log_targets = True
    send_targets = False
    
    if send_targets:
        # Set the serial port and baud rate (adjust '/dev/ttyACM0' or 'COM3' as necessary)
        ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
        time.sleep(2)  # Allow time for the Arduino to initialize/reset

    sine_params = SineParams(
        amp_range=(np.pi/10, np.pi), #min is 0, max is pi radians
        freq_range=(0.5, 2.0),
        max_segment_time=10
    )
    #targets_history = run_target_generation("sin", duration, update_freq, sine_params, log_targets, send_targets=False)
    
        # Example uniform mode
    uniform_params = UniformParams(
        target_range=(-np.pi, np.pi), #min is -pi, max is pi
        delta_t_range=(0.2, 2.0)
    )
    targets_history = run_target_generation("uniform", duration, update_freq, uniform_params, log_targets, send_targets)
    
    if log_targets: plot_targets(targets_history, update_freq)