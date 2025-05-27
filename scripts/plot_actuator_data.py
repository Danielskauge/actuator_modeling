import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import glob
import os

def plot_all_components(df, output_path=None):
    """
    Plot all components in the synthetic data
    """
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))
    
    # Plot angles
    axs[0, 0].plot(df['time'], df['target_angle'], 'r-', label='Target')
    axs[0, 0].plot(df['time'], df['angle'], 'b-', label='Actual')
    axs[0, 0].set_ylabel('Angle (rad)')
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    axs[0, 0].set_title('Joint Angle')
    
    # Plot angular velocities
    axs[0, 1].plot(df['time'], df['ang_vel_x'], 'r-', label='X')
    axs[0, 1].plot(df['time'], df['ang_vel_y'], 'g-', label='Y')
    axs[0, 1].plot(df['time'], df['ang_vel_z'], 'b-', label='Z')
    axs[0, 1].set_ylabel('Angular Velocity (rad/s)')
    axs[0, 1].grid(True)
    axs[0, 1].legend()
    axs[0, 1].set_title('Angular Velocity')
    
    # Plot linear accelerations
    axs[1, 0].plot(df['time'], df['lin_acc_x'], 'r-', label='X')
    axs[1, 0].plot(df['time'], df['lin_acc_y'], 'g-', label='Y')
    axs[1, 0].plot(df['time'], df['lin_acc_z'], 'b-', label='Z')
    axs[1, 0].set_ylabel('Linear Acceleration (m/s²)')
    axs[1, 0].grid(True)
    axs[1, 0].legend()
    axs[1, 0].set_title('Linear Acceleration')
    
    # Plot tracking error
    axs[1, 1].plot(df['time'], df['target_angle'] - df['angle'], 'k-')
    axs[1, 1].set_ylabel('Error (rad)')
    axs[1, 1].grid(True)
    axs[1, 1].set_title('Tracking Error')
    
    # Plot angular velocity magnitude
    vel_mag = np.sqrt(df['ang_vel_x']**2 + df['ang_vel_y']**2 + df['ang_vel_z']**2)
    axs[2, 0].plot(df['time'], vel_mag, 'm-')
    axs[2, 0].set_ylabel('Magnitude (rad/s)')
    axs[2, 0].set_xlabel('Time (s)')
    axs[2, 0].grid(True)
    axs[2, 0].set_title('Angular Velocity Magnitude')
    
    # Plot linear acceleration magnitude
    acc_mag = np.sqrt(df['lin_acc_x']**2 + df['lin_acc_y']**2 + df['lin_acc_z']**2)
    axs[2, 1].plot(df['time'], acc_mag, 'm-')
    axs[2, 1].set_ylabel('Magnitude (m/s²)')
    axs[2, 1].set_xlabel('Time (s)')
    axs[2, 1].grid(True)
    axs[2, 1].set_title('Linear Acceleration Magnitude')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot all components from synthetic joint data')
    parser.add_argument('--file', type=str, help='Path to specific CSV file to plot')
    parser.add_argument('--latest', action='store_true', help='Plot the most recent data file')
    parser.add_argument('--output', type=str, help='Path to save the plot')
    args = parser.parse_args()
    
    # Determine which file to plot
    if args.file:
        file_path = args.file
    elif args.latest:
        # Find the most recent file
        files = glob.glob('data/synthetic/joint_data_*.csv')
        if not files:
            print("No data files found in data/synthetic/")
            return
        file_path = max(files, key=os.path.getctime)
        print(f"Using latest file: {file_path}")
    else:
        # List available files and let user choose
        files = glob.glob('data/synthetic/joint_data_*.csv')
        if not files:
            print("No data files found in data/synthetic/")
            return
        
        print("Available data files:")
        for i, file in enumerate(files):
            print(f"{i}: {os.path.basename(file)}")
        
        choice = input("Enter the number of the file to plot (or 'latest' for most recent): ")
        if choice.lower() == 'latest':
            file_path = max(files, key=os.path.getctime)
        else:
            try:
                file_path = files[int(choice)]
            except (ValueError, IndexError):
                print("Invalid choice. Using the most recent file.")
                file_path = max(files, key=os.path.getctime)
    
    # Load data
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data from {file_path}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Determine output path
    output_path = args.output
    if not output_path and file_path:
        # Create a matching plot filename based on the data file
        base_name = os.path.basename(file_path).replace('.csv', '_all_components.png')
        output_path = os.path.join('data/plots', base_name)
        os.makedirs('data/plots', exist_ok=True)
    
    # Plot data
    plot_all_components(df, output_path)
    if output_path:
        print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main() 