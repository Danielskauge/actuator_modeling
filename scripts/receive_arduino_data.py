import serial
import csv
import time

# Set the serial port and baud rate (adjust '/dev/ttyACM0' or 'COM3' as necessary)
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
time.sleep(2)  # Allow time for the Arduino to initialize/reset

# Open the CSV file for writing (each row will be a list of strings)
with open('arduino_data.csv', mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    # Write header row
    csv_writer.writerow(["timestamp", "angle", "lin_acc_x", "lin_acc_y", "lin_acc_z", "ang_vel_x", "ang_vel_y", "ang_vel_z"])
    
    try:
        while True:
            if ser.in_waiting > 0:
                # Read a line from the serial port and decode it as UTF-8 text.
                line = ser.readline().decode('utf-8').strip()
                # Split the line into components using comma as delimiter.
                data = line.split(',')
                # Basic check: ensure we have exactly 8 fields
                if len(data) == 8:
                    csv_writer.writerow(data)
                    print("Logged:", data)
                else:
                    print("Unexpected data format:", line)
    except KeyboardInterrupt:
        print("Data logging stopped.")