*   Performs crucial preprocessing and feature engineering:
    *   Converts time from milliseconds to seconds and determines sampling frequency.
    *   Converts angles from degrees to radians.
    *   Calculates current angular velocity from gyroscope data.
    *   Calculates the target `tau_measured` (measured torque) using the formula `inertia * angular_acceleration_from_tangential_accelerometer`.
    *   **Applies a Savitzky-Golay filter to smooth the calculated `tau_measured` to reduce noise from accelerometer readings.**
    *   Extracts a fixed set of **3 input features** for the model: `current_angle_rad`, `target_angle_rad`, and `current_ang_vel_rad_s`. The explicit `target_ang_vel_rad_s` feature has been removed, relying on the sequence of target angles to implicitly convey this information to the neural network.
*   Shapes the data into sequences of a fixed length (`SEQUENCE_LENGTH = 2` by default) suitable for recurrent or time-aware models. The target torque corresponds to the state at the end of the sequence.