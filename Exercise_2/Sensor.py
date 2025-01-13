import numpy as np
import math

def quantize_voltage(q, noise_std,bit):
    U_in = 5  # V
    R1 = 11000  # Ohm
    R2 = 16000  # Ohm
    ADC_max_output_voltage = 2.8  # Maximum voltage range for ADC

    # Voltage calculation based on input angles
    U_gait = U_in * (1 - R2 / (R1 + R2) - q / 270)

    # Amplification factor calculation
    angle_min = np.min(q)
    angle_max = np.max(q)
    angles_range = np.linspace(angle_min, angle_max, 100)
    U_range = U_in * (1 - R2 / (R1 + R2) - angles_range / 270)

    current_max_voltage = np.max(U_range)
    amplification_factor = ADC_max_output_voltage / current_max_voltage / 0.9

    R3 = 10000.0  # Ohm
    R4 = R3 * amplification_factor
    R4 = math.floor(R4 / 1000) * 1000

    amplification_factor = R4 / R3

    # Add noise and apply amplification
    noise = np.random.normal(0, noise_std, len(q))
    noisy_amplified_U_gait = (U_gait + noise) * amplification_factor

    # Filter function
    def H(f):
        R = 80000  # Ohm
        C = 1e-6  # Farad
        return 1 / np.sqrt(1 + (2 * np.pi * f * R * C) ** 2)

    # Apply filter
    filtered_amplified_U_gait = (H(1) * U_gait + noise) * amplification_factor

    # Quantization
    U_scaled = filtered_amplified_U_gait * ((2**bit-1) / 3)
    quantized_voltage = np.round(U_scaled) * (3 / (2**bit-1))

    return quantized_voltage


#################################################################################


for i in range(num_steps):
    current_time = t_control[i]

    # Reference trajectory and measurement
    reference = sensor(reference_trajectory(current_time), 0)  # generate digital reference signal
    measured = sensor(current_state[1], 0.01)

    # Calculate control signal
    u_control = controller.compute_control(-reference, -measured)
    if u_control > 1: 
        u_control = 1
    if u_control < -1: 
        u_control = -1