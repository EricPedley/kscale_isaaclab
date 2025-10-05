import numpy as np
import json

def sample_config(filepath: str):
    thrust_to_weight_ratio = np.random.uniform(1.5, 5)
    mass_min = 0.02
    mass_max  = 5
    mass = np.random.uniform(np.cbrt(mass_min), np.cbrt(mass_max))**3
    # crazyflie thrust curve coefficients
    C0 = 0.038
    C1 = 0.154
    C2 = 0.987
    C_scale = 1/sum([C0, C1, C2])
    T = thrust_to_weight_ratio * mass * 9.81
    c0 = C0 * C_scale * T / 4
    c1 = C1 * C_scale * T / 4
    c2 = C2 * C_scale * T / 4

    R_ms = 7.9 # crazyflie mass to size ratio
    u = np.random.normal(-0.1, 0.1)
    s_ms = 1/(1-u) if u < 0 else 1+u
    l_arm = np.cbrt(mass)/(s_ms*R_ms)

    rotor_positions = [
        [l_arm, -l_arm, 0.0],
        [-l_arm, -l_arm, 0.0],
        [-l_arm, l_arm, 0.0],
        [l_arm, l_arm, 0.0]
    ]

    r_t2i = np.random.uniform(40, 1200)
    tau = T * np.sqrt(2) * l_arm
    J_xx = tau / r_t2i
    J_yy = J_xx # TODO: change symmetric inertia assumption
    J_zz = J_xx * 1.832 # magic number, based on the paper "Data-Driven System Identification of Quadrotors Subject to Motor Delays"
    inertia_diag = [J_xx, J_yy, J_zz]
    moment_constant = np.random.uniform(5e-3, 5e-2)

    config = {
        "mass": mass,
        "inertia_diag": inertia_diag,
        "rotor_positions": rotor_positions,
        "thrust_coefficients": [
            [c0, c1, c2],
            [c0, c1, c2],
            [c0, c1, c2],
            [c0, c1, c2]
        ],
        "rotor_torque_constants": [moment_constant]*4,
        "thrust_directions": [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0]
        ],
        "rotor_torque_directions": [
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0]
        ],
    }

    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)
    
if __name__ == "__main__":
    for i in range(1000):
        sample_config(f'parameters/config_{i}.json')