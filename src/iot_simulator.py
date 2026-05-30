import random

def generate_sensor_data():
    return {
        "Type": random.choice([0, 1, 2]),
        "Air temperature [K]": round(random.uniform(295, 310), 2),
        "Process temperature [K]": round(random.uniform(305, 320), 2),
        "Rotational speed [rpm]": random.randint(1200, 1800),
        "Torque [Nm]": round(random.uniform(25, 70), 2),
        "Tool wear [min]": random.randint(0, 250),
        "TWF": 0,
        "HDF": 0,
        "PWF": 0,
        "OSF": 0,
        "RNF": 0
    }