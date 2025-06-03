import numpy as np

# --- Funciones Objetivo ---
def f1(x):
    if len(x) != 2:
        raise ValueError("f1 requiere un vector de 2 dimensiones")
    return 4 - 4 * x[0]**3 - 4 * x[0] + x[1]**2

def f2(x):
    if len(x) != 6:
        raise ValueError("f2 requiere un vector de 6 dimensiones")
    return (1/899.0) * np.sum(x**2) - 1745

def f3(x):
    if len(x) != 2:
        raise ValueError("f3 requiere un vector de 2 dimensiones")
    return (x[0]**6 + x[1]**4 - 17)**2 + (2 * x[0] + x[1] - 4)**2

def f4(x):
    if len(x) != 10:
        raise ValueError("f4 requiere un vector de 10 dimensiones")

    # Penalizar por fuera del dominio
    if np.any(x <= 2.0) or np.any(x >= 10.0):
        return np.inf

    sum_terms = np.sum((np.log(x - 2))**2 + (np.log(10 - x))**2)
    prod_term = np.prod(x)

    return sum_terms - (prod_term**0.2)

# --- PSO ---
def pso(objective_func, dim, bounds, num_particulas, max_iter, w, c1, c2, max_vel_ratio=0.2):
    lower_bound, upper_bound = bounds
    max_velocity = max_vel_ratio * (upper_bound - lower_bound)

    positions = lower_bound + (upper_bound - lower_bound) * np.random.rand(num_particulas, dim)
    velocities = (np.random.rand(num_particulas, dim) * 2 - 1) * max_velocity

    lbest_positions = np.copy(positions)
    lbest_values = np.array([objective_func(p) for p in lbest_positions])

    gbest_index = np.argmin(lbest_values)
    gbest_position = np.copy(lbest_positions[gbest_index])
    gbest_value = lbest_values[gbest_index]

    history = [gbest_value]

    for iter_count in range(max_iter):
        # Obtener el valor de w
        current_w = w(iter_count, max_iter) if callable(w) else w

        for i in range(num_particulas):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)

            # Actualizar velocidad
            cognitive_velocity = c1 * r1 * (lbest_positions[i] - positions[i])
            social_velocity = c2 * r2 * (gbest_position - positions[i])
            velocities[i] = current_w * velocities[i] + cognitive_velocity + social_velocity

            # Limitar velocidad
            velocities[i] = np.clip(velocities[i], -max_velocity, max_velocity)

            # Actualizar posición
            positions[i] = positions[i] + velocities[i]

            # Manejar límites del dominio
            positions[i] = np.clip(positions[i], lower_bound, upper_bound)

            # Evaluar nueva posición
            current_value = objective_func(positions[i])

            # Actualizar pbest
            if current_value < lbest_values[i]:
                lbest_values[i] = current_value
                lbest_positions[i] = np.copy(positions[i])

        # Actualizar gbest
        best_lbest_index = np.argmin(lbest_values)
        if lbest_values[best_lbest_index] < gbest_value:
            gbest_value = lbest_values[best_lbest_index]
            gbest_position = np.copy(lbest_positions[best_lbest_index])

        history.append(gbest_value)
    return gbest_position, gbest_value, history

info_funciones_objetivo = [
    {"name": "f1", "func": f1, "dim": 2, "bounds": (-5, 5)},
    {"name": "f2", "func": f2, "dim": 6, "bounds": (0, 1)},
    {"name": "f3", "func": f3, "dim": 2, "bounds": (-500, 500)},
    {"name": "f4", "func": f4, "dim": 10, "bounds": (2.0001, 9.9999)},
]

parametros = [
    {"name": "Config_1_Estándar", "w": 0.8, "c1": 2.0, "c2": 2.0, "num_particulas": 50, "max_iter": 1000},
    {"name": "Config_2_Más_Exploratoria", "w": 0.9, "c1": 2.5, "c2": 1.5, "num_particulas": 50, "max_iter": 1000},
    {"name": "Config_3_Más_Explotadora", "w": 0.7, "c1": 1.5, "c2": 2.5, "num_particulas": 50, "max_iter": 1000},
    {"name": "Config_4_WDecreciente", "w": lambda iter, max_iter: 0.9 - iter * (0.9 - 0.4) / max_iter, "c1": 2.0, "c2": 2.0, "num_particulas": 50, "max_iter": 1000},
]
num_ejecuciones = 10