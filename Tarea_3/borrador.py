import numpy as np
import matplotlib.pyplot as plt
import time

# --- Definición de las Funciones Objetivo ---

def f1(x):
    # Dominio: [-5, 5]^2
    # Dimensión: 2
    if len(x) != 2:
        raise ValueError("f1 requiere un vector de 2 dimensiones")
    return 4 - 4 * x[0]**3 - 4 * x[0] + x[1]**2

def f2(x):
    # Dominio: [0, 1]^6
    # Dimensión: 6
    # Nota: Esta es la función Esfera (escalada y desplazada), unimodal
    if len(x) != 6:
        raise ValueError("f2 requiere un vector de 6 dimensiones")
    return (1/899.0) * np.sum(x**2) - 1745

def f3(x):
    # Dominio: [-500, 500]^2
    # Dimensión: 2
    if len(x) != 2:
        raise ValueError("f3 requiere un vector de 2 dimensiones")
    return (x[0] + x[1]**4 - 17)**2 + (2 * x[0] + x[1] - 4)**2

def f4(x):
    # Dominio especificado: -2.001 < xi <= 10
    # Dominio efectivo para ln: 2 < xi < 10
    # Dimensión: 10
    # Usaremos un rango ligeramente menor para evitar log(0) o log(negativo)
    # Dominio implementado: [2.0001, 9.9999]
    if len(x) != 10:
        raise ValueError("f4 requiere un vector de 10 dimensiones")

    # Penalizar puntos fuera del dominio efectivo (aunque PSO con clamp debería evitarlo)
    # Si se llega aquí con valores inválidos, retornar un valor muy alto
    if np.any(x <= 2.0) or np.any(x >= 10.0):
         return np.inf # Infinito positivo

    sum_terms = np.sum((np.log(x - 2))**2 + (np.log(10 - x))**2)
    prod_term = np.prod(x)

    # Para el prod_term**0.2, si prod_term es negativo, el resultado real no está definido.
    # Dado el dominio (2, 10), todos los x_i son positivos, por lo que el producto siempre es positivo.
    # Si el dominio fuera diferente, habría que manejar esta parte (e.g., retornar inf).
    return sum_terms - (prod_term**0.2)

# --- Implementación del Algoritmo PSO ---

def pso(objective_func, dim, bounds, num_particulas, max_iter, w, c1, c2, max_vel_ratio=0.2):
    """
    Implementación básica de Particle Swarm Optimization (Optimización por Enjambre de Partículas).

    Args:
        objective_func: La función a minimizar.
        dim: Dimensión del problema.
        bounds: Tupla (límite_inferior, límite_superior) para cada dimensión.
        num_particulas: Número de partículas en el enjambre.
        max_iter: Número máximo de iteraciones.
        w: Parámetro de inercia (o función que devuelve w por iteración).
        c1: Parámetro cognitivo.
        c2: Parámetro social.
        max_vel_ratio: Ratio para definir la velocidad máxima (vel_max = ratio_vel_max * (superior - inferior)).

    Returns:
        Tupla (posicion_gbest, valor_gbest, historial):
            posicion_gbest: Mejor posición encontrada por el enjambre.
            valor_gbest: Valor de la función en posicion_gbest.
            historial: Lista de los valores de valor_gbest en cada iteración.
    """
    lower_bound, upper_bound = bounds
    max_velocity = max_vel_ratio * (upper_bound - lower_bound)

    # Inicialización
    positions = lower_bound + (upper_bound - lower_bound) * np.random.rand(num_particulas, dim)
    velocities = (np.random.rand(num_particulas, dim) * 2 - 1) * max_velocity # Velocidades iniciales aleatorias

    pbest_positions = np.copy(positions)
    pbest_values = np.array([objective_func(p) for p in pbest_positions])

    gbest_index = np.argmin(pbest_values)
    gbest_position = np.copy(pbest_positions[gbest_index])
    gbest_value = pbest_values[gbest_index]

    history = [gbest_value] # Para el gráfico de convergencia

    # Bucle de iteración
    for iter_count in range(max_iter):
        # Obtener el valor de w para la iteración actual
        current_w = w(iter_count, max_iter) if callable(w) else w

        for i in range(num_particulas):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)

            # Actualizar velocidad
            cognitive_velocity = c1 * r1 * (pbest_positions[i] - positions[i])
            social_velocity = c2 * r2 * (gbest_position - positions[i])
            velocities[i] = current_w * velocities[i] + cognitive_velocity + social_velocity

            # Limitar velocidad
            velocities[i] = np.clip(velocities[i], -max_velocity, max_velocity)

            # Actualizar posición
            positions[i] = positions[i] + velocities[i]

            # Manejar límites del dominio (clamp)
            positions[i] = np.clip(positions[i], lower_bound, upper_bound)

            # Evaluar la nueva posición
            current_value = objective_func(positions[i])

            # Actualizar pbest
            if current_value < pbest_values[i]:
                pbest_values[i] = current_value
                pbest_positions[i] = np.copy(positions[i])

        # Actualizar gbest
        best_pbest_index = np.argmin(pbest_values)
        if pbest_values[best_pbest_index] < gbest_value:
            gbest_value = pbest_values[best_pbest_index]
            gbest_position = np.copy(pbest_positions[best_pbest_index])

        history.append(gbest_value) # Registrar el mejor valor encontrado en esta iteración

    return gbest_position, gbest_value, history

# --- Configuración de Experimentos ---

# Definir las funciones, dimensiones y dominios
functions_info = [
    {"name": "f1", "func": f1, "dim": 2, "bounds": (-5, 5)},
    {"name": "f2", "func": f2, "dim": 6, "bounds": (0, 1)},
    {"name": "f3", "func": f3, "dim": 2, "bounds": (-500, 500)},
    # Usamos límites ajustados para f4 debido a logaritmos
    {"name": "f4", "func": f4, "dim": 10, "bounds": (2.0001, 9.9999)},
]

# Definir las 4+ configuraciones de parámetros de PSO a probar
# Justificación: Exploramos diferentes balances de exploración vs explotación
# Config 1: Estándar balanceado
# Config 2: Más exploratorio (mayor c1)
# Config 3: Más explotador (mayor c2)
# Config 4: Inercia decreciente (exploración inicial, explotación final)
# Puedes añadir más si quieres
pso_configs = [
    {"name": "Config_1_Estándar", "w": 0.8, "c1": 2.0, "c2": 2.0, "num_particulas": 50, "max_iter": 1000},
    {"name": "Config_2_MásExploratoria", "w": 0.9, "c1": 2.5, "c2": 1.5, "num_particulas": 50, "max_iter": 1000},
    {"name": "Config_3_MásExplotadora", "w": 0.7, "c1": 1.5, "c2": 2.5, "num_particulas": 50, "max_iter": 1000},
    # Inercia linealmente decreciente de w_inicio a w_fin
    {"name": "Config_4_WDecreciente", "w": lambda iter, max_iter: 0.9 - iter * (0.9 - 0.4) / max_iter, "c1": 2.0, "c2": 2.0, "num_particulas": 50, "max_iter": 1000},
    # Puedes probar otra configuración, e.g., con más partículas o iteraciones
    # {"name": "Config_5_MásPartículas", "w": 0.8, "c1": 2.0, "c2": 2.0, "num_particulas": 100, "max_iter": 1000},
]

num_runs = 10 # Número de ejecuciones por cada función y configuración

# --- Ejecución de Experimentos ---

results = {}

for func_info in functions_info:
    func_name = func_info["name"]
    objective_func = func_info["func"]
    dim = func_info["dim"]
    bounds = func_info["bounds"]

    print(f"Ejecutando experimentos para la función: {func_name} (Dim: {dim})")
    results[func_name] = {}

    for config in pso_configs:
        config_name = config["name"]
        print(f"  Usando configuración: {config_name}")
        run_results = []
        run_histories = []
        start_time = time.time()

        for run in range(num_runs):
            _, best_value, history = pso(
                objective_func=objective_func,
                dim=dim,
                bounds=bounds,
                num_particulas=config["num_particulas"], # Usar nombre traducido
                max_iter=config["max_iter"],
                w=config["w"],
                c1=config["c1"],
                c2=config["c2"]
            )
            run_results.append(best_value)
            run_histories.append(history)
            print(f"    Ejecución {run+1}/{num_runs}: Mejor valor = {best_value:.6f}")

        end_time = time.time()
        mean_best_value = np.mean(run_results)
        std_best_value = np.std(run_results)
        avg_history = np.mean(run_histories, axis=0) # Promedio del historial de convergencia

        results[func_name][config_name] = {
            "mean_best_value": mean_best_value,
            "std_best_value": std_best_value,
            "all_best_values": run_results,
            "avg_history": avg_history,
            "runtime": end_time - start_time
        }
        print(f"  Configuración {config_name} terminada en {end_time - start_time:.2f}s. Mejor valor promedio: {mean_best_value:.6f} (Desv Est: {std_best_value:.6f})")
        print("-" * 20)

# --- Mostrar Resultados y Generar Gráficos ---

print("\n--- Resumen de Resultados ---")
for func_name, func_results in results.items():
    print(f"\nFunción: {func_name}")
    plt.figure(figsize=(12, 8))
    plt.title(f'Gráfico de Convergencia para {func_name}')
    plt.xlabel('Iteración')
    plt.ylabel('Mejor Fitness Encontrado (escala log)') # Usar escala logarítmica es común para visualizar convergencia
    # La escala logarítmica no funciona con valores negativos o cero.
    # Si tienes valores <= 0, esta línea generará una advertencia o error.
    # Considera comentarla si tus mínimos son negativos, o usar escala lineal.
    # Para f1 y f2 que tienen mínimos negativos, la advertencia es esperable.
    # Para f3 que tiene mínimo 0, también. Solo f4 podría beneficiarse.
    # plt.yscale('log') # Descomentar si los valores son todos positivos

    for config_name, config_results in func_results.items():
        mean_val = config_results["mean_best_value"]
        std_val = config_results["std_best_value"]
        avg_hist = config_results["avg_history"]

        print(f"  Configuración: {config_name}")
        print(f"    Mejor Valor Promedio: {mean_val:.6f}")
        print(f"    Desviación Estándar: {std_val:.6f}")
        print(f"    Tiempo de ejecución (10 ejecuciones): {config_results['runtime']:.2f}s")

        # Ajustar escala log si es necesario, solo si todos los valores en avg_hist son > 0
        if np.all(np.array(avg_hist) > 0):
             plt.plot(avg_hist, label=f'{config_name} (Promedio: {mean_val:.2e})') # Usar notación científica para valores pequeños
        else:
             # Si hay valores <= 0, graficar en escala lineal y ajustar la etiqueta del eje Y si se había puesto log
             plt.plot(avg_hist, label=f'{config_name} (Promedio: {mean_val:.2e})')
             plt.yscale('linear') # Asegurarse de que es lineal
             plt.ylabel('Mejor Fitness Encontrado') # Cambiar etiqueta del eje Y


    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig(f'grafico_convergencia_{func_name}.png') # Guardar el gráfico
    # plt.show() # Descomentar para mostrar los gráficos inmediatamente

print("\nExperimentación completa. Los gráficos de convergencia han sido guardados como archivos .png.")
print("Revisa los gráficos guardados y el resumen de resultados para el análisis.")