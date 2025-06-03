import numpy as np
import matplotlib.pyplot as plt
import time
from PSO import pso, info_funciones_objetivo, parametros, num_ejecuciones
import os

resultados = {}
for funcion_info in info_funciones_objetivo:
    funcion_nombre = funcion_info["name"]
    funcion_objetivo = funcion_info["func"]
    grado = funcion_info["grado"]
    bounds = funcion_info["bounds"]

    print(f"EJECUTANDO EXPERIMENTO PARA LA FUNCIÓN: {funcion_nombre}")
    resultados[funcion_nombre] = {}

    for config in parametros:
        config_nombre = config["name"]
        print(f"  Usando configuración: {config_nombre}")
        resultados_ejecuciones = []
        ejecucion_historico = []
        start_time = time.time()

        for ejecucion in range(num_ejecuciones):
            _, best_value, history = pso(
                objective_func=funcion_objetivo,
                grado=grado,
                bounds=bounds,
                num_particulas=config["num_particulas"],
                max_iter=config["max_iter"],
                w=config["w"],
                c1=config["c1"],
                c2=config["c2"]
            )
            resultados_ejecuciones.append(best_value)
            ejecucion_historico.append(history)
            print(f"    Ejecución {ejecucion+1}/{num_ejecuciones}: Mejor valor = {best_value:.6f}")

        end_time = time.time()
        mean_best_value = np.mean(resultados_ejecuciones)
        std_best_value = np.std(resultados_ejecuciones)
        avg_history = np.mean(ejecucion_historico, axis=0)

        resultados[funcion_nombre][config_nombre] = {
            "mean_best_value": mean_best_value,
            "std_best_value": std_best_value,
            "all_best_values": resultados_ejecuciones,
            "avg_history": avg_history,
            "runtime": end_time - start_time
        }
        print(f"  Configuración {config_nombre} terminada en {end_time - start_time:.2f}s. Mejor valor promedio: {mean_best_value:.6f} (Desviación Estándar: {std_best_value:.6f})")
        print("-" * 20)

print("\n--- RESUMEN DE RESULTADOS ---")
carpeta_graficos = "graficos"
os.makedirs(carpeta_graficos, exist_ok=True)
print(f"\nGuardando gráficos en la carpeta: {carpeta_graficos}")

for funcion_nombre, funcion_resultados in resultados.items():
    print(f"\nFunción: {funcion_nombre}")
    plt.figure(figsize=(12, 8))
    plt.title(f'Gráfico de Convergencia para {funcion_nombre}')
    plt.xlabel('Iteración')
    plt.ylabel('Mejor Fitness Encontrado (en escala log)')

    for config_nombre, config_resultados in funcion_resultados.items():
        mean_val = config_resultados["mean_best_value"]
        std_val = config_resultados["std_best_value"]
        avg_hist = config_resultados["avg_history"]

        print(f"  Configuración: {config_nombre}")
        print(f"    Mejor Valor Promedio: {mean_val:.6f}")
        print(f"    Desviación Estándar: {std_val:.6f}")
        print(f"    Tiempo de ejecución : {config_resultados['runtime']:.2f}s")

        if np.all(np.array(avg_hist) > 0):
            plt.plot(avg_hist, label=f'{config_nombre} (Promedio: {mean_val:.2e})')
        else:
            plt.plot(avg_hist, label=f'{config_nombre} (Promedio: {mean_val:.2e})')
            plt.yscale('linear')
            plt.ylabel('Mejor Fitness Encontrado')

    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig(os.path.join(carpeta_graficos, f'grafico_convergencia_{funcion_nombre}.png'))

print("\nExperimentación completa. Gráficos guardados")