from leer_data import leer_datos_archivo
import random
import copy

def calcular_costo(secuencia, aviones, matriz_tiempos):
    """Calcula el costo total de una secuencia de aterrizajes."""
    costo_total = 0
    tiempos_aterrizaje = [0] * len(secuencia)

    for i, avion_idx in enumerate(secuencia):
        avion = aviones[avion_idx]
        tiempo_minimo = avion['t_temprano']

        if i > 0:
            tiempo_minimo = max(tiempo_minimo, tiempos_aterrizaje[secuencia[i-1]] + matriz_tiempos[secuencia[i-1]][avion_idx])

        tiempo_aterrizaje = max(tiempo_minimo, avion['t_pref'])
        tiempos_aterrizaje[avion_idx] = tiempo_aterrizaje

        if tiempo_aterrizaje < avion['t_pref']:
            costo_total += (avion['t_pref'] - tiempo_aterrizaje) * avion['pena_temprano']
        elif tiempo_aterrizaje > avion['t_pref']:
            costo_total += (tiempo_aterrizaje - avion['t_pref']) * avion['pena_tarde']

        # Asegurar que se cumpla el tiempo máximo de aterrizaje
        if tiempo_aterrizaje > avion['t_tarde']:
            return float('inf') # Solución inválida

    return costo_total

def generar_vecino_intercambio(solucion):
    """Genera un vecino intercambiando dos aviones en la secuencia."""
    vecino = list(solucion)
    idx1, idx2 = random.sample(range(len(vecino)), 2)
    vecino[idx1], vecino[idx2] = vecino[idx2], vecino[idx1]
    return vecino

def hill_climbing_alguna_mejora(solucion_inicial, aviones, matriz_tiempos, max_iter_local=100):
    """Implementación del algoritmo Hill Climbing con la estrategia de alguna mejora."""
    mejor_solucion = list(solucion_inicial)
    mejor_costo = calcular_costo(mejor_solucion, aviones, matriz_tiempos)

    for _ in range(max_iter_local):
        vecino = generar_vecino_intercambio(mejor_solucion)
        costo_vecino = calcular_costo(vecino, aviones, matriz_tiempos)

        if costo_vecino < mejor_costo:
            mejor_solucion = vecino
            mejor_costo = costo_vecino
            return mejor_solucion, mejor_costo # Retorna la primera mejora encontrada

    return mejor_solucion, mejor_costo

def grasp_alguna_mejora(D, aviones, matriz_tiempos, max_iter_local=100, num_restarts_estocastico=5):
    """Implementación del algoritmo GRASP con Hill Climbing (alguna mejora) y reinicios."""
    mejor_solucion_global = None
    mejor_costo_global = float('inf')

    # 1. Usar el greedy determinista como punto de partida (un solo "reinicio")
    solucion_greedy_det = [0, 1, 11, 10, 14, 12, 13, 9, 8, 6, 5, 7, 4, 3, 2] # Reemplaza con tu función greedy determinista si es diferente
    costo_greedy_det = calcular_costo(solucion_greedy_det, aviones, matriz_tiempos)
    solucion_local_det, costo_local_det = hill_climbing_alguna_mejora(solucion_greedy_det, aviones, matriz_tiempos, max_iter_local)
    if costo_local_det < mejor_costo_global:
        mejor_costo_global = costo_local_det
        mejor_solucion_global = solucion_local_det
    print(f"GRASP (alguna mejora) - Inicial (Greedy Determinista): Costo = {costo_local_det}, Secuencia = {solucion_local_det}")

    # 2. Usar las soluciones del greedy estocástico como puntos de partida con múltiples reinicios del Hill Climbing
    soluciones_greedy_estocastico = [
        [13, 6, 14, 7, 0, 5, 12, 11, 4, 3, 9, 8, 10, 1, 2],
        [2, 10, 14, 1, 6, 3, 12, 13, 7, 11, 8, 4, 0, 9, 5],
        [13, 14, 0, 2, 3, 8, 5, 9, 6, 11, 4, 1, 12, 7, 10],
        [3, 10, 9, 2, 7, 14, 12, 1, 8, 0, 11, 6, 13, 4, 5],
        [3, 5, 1, 14, 9, 11, 4, 2, 0, 6, 12, 10, 7, 8, 13],
        [9, 4, 13, 6, 14, 11, 0, 12, 10, 2, 1, 5, 3, 8, 7],
        [12, 9, 1, 8, 5, 0, 2, 6, 13, 11, 10, 7, 4, 3, 14],
        [5, 2, 8, 13, 0, 3, 14, 4, 7, 11, 1, 9, 6, 10, 12],
        [3, 6, 8, 2, 5, 0, 4, 9, 7, 13, 10, 14, 12, 1, 11],
        [7, 10, 5, 4, 2, 3, 0, 12, 11, 9, 14, 1, 8, 6, 13]
    ]
    for i, solucion_inicial in enumerate(soluciones_greedy_estocastico):
        print(f"\n--- Resultados GRASP (alguna mejora) - Inicial (Greedy Estocástico {i+1}) ---")
        for restart in range(num_restarts_estocastico):
            solucion_local, costo_local = hill_climbing_alguna_mejora(solucion_inicial, aviones, matriz_tiempos, max_iter_local)
            print(f"  Reinicio {restart+1}: Costo = {costo_local}, Secuencia = {solucion_local}")
            if costo_local < mejor_costo_global:
                mejor_costo_global = costo_local
                mejor_solucion_global = solucion_local

    print("\n--- Resultados Finales GRASP (Hill Climbing - Alguna Mejora) ---")
    print(f"Mejor secuencia encontrada: {mejor_solucion_global}")
    print(f"Costo total de la mejor secuencia: {mejor_costo_global}")

    return mejor_solucion_global, mejor_costo_global

D, aviones, matriz_tiempos = leer_datos_archivo("cases/case1.txt")

# --- Ejecutar GRASP con Hill Climbing (alguna mejora) ---
mejor_secuencia_am, mejor_costo_am = grasp_alguna_mejora(D, aviones, matriz_tiempos)