import random
from leer_data import leer_datos_archivo

def greedy_estocastico(D, aviones, matriz_tiempos, seed):
    random.seed(seed)
    secuencia_aterrizaje_indices = []
    tiempos_aterrizaje_programados = [0] * D
    costo_total = 0

    for _ in range(D):
        aviones_candidatos = []
        costos_candidatos = []
        tiempos_factibles_candidatos = []

        for i in range(D):
            if i not in secuencia_aterrizaje_indices:
                # Calcular el tiempo factible más temprano considerando todos los aviones programados
                tiempo_factible = aviones[i]['t_temprano']
                for prev_idx in secuencia_aterrizaje_indices:
                    tiempo_factible = max(tiempo_factible, tiempos_aterrizaje_programados[prev_idx] + matriz_tiempos[prev_idx][i])

                # Si el tiempo factible está dentro del rango permitido
                if tiempo_factible <= aviones[i]['t_tarde']:
                    costo = 0
                    # Penalización por adelanto
                    if tiempo_factible < aviones[i]['t_pref']:
                        costo += aviones[i]['pena_temprano'] * (aviones[i]['t_pref'] - tiempo_factible)
                    # Penalización por retraso
                    elif tiempo_factible > aviones[i]['t_pref']:
                        costo += aviones[i]['pena_tarde'] * (tiempo_factible - aviones[i]['t_pref'])

                    aviones_candidatos.append(i)
                    costos_candidatos.append(costo)
                    tiempos_factibles_candidatos.append(tiempo_factible)

        if not aviones_candidatos:
            # No se encontró un avión factible, forzar la selección del primero disponible
            for i in range(D):
                if i not in secuencia_aterrizaje_indices:
                    tiempo_factible = aviones[i]['t_temprano']
                    for prev_idx in secuencia_aterrizaje_indices:
                        tiempo_factible = max(tiempo_factible, tiempos_aterrizaje_programados[prev_idx] + matriz_tiempos[prev_idx][i])
                    tiempo_factible = min(tiempo_factible, aviones[i]['t_tarde'])

                    costo = 0
                    if tiempo_factible < aviones[i]['t_pref']:
                        costo += aviones[i]['pena_temprano'] * (aviones[i]['t_pref'] - tiempo_factible)
                    elif tiempo_factible > aviones[i]['t_pref']:
                        costo += aviones[i]['pena_tarde'] * (tiempo_factible - aviones[i]['t_pref'])

                    aviones_candidatos = [i]
                    costos_candidatos = [costo]
                    tiempos_factibles_candidatos = [tiempo_factible]
                    break

        # Selección estocástica
        indice_seleccionado = random.randint(0, len(aviones_candidatos) - 1)
        mejor_avion_indice = aviones_candidatos[indice_seleccionado]
        costo_total += costos_candidatos[indice_seleccionado]
        tiempo_aterrizaje_seleccionado = tiempos_factibles_candidatos[indice_seleccionado]

        secuencia_aterrizaje_indices.append(mejor_avion_indice)
        tiempos_aterrizaje_programados[mejor_avion_indice] = tiempo_aterrizaje_seleccionado

    # Crear la secuencia de aviones en el orden de aterrizaje
    secuencia_aterrizaje_ordenada = [secuencia_aterrizaje_indices[i] for i in range(D)]

    # Crear la lista de tiempos de aterrizaje en el orden de la secuencia
    tiempos_aterrizaje_ordenados = [tiempos_aterrizaje_programados[i] for i in secuencia_aterrizaje_indices]

    return secuencia_aterrizaje_ordenada, costo_total, tiempos_aterrizaje_ordenados

if __name__ == '__main__':
    # Ejemplo de uso (asumiendo que ya tienes la función leer_datos_archivo)
    D, aviones, matriz_tiempos = leer_datos_archivo("cases/case1.txt")  # Reemplaza con el nombre de tu archivo

    # Greedy Estocástico (múltiples ejecuciones con verificación)
    for i in range(10):
        secuencia_estoc, costo_estoc, tiempos_estoc_ordenados = greedy_estocastico(D, aviones, matriz_tiempos, seed=i)
        print(f"Greedy Estocástico (Seed={i}) - Secuencia:", secuencia_estoc, "Costo:", costo_estoc, "Tiempos de Aterrizaje:", tiempos_estoc_ordenados)

        # Verificar tiempos de separación
        for j in range(len(secuencia_estoc) - 1):
            avion1_index = secuencia_estoc[j]
            avion2_index = secuencia_estoc[j+1]
            tiempo_aterrizaje1 = tiempos_estoc_ordenados[j]
            tiempo_aterrizaje2 = tiempos_estoc_ordenados[j+1]
            tiempo_separacion = matriz_tiempos[avion1_index][avion2_index]
            if tiempo_aterrizaje2 < tiempo_aterrizaje1 + tiempo_separacion:
                print(f"  ¡Advertencia! El tiempo de separación entre el avión {avion1_index} y el avión {avion2_index} no se cumple.")
                print(f"    Avión {avion1_index} aterriza en: {tiempo_aterrizaje1}")
                print(f"    Avión {avion2_index} aterriza en: {tiempo_aterrizaje2}")
                print(f"    Tiempo de separación requerido: {tiempo_separacion}")