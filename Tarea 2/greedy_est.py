import random
from leer_data import leer_datos_archivo

def greedy_estocastico(D, aviones, matriz_tiempos, seed):
    """
    Implementación del algoritmo Greedy Estocástico para el problema de aterrizaje de aviones.

    Args:
        D (int): Número de aviones.
        aviones (list): Lista de diccionarios, donde cada diccionario representa un avión
                      y contiene sus tiempos de aterrizaje y penalizaciones.
        matriz_tiempos (list of list): Matriz de tiempos de separación entre aviones.
        seed (int): Semilla para la generación de números aleatorios.

    Returns:
        tuple: Una tupla que contiene la secuencia de aterrizaje y el costo total.
    """

    random.seed(seed)
    secuencia_aterrizaje = []
    tiempos_aterrizaje = [0] * D
    costo_total = 0

    # Inicializar tiempos de aterrizaje con el tiempo más temprano
    for i in range(D):
        tiempos_aterrizaje[i] = aviones[i]['t_temprano']

    for _ in range(D):
        aviones_candidatos = []
        costos_candidatos = []

        for i in range(D):
            if i not in secuencia_aterrizaje:
                # Calcular el tiempo de aterrizaje factible más temprano
                tiempo_factible = aviones[i]['t_temprano']
                if secuencia_aterrizaje:
                    tiempo_factible = max(tiempo_factible, max(tiempos_aterrizaje[j] + matriz_tiempos[j][i] 
                                            for j in secuencia_aterrizaje))
                
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
        
        if not aviones_candidatos:
            # No se encontró un avión factible, se fuerza a aterrizar el primero disponible dentro de lo posible
            for i in range(D):
                if i not in secuencia_aterrizaje:
                    tiempo_factible = aviones[i]['t_temprano']
                    if secuencia_aterrizaje:
                        tiempo_factible = max(tiempo_factible, max(tiempos_aterrizaje[j] + matriz_tiempos[j][i] 
                                                for j in secuencia_aterrizaje))
                    tiempo_factible = min(tiempo_factible, aviones[i]['t_tarde']) # Asegurar que no se pase del límite tardío
                    
                    costo = 0
                    if tiempo_factible < aviones[i]['t_pref']:
                        costo += aviones[i]['pena_temprano'] * (aviones[i]['t_pref'] - tiempo_factible)
                    elif tiempo_factible > aviones[i]['t_pref']:
                        costo += aviones[i]['pena_tarde'] * (tiempo_factible - aviones[i]['t_pref'])
                    
                    aviones_candidatos = [i]
                    costos_candidatos = [costo]
                    break

        # Selección estocástica: elegir un avión aleatorio entre los candidatos
        indice_seleccionado = random.randint(0, len(aviones_candidatos) - 1)
        mejor_avion = aviones_candidatos[indice_seleccionado]
        costo_total += costos_candidatos[indice_seleccionado]
        
        # Calcular el tiempo de aterrizaje factible para el avión seleccionado
        tiempo_factible = aviones[mejor_avion]['t_temprano']
        if secuencia_aterrizaje:
            tiempo_factible = max(tiempo_factible, max(tiempos_aterrizaje[j] + matriz_tiempos[j][mejor_avion] for j in secuencia_aterrizaje))
        tiempos_aterrizaje[mejor_avion] = tiempo_factible
        secuencia_aterrizaje.append(mejor_avion)

    return secuencia_aterrizaje, costo_total

if __name__ == '__main__':
    # Ejemplo de uso (asumiendo que ya tienes la función leer_datos_archivo)
    D, aviones, matriz_tiempos = leer_datos_archivo("cases/case1.txt")  # Reemplaza con el nombre de tu archivo

    # Greedy Estocástico (múltiples ejecuciones)
    for i in range(10):
        secuencia_estoc, costo_estoc = greedy_estocastico(D, aviones, matriz_tiempos, seed=i)
        print(f"Greedy Estocástico (Seed={i}) - Secuencia:", secuencia_estoc, "Costo:", costo_estoc)