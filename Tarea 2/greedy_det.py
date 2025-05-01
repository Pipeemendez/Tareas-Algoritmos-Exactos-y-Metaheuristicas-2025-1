from leer_data import leer_datos_archivo

""" ruta_archivo = 'cases/case1.txt' 
D, aviones, matriz = leer_datos_archivo(ruta_archivo)

print(f"Número de aviones: {D}")
print("Primer avión:", aviones[0])
print("Tiempos de separación del primer avión:", matriz[0])

print("Primer avión:", aviones[1])
print("Tiempos de separación del segundo avión:", matriz[1]) """

#greedy determinista
def greedy_determinista(D, aviones, matriz_tiempos):
    """
    Implementación del algoritmo Greedy Determinista para el problema de aterrizaje de aviones.

    Args:
        D (int): Número de aviones.
        aviones (list): Lista de diccionarios, donde cada diccionario representa un avión
                      y contiene sus tiempos de aterrizaje y penalizaciones.
        matriz_tiempos (list of list): Matriz de tiempos de separación entre aviones.

    Returns:
        tuple: Una tupla que contiene la secuencia de aterrizaje y el costo total.
    """

    secuencia_aterrizaje = []
    tiempos_aterrizaje = [0] * D
    costo_total = 0

    # Inicializar tiempos de aterrizaje con el tiempo más temprano
    for i in range(D):
        tiempos_aterrizaje[i] = aviones[i]['t_temprano']
    
    for _ in range(D):
        mejor_avion = -1
        mejor_costo = float('inf')

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

                    if costo < mejor_costo:
                        mejor_costo = costo
                        mejor_avion = i
        
        if mejor_avion == -1:
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
                    
                    mejor_costo = costo
                    mejor_avion = i
                    break  # Tomar el primer avión disponible

        secuencia_aterrizaje.append(mejor_avion)
        tiempos_aterrizaje[mejor_avion] = tiempo_factible
        costo_total += mejor_costo

    return secuencia_aterrizaje, costo_total

if __name__ == '__main__':
    # Ejemplo de uso (asumiendo que ya tienes la función leer_datos_archivo)
    D, aviones, matriz_tiempos = leer_datos_archivo("cases/case1.txt")  # Reemplaza con el nombre de tu archivo

    # Greedy Determinista
    secuencia_det, costo_det = greedy_determinista(D, aviones, matriz_tiempos)
    print("Greedy Determinista - Secuencia:", secuencia_det, "Costo:", costo_det)