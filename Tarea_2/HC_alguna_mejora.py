from leer_data import leer_datos_archivo
from greedy_det import greedy_determinista
from greedy_est import greedy_estocastico
import random

def calcular_costo(secuencia, aviones, matriz_tiempos):
    costo_total = 0
    tiempos_aterrizaje = [0] * len(secuencia)

    for i, avion_idx in enumerate(secuencia):
        avion = aviones[avion_idx]
        tiempo_minimo = avion['t_temprano']

        # restriccion tiempo minimo con el y los aviones anteriores para aterrizar
        for j in range(i):
            prev_avion_idx = secuencia[j]
            tiempo_minimo = max(tiempo_minimo, tiempos_aterrizaje[prev_avion_idx] + matriz_tiempos[prev_avion_idx][avion_idx])

        tiempo_aterrizaje = tiempo_minimo
        # restriccion para asegurar que aterrice en el tiempo que corresponde
        if tiempo_aterrizaje > avion['t_tarde']:
            return float('inf')
        tiempo_aterrizaje = min(tiempo_aterrizaje, avion['t_tarde'])
        tiempos_aterrizaje[avion_idx] = tiempo_aterrizaje

        # penalizaciones
        if tiempo_aterrizaje < avion['t_pref']:
            costo_total += (avion['t_pref'] - tiempo_aterrizaje) * avion['pena_temprano']
        elif tiempo_aterrizaje > avion['t_pref']:
            costo_total += (tiempo_aterrizaje - avion['t_pref']) * avion['pena_tarde']

    return costo_total

def generar_vecino_intercambio(solucion, retornar_indices=False):
    vecino = list(solucion)
    idx1, idx2 = random.sample(range(len(vecino)), 2)
    vecino[idx1], vecino[idx2] = vecino[idx2], vecino[idx1]
    
    if retornar_indices:
        return vecino, (idx1, idx2)
    else:
        return vecino

def hill_climbing_alguna_mejora(solucion_inicial, aviones, matriz_tiempos, max_iter_local=100):
    mejor_solucion = list(solucion_inicial)
    mejor_costo = calcular_costo(mejor_solucion, aviones, matriz_tiempos)
    # costo inicial para verficar dps
    costo_inicial = mejor_costo  

    for _ in range(max_iter_local):
        vecino = generar_vecino_intercambio(mejor_solucion)
        costo_vecino = calcular_costo(vecino, aviones, matriz_tiempos)

        if costo_vecino < mejor_costo:
            mejor_solucion = vecino
            mejor_costo = costo_vecino
            print(f"    Hill Climbing: Mejora encontrada - Costo inicial={costo_inicial}, Nuevo costo={mejor_costo}")
            return mejor_solucion, mejor_costo

    # aqui se asegura que el costo no sea mayor que el inicial (mejorar)
    if mejor_costo > costo_inicial:
        print(f"    ¡Error! Hill Climbing retornó costo mayor: Inicial={costo_inicial}, Retornado={mejor_costo}")
        mejor_costo = costo_inicial
        mejor_solucion = list(solucion_inicial)

    return mejor_solucion, mejor_costo

def grasp_con_hc(D, aviones, matriz_tiempos, num_seeds_estocastico=10, max_iter_local=100, num_restarts_estocastico=5):
    mejor_solucion_global = None
    mejor_costo_global = float('inf')

    #1. caso del greedy determinista

    #obtencion de la solución del greedy
    secuencia_greedy_det, costo_greedy_det, _ = greedy_determinista(D, aviones, matriz_tiempos)
    print(f"Greedy Determinista: Costo = {costo_greedy_det}, Secuencia = {secuencia_greedy_det}")

    costo_verificado = calcular_costo(secuencia_greedy_det, aviones, matriz_tiempos)
    print(f"  Verificación: Costo calculado con calcular_costo = {costo_verificado}")
    if costo_verificado != costo_greedy_det:
        print(f"  ¡Advertencia! Costo inconsistente: Greedy={costo_greedy_det}, Calcular_costo={costo_verificado}")

    #comparar costo inicial
    if costo_greedy_det < mejor_costo_global:
        mejor_costo_global = costo_greedy_det
        mejor_solucion_global = list(secuencia_greedy_det)
        print(f"  Actualización: Nueva mejor solución global (Greedy Determinista) - Costo = {mejor_costo_global}")

    #aplicar hc
    solucion_local_det, costo_local_det = hill_climbing_alguna_mejora(secuencia_greedy_det, aviones, matriz_tiempos, max_iter_local)
    print(f"GRASP (alguna mejora) - Inicial (Greedy Determinista): Costo = {costo_local_det}, Secuencia = {solucion_local_det}")
    if costo_local_det < mejor_costo_global:
        mejor_costo_global = costo_local_det
        mejor_solucion_global = list(solucion_local_det)
        print(f"  Actualización: Nueva mejor solución global (Hill Climbing Determinista) - Costo = {mejor_costo_global}")

    ''' -------------------------------------------------------------------------------------------------------------- '''
    #2. caso del greedy estocastico

    #sol iniciales con el greedy estocástico
    print(f"\n--- Resultados GRASP (alguna mejora) - Iniciales (Greedy Estocástico) ---")
    for i in range(num_seeds_estocastico):
        secuencia_greedy_estoc, costo_greedy_estoc, _ = greedy_estocastico(D, aviones, matriz_tiempos, seed=i)
        print(f"\n  Seed = {i} - Solución Greedy Estocástica: Costo = {costo_greedy_estoc}, Secuencia = {secuencia_greedy_estoc}")

        costo_verificado = calcular_costo(secuencia_greedy_estoc, aviones, matriz_tiempos)
        print(f"    Verificación: Costo calculado con calcular_costo = {costo_verificado}")
        if costo_verificado != costo_greedy_estoc:
            print(f"    ¡Advertencia! Costo inconsistente: Greedy={costo_greedy_estoc}, Calcular_costo={costo_verificado}")
        #comparar costo inicial
        if costo_greedy_estoc < mejor_costo_global:
            mejor_costo_global = costo_greedy_estoc
            mejor_solucion_global = list(secuencia_greedy_estoc)
            print(f"    Actualización: Nueva mejor solución global (Greedy Estocástico, Seed={i}) - Costo = {mejor_costo_global}")

        #reincios
        for restart in range(num_restarts_estocastico):
            solucion_local_estoc, costo_local_estoc = hill_climbing_alguna_mejora(secuencia_greedy_estoc, aviones, matriz_tiempos, max_iter_local)
            print(f"    Reinicio {restart+1}: Costo = {costo_local_estoc}, Secuencia = {solucion_local_estoc}")
            if costo_local_estoc < mejor_costo_global:
                mejor_costo_global = costo_local_estoc
                mejor_solucion_global = list(solucion_local_estoc)
                print(f"    Actualización: Nueva mejor solución global (Hill Climbing Estocástico, Seed={i}, Reinicio={restart+1}) - Costo = {mejor_costo_global}")

    print("\n--- Resultados Finales GRASP (Hill Climbing - Alguna Mejora) ---")
    print(f"Mejor secuencia encontrada: {mejor_solucion_global}")
    print(f"Costo total de la mejor secuencia: {mejor_costo_global}")

    return mejor_solucion_global, mejor_costo_global

if __name__ == '__main__':
    D, aviones, matriz_tiempos = leer_datos_archivo("cases/case1.txt")

    mejor_secuencia_am, mejor_costo_am = grasp_con_hc(D, aviones, matriz_tiempos)