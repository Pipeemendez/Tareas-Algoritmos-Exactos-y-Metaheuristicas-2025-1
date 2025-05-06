from greedy_det import greedy_determinista
from greedy_est import greedy_estocastico
from leer_data import leer_datos_archivo
from HC_alguna_mejora import generar_vecino_intercambio, calcular_costo

#para mejorar una sol
def tabu_search(solucion_inicial, aviones, matriz_tiempos, tenure=5, max_iter=100, max_iter_sin_mejora=50):
    mejor_solucion_global = list(solucion_inicial)
    mejor_costo_global = calcular_costo(mejor_solucion_global, aviones, matriz_tiempos)
    solucion_actual = list(solucion_inicial)
    costo_actual = mejor_costo_global
    lista_tabu = []
    iter_sin_mejora = 0
    for iteracion in range(max_iter):
        mejor_vecino = None
        mejor_costo_vecino = float('inf')
        mejor_movimiento = None
        for _ in range(len(solucion_actual)):
            vecino, movimiento = generar_vecino_intercambio(solucion_actual, True)
            costo_vecino = calcular_costo(vecino, aviones, matriz_tiempos)
            if movimiento not in lista_tabu or costo_vecino < mejor_costo_global:
                if costo_vecino < mejor_costo_vecino:
                    mejor_vecino = vecino
                    mejor_costo_vecino = costo_vecino
                    mejor_movimiento = movimiento
        if mejor_vecino is None:
            continue
        solucion_actual = list(mejor_vecino)
        costo_actual = mejor_costo_vecino
        lista_tabu.append(mejor_movimiento)
        if len(lista_tabu) > tenure:
            lista_tabu.pop(0)
        if costo_actual < mejor_costo_global:
            mejor_solucion_global = list(solucion_actual)
            mejor_costo_global = costo_actual
            iter_sin_mejora = 0
            print(f"    Tabu Search: Nueva mejor solución - Iteración={iteracion+1}, Costo={mejor_costo_global}")
        else:
            iter_sin_mejora += 1
        if iter_sin_mejora >= max_iter_sin_mejora:
            print(f"    Tabu Search: Detenido por {max_iter_sin_mejora} iteraciones sin mejora")
            break
    return mejor_solucion_global, mejor_costo_global

#para probar varias soluciones iniciales, aplica tabu_search a cada una y se queda con la mejor.
def tabu_search_main(D, aviones, matriz_tiempos, tenure=5, max_iter=100, max_iter_sin_mejora=50, num_seeds_estocastico=10, num_restarts_estocastico=5):
    mejor_solucion_global = None
    mejor_costo_global = float('inf')

    #1. generar greedy determinista
    secuencia_greedy_det, costo_greedy_det, _ = greedy_determinista(D, aviones, matriz_tiempos)
    print(f"Greedy Determinista: Costo = {costo_greedy_det}, Secuencia = {secuencia_greedy_det}")
    costo_verificado = calcular_costo(secuencia_greedy_det, aviones, matriz_tiempos)
    print(f"  Verificación: Costo calculado con calcular_costo = {costo_verificado}")
    if costo_verificado != costo_greedy_det:
        print(f"  ¡Advertencia! Costo inconsistente: Greedy={costo_greedy_det}, Calcular_costo={costo_verificado}")
    if costo_greedy_det < mejor_costo_global:
        mejor_costo_global = costo_greedy_det
        mejor_solucion_global = list(secuencia_greedy_det)
        print(f"  Actualización: Nueva mejor solución global (Greedy Determinista) - Costo = {mejor_costo_global}")

    # aplicar tabu_search desde la sol del greedy creada
    solucion_tabu_det, costo_tabu_det = tabu_search(secuencia_greedy_det, aviones, matriz_tiempos, tenure, max_iter, max_iter_sin_mejora)
    print(f"Tabu Search (Greedy Determinista): Costo = {costo_tabu_det}, Secuencia = {solucion_tabu_det}")
    if costo_tabu_det < mejor_costo_global:
        mejor_costo_global = costo_tabu_det
        mejor_solucion_global = list(solucion_tabu_det)
        print(f"  Actualización: Nueva mejor solución global (Tabu Search Determinista) - Costo = {mejor_costo_global}")
    
    ''' ------------------------------------------------------------------------------------------------------------------------------- '''

    #2. greedy est con reinicios
    print(f"\n--- Resultados Tabu Search (Greedy Estocástico) ---")
    for i in range(num_seeds_estocastico):
        secuencia_greedy_estoc, costo_greedy_estoc, _ = greedy_estocastico(D, aviones, matriz_tiempos, seed=i)
        print(f"\n  Seed = {i} - Solución Greedy Estocástica: Costo = {costo_greedy_estoc}, Secuencia = {secuencia_greedy_estoc}")
        costo_verificado = calcular_costo(secuencia_greedy_estoc, aviones, matriz_tiempos)
        print(f"    Verificación: Costo calculado con calcular_costo = {costo_verificado}")
        if costo_verificado != costo_greedy_estoc:
            print(f"    ¡Advertencia! Costo inconsistente: Greedy={costo_greedy_estoc}, Calcular_costo={costo_verificado}")
        if costo_greedy_estoc < mejor_costo_global:
            mejor_costo_global = costo_greedy_estoc
            mejor_solucion_global = list(secuencia_greedy_estoc)
            print(f"    Actualización: Nueva mejor solución global (Greedy Estocástico, Seed={i}) - Costo = {mejor_costo_global}")
        for restart in range(num_restarts_estocastico):
            solucion_tabu_estoc, costo_tabu_estoc = tabu_search(secuencia_greedy_estoc, aviones, matriz_tiempos, tenure, max_iter, max_iter_sin_mejora)
            print(f"    Reinicio {restart+1}: Costo = {costo_tabu_estoc}, Secuencia = {solucion_tabu_estoc}")
            if costo_tabu_estoc < mejor_costo_global:
                mejor_costo_global = costo_tabu_estoc
                mejor_solucion_global = list(solucion_tabu_estoc)
                print(f"    Actualización: Nueva mejor solución global (Tabu Search Estocástico, Seed={i}, Reinicio={restart+1}) - Costo = {mejor_costo_global}")
    print("\n--- Resultados Finales Tabu Search ---")
    print(f"Mejor secuencia encontrada: {mejor_solucion_global}")
    print(f"Costo total de la mejor secuencia: {mejor_costo_global}")
    return mejor_solucion_global, mejor_costo_global

if __name__ == '__main__':
    D, aviones, matriz_tiempos = leer_datos_archivo("cases/case1.txt")
    configuraciones = [
        {"tenure": 5, "max_iter": 100, "max_iter_sin_mejora": 50, "num_seeds_estocastico": 10, "num_restarts_estocastico": 5},
        {"tenure": 10, "max_iter": 200, "max_iter_sin_mejora": 100, "num_seeds_estocastico": 5, "num_restarts_estocastico": 3},
        {"tenure": 3, "max_iter": 50, "max_iter_sin_mejora": 25, "num_seeds_estocastico": 10, "num_restarts_estocastico": 5},
        {"tenure": 7, "max_iter": 150, "max_iter_sin_mejora": 75, "num_seeds_estocastico": 8, "num_restarts_estocastico": 4},
        {"tenure": 15, "max_iter": 300, "max_iter_sin_mejora": 150, "num_seeds_estocastico": 3, "num_restarts_estocastico": 2}
    ]
    for idx, config in enumerate(configuraciones, 1):
        print(f"\n=== Configuración {idx} ===\nParámetros: tenure={config['tenure']}, max_iter={config['max_iter']}, "
              f"max_iter_sin_mejora={config['max_iter_sin_mejora']}, num_seeds_estocastico={config['num_seeds_estocastico']}, "
              f"num_restarts_estocastico={config['num_restarts_estocastico']}")
        mejor_secuencia, mejor_costo = tabu_search_main(
            D, aviones, matriz_tiempos,
            tenure=config['tenure'],
            max_iter=config['max_iter'],
            max_iter_sin_mejora=config['max_iter_sin_mejora'],
            num_seeds_estocastico=config['num_seeds_estocastico'],
            num_restarts_estocastico=config['num_restarts_estocastico']
        )
        print(f"Resultado Configuración {idx}: Costo = {mejor_costo}, Secuencia = {mejor_secuencia}")