import time

# Datos del problema
comunas = list(range(1, 16))  # Comunas 1 a 15
costos = {1: 60, 2: 30, 3: 60, 4: 70, 5: 130, 6: 60, 7: 70, 8: 60, 9: 80, 
          10: 70, 11: 50, 12: 90, 13: 30, 14: 30, 15: 100}

# Matriz de cobertura completa (supuesta basada en el texto y lógica razonable)
cobertura = {
    1: [1, 2, 3, 4, 13], 2: [1, 2, 4, 12, 15], 3: [1, 3, 4, 13], 4: [1, 2, 3, 4, 5, 12],
    5: [3, 4, 5, 6, 7, 8, 9, 12], 6: [3, 5, 6, 9], 7: [5, 7, 8, 10, 11, 12, 14, 15], 8: [5, 7, 8, 9, 10],
    9: [5, 6, 8, 9, 10, 11], 10: [7, 8, 9, 10, 11], 11: [7, 9, 10, 11, 14], 12: [2, 4, 5, 7, 12, 15],
    13: [1, 3, 13], 14: [7, 11, 14, 15], 15: [2, 7, 12, 14, 15]
}

# Verificar si todas las comunas están cubiertas
def todas_cubiertas(solucion, comunas_totales):
    cubiertas = set()
    for comuna in solucion:
        cubiertas.update(cobertura[comuna])
    return set(comunas_totales).issubset(cubiertas)

# Forward-checking: verifica si las comunas restantes pueden cubrir lo que falta
def forward_check(solucion_parcial, comunas_restantes):
    cubiertas = set()
    for comuna in solucion_parcial:
        cubiertas.update(cobertura[comuna])
    no_cubiertas = set(comunas) - cubiertas
    if not no_cubiertas:
        return True
    for comuna in comunas_restantes:
        if no_cubiertas.issubset(set().union(*[cobertura[c] for c in comunas_restantes])):
            return True
    return False

# Técnica completa con forward-checking
def resolver_completo(comunas_restantes, solucion_parcial, mejor_solucion, costo_actual, historial):
    global mejor_costo
    if todas_cubiertas(solucion_parcial, comunas):
        costo = sum(costos[c] for c in solucion_parcial)
        if costo < mejor_costo:
            mejor_costo = costo
            mejor_solucion[:] = solucion_parcial[:]
            historial.append((time.time(), costo))
        return
    
    if not comunas_restantes or costo_actual >= mejor_costo:
        return

    comuna = comunas_restantes[0]
    resto = comunas_restantes[1:]
    
    # Incluir la comuna
    solucion_parcial.append(comuna)
    if forward_check(solucion_parcial, resto):
        resolver_completo(resto, solucion_parcial, mejor_solucion, costo_actual + costos[comuna], historial)
    solucion_parcial.pop()
    
    # No incluir la comuna
    if forward_check(solucion_parcial, resto):
        resolver_completo(resto, solucion_parcial, mejor_solucion, costo_actual, historial)

# Heurística: selecciona comuna con mayor cobertura/costo
def heuristica(comunas_restantes, solucion_parcial):
    cubiertas = set()
    for comuna in solucion_parcial:
        cubiertas.update(cobertura[comuna])
    no_cubiertas = set(comunas) - cubiertas
    if not no_cubiertas:
        return None
    return max(comunas_restantes, 
               key=lambda c: len(set(cobertura[c]) & no_cubiertas) / costos[c], 
               default=None)

# Variante con heurística
def resolver_heuristico(comunas_restantes, solucion_parcial, mejor_solucion, costo_actual, historial):
    global mejor_costo
    if todas_cubiertas(solucion_parcial, comunas):
        costo = sum(costos[c] for c in solucion_parcial)
        if costo < mejor_costo:
            mejor_costo = costo
            mejor_solucion[:] = solucion_parcial[:]
            historial.append((time.time(), costo))
        return
    
    if not comunas_restantes or costo_actual >= mejor_costo:
        return

    comuna = heuristica(comunas_restantes, solucion_parcial)
    if comuna is None:
        return
    resto = [c for c in comunas_restantes if c != comuna]
    
    # Incluir la comuna
    solucion_parcial.append(comuna)
    if forward_check(solucion_parcial, resto):
        resolver_heuristico(resto, solucion_parcial, mejor_solucion, costo_actual + costos[comuna], historial)
    solucion_parcial.pop()
    
    # No incluir la comuna
    if forward_check(solucion_parcial, resto):
        resolver_heuristico(resto, solucion_parcial, mejor_solucion, costo_actual, historial)

# Ejecución
mejor_costo = float('inf')
mejor_solucion_completo = []
historial_completo = []
inicio = time.time()
resolver_completo(comunas, [], mejor_solucion_completo, 0, historial_completo)
tiempo_completo = time.time() - inicio

mejor_costo = float('inf')
mejor_solucion_heuristico = []
historial_heuristico = []
inicio = time.time()
resolver_heuristico(comunas, [], mejor_solucion_heuristico, 0, historial_heuristico)
tiempo_heuristico = time.time() - inicio

# Resultados
print("Técnica Completa (Forward-checking):")
print(f"Solución: {mejor_solucion_completo}")
print(f"Costo: {sum(costos[c] for c in mejor_solucion_completo)}")
print(f"Tiempo: {tiempo_completo:.4f} segundos")
print("Evolución del costo:", [(t - historial_completo[0][0], c) for t, c in historial_completo] if historial_completo else [])

print("\nVariante con Heurística:")
print(f"Solución: {mejor_solucion_heuristico}")
print(f"Costo: {sum(costos[c] for c in mejor_solucion_heuristico)}")
print(f"Tiempo: {tiempo_heuristico:.4f} segundos")
print("Evolución del costo:", [(t - historial_heuristico[0][0], c) for t, c in historial_heuristico] if historial_heuristico else [])