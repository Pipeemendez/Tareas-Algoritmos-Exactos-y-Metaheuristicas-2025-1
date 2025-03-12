comunas = [
    {"num": 1, "costo": 60},
    {"num": 2, "costo": 30},
    {"num": 3, "costo": 60},
    {"num": 4, "costo": 70},
    {"num": 5, "costo": 130},
    {"num": 6, "costo": 60},
    {"num": 7, "costo": 70},
    {"num": 8, "costo": 60},
    {"num": 9, "costo": 80},
    {"num": 10, "costo": 70},
    {"num": 11, "costo": 50},
    {"num": 12, "costo": 90},
    {"num": 13, "costo": 30},
    {"num": 14, "costo": 30},
    {"num": 15, "costo": 100},
]

comunas_ordenadas = sorted(comunas, key=lambda x: x["costo"])

for comuna in comunas_ordenadas:
    print(f"Número: {comuna['num']}, Costo: {comuna['costo']}")

#agregar vecinos a cada comuna, ej: {"num": 1, "costo": 60, [2,3,4,13]}

#recorrer el numero 2 y buscar su propiedad vecino, todos los vecinos los borramos y el costo del 2 lo guardamos en un array junto con la cantidad de nodos. Asi con todos los que vayan quedando, hasta tener un costo Total y ese costo Total lo agregamos a un arreglo, el cual al final buscaremos el menor.