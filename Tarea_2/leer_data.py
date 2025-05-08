def leer_datos_archivo(path):
    with open(path, 'r') as file:
        lines = [line.strip() for line in file if line.strip()]

    D = int(lines[0])
    aviones = []
    matriz_tiempos = []
    idx = 1
    for _ in range(D):
        E, P, L, C1, C2 = lines[idx].split()
        avion = {
            't_temprano': int(E),
            't_pref': int(P),
            't_tarde': int(L),
            'pena_temprano': float(C1),
            'pena_tarde': float(C2)
        }
        aviones.append(avion)

        separacion = list(map(int, lines[idx + 1].split() + lines[idx + 2].split()))
        matriz_tiempos.append(separacion)

        idx += 3

    return D, aviones, matriz_tiempos