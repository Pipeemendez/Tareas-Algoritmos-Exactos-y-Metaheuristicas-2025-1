import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import random
import time
import math # Aunque no usaremos sigmoid/exp en GA binario directo
import warnings
import matplotlib.pyplot as plt # Para graficar convergencia

# Ignorar warnings para mayor claridad en la salida
warnings.filterwarnings('ignore')

# --- 1. Carga y Pre-procesamiento del Dataset ---

def load_and_preprocess_data(train_path, test_path):
    print("Cargando datos...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print("Datos cargados.")

    # Combinar para asegurar el mismo pre-procesamiento y obtener el esquema completo de columnas
    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    # Identificar características y la variable objetivo (label binaria)
    # Según el paper, el label binario es 'label' (0 para normal, 1 para ataque)
    # Eliminamos 'id' y 'attack_cat' ya que 'label' es la objetivo binaria
    X = combined_df.drop(['id', 'attack_cat', 'label'], axis=1)
    y = combined_df['label']

    print(f"Número de características originales antes del pre-procesamiento: {X.shape[1]}")

    # Identificar columnas categóricas y numéricas
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=np.number).columns

    # Crear preprocesador
    # One-Hot Encoding para categóricas, Min-Max Scaling para numéricas
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])

    print("Aplicando pre-procesamiento...")
    # Ajustar y transformar los datos
    X_processed = preprocessor.fit_transform(X)
    print(f"Dimensiones de los datos pre-procesados: {X_processed.shape}")
    print(f"Número de características después del pre-procesamiento (espacio de búsqueda para la metaheurística): {X_processed.shape[1]}")

    # Separar de nuevo en conjuntos de entrenamiento y prueba usando los índices originales
    train_size = len(train_df)
    X_train_processed = X_processed[:train_size]
    X_test_processed = X_processed[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]

    print("Datos pre-procesados y separados (train/test).")

    return X_train_processed, X_test_processed, y_train, y_test, X.columns, categorical_features

# --- 2. Definición de la Fitness Function (Wrapper) ---

def fitness_function(individual, X_train, X_test, y_train, y_test):
    # Individual es un vector binario (numpy array) que representa el subconjunto de características
    selected_features_indices = np.where(individual == 1)[0]
    num_selected_features = len(selected_features_indices)
    total_features = len(individual)

    # Penalizar fuertemente si no se selecciona ninguna característica o si se seleccionan todas
    if num_selected_features == 0 or num_selected_features == total_features:
         return -1.0 # Fitness muy bajo

    # Seleccionar solo las características indicadas del dataset pre-procesado
    X_train_selected = X_train[:, selected_features_indices]
    X_test_selected = X_test[:, selected_features_indices]

    # Entrenar y evaluar el clasificador (Random Forest)
    # Usaremos n_estimators=50, max_depth=4 para una evaluación más rápida dentro del fitness
    classifier = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42, n_jobs=-1)
    classifier.fit(X_train_selected, y_train)
    y_pred = classifier.predict(X_test_selected)

    # Calcular la métrica de rendimiento (Accuracy)
    accuracy = accuracy_score(y_test, y_pred)

    # Calcular el fitness (Maximizar Accuracy, Minimizar Features)
    # w1=1.0, w2=0.05 (penalización pequeña por característica)
    w1 = 1.0
    w2 = 0.05
    fitness = w1 * accuracy - w2 * (num_selected_features / total_features)

    return fitness

# --- 3. Implementación de las operaciones de GA ---

def initialize_population(pop_size, total_features):
    population = []
    for _ in range(pop_size):
        individual = np.random.randint(0, 2, size=total_features)
        # Asegurar que el individuo no sea todo ceros o todo unos
        while np.sum(individual) == 0 or np.sum(individual) == total_features:
             individual = np.random.randint(0, 2, size=total_features)
        population.append(individual)
    return np.array(population)

def select_parents_tournament(population, fitness_scores, num_parents, tournament_size):
    parents = []
    pop_size = len(population)
    for _ in range(num_parents):
        # Seleccionar 'tournament_size' individuos aleatorios para el torneo
        tournament_indices = random.sample(range(pop_size), tournament_size)
        tournament_fitness = fitness_scores[tournament_indices]
        # Elige al individuo con el mejor fitness del torneo como padre
        winner_index_in_tournament = np.argmax(tournament_fitness)
        winner_original_index = tournament_indices[winner_index_in_tournament]
        parents.append(population[winner_original_index])
    return parents # Devuelve una lista de individuos padres (como numpy arrays)

def crossover_one_point(parent1, parent2):
    total_features = len(parent1)
    # Elegir un punto de cruzamiento aleatorio (evitando los extremos 0 y total_features)
    point = random.randint(1, total_features - 1)

    # Crear descendientes
    offspring1 = np.concatenate((parent1[:point], parent2[point:]))
    offspring2 = np.concatenate((parent2[:point], parent1[point:]))

    return offspring1, offspring2

def mutate(individual, mutation_rate):
    total_features = len(individual)
    mutated_individual = individual.copy()
    for i in range(total_features):
        # Mutar cada bit con una probabilidad 'mutation_rate'
        if random.random() < mutation_rate:
            mutated_individual[i] = 1 - mutated_individual[i] # Flip el bit (0->1, 1->0)

    # Asegurar que la mutación no resulte en un individuo inválido (todo ceros o todo unos)
    if np.sum(mutated_individual) == 0 or np.sum(mutated_individual) == total_features:
         # Si la mutación creó un individuo inválido, lo revertimos o generamos uno válido aleatorio
         # Revertir es más simple aquí:
         return individual # Devuelve el original si la mutación lo invalidó
    else:
         return mutated_individual


# --- 4. Implementación del Algoritmo Genético (GA) ---

def run_ga(X_train, X_test, y_train, y_test, pop_size, num_generations, crossover_rate, mutation_rate, tournament_size, num_elite, total_features):
    print(f"\nEjecutando Algoritmo Genético (GA) con {pop_size} individuos por {num_generations} generaciones...")
    print(f"Espacio de búsqueda binario tiene {total_features} dimensiones.")
    print(f"Crossover Rate: {crossover_rate}, Mutation Rate (per bit): {mutation_rate}, Tournament Size: {tournament_size}, Elite: {num_elite}")

    # Inicialización de la población
    population = initialize_population(pop_size, total_features)

    # Evaluación inicial
    fitness_scores = np.array([fitness_function(ind, X_train, X_test, y_train, y_test) for ind in population])

    # Rastrear el mejor individuo global
    best_individual_idx = np.argmax(fitness_scores)
    best_individual = population[best_individual_idx].copy()
    best_fitness = fitness_scores[best_individual_idx]
    history_best_fitness = [best_fitness]

    start_time = time.time()

    # Bucle principal de generaciones
    for gen in range(num_generations):
        next_population = []

        # Elitismo: Preservar los mejores individuos
        elite_indices = np.argsort(fitness_scores)[-num_elite:] # Índices de los mejores
        for idx in elite_indices:
            next_population.append(population[idx].copy()) # Añadir copias de los élite

        # Rellenar el resto de la próxima generación a través de selección, cruzamiento y mutación
        while len(next_population) < pop_size:
            # Selección de dos padres usando Torneo
            parent1, parent2 = select_parents_tournament(population, fitness_scores, 2, tournament_size)

            # Aplicar Cruzamiento con probabilidad
            if random.random() < crossover_rate:
                offspring1, offspring2 = crossover_one_point(parent1, parent2)
            else:
                # Si no hay cruzamiento, los descendientes son copias de los padres (o uno de ellos)
                # Aquí, por simplicidad, creamos dos copias del padre 1
                offspring1, offspring2 = parent1.copy(), parent1.copy()

            # Aplicar Mutación a cada descendiente
            offspring1 = mutate(offspring1, mutation_rate)
            offspring2 = mutate(offspring2, mutation_rate)

            # Añadir descendientes a la próxima población
            next_population.append(offspring1)
            if len(next_population) < pop_size: # Asegurar no exceder el tamaño de la población
                next_population.append(offspring2)

        # Asegurarse de que la próxima población sea exactamente del tamaño pop_size
        next_population = np.array(next_population[:pop_size])

        # Reemplazar la población actual con la nueva
        population = next_population

        # Evaluar la nueva población
        fitness_scores = np.array([fitness_function(ind, X_train, X_test, y_train, y_test) for ind in population])

        # Actualizar el mejor individuo global
        current_best_individual_idx = np.argmax(fitness_scores)
        current_best_fitness = fitness_scores[current_best_individual_idx]
        current_best_individual = population[current_best_individual_idx].copy()

        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = current_best_individual.copy()

        history_best_fitness.append(best_fitness)

        # Imprimir progreso
        if (gen + 1) % 10 == 0 or gen == num_generations - 1:
             elapsed_time = time.time() - start_time
             num_selected = np.sum(best_individual)
             print(f"Generación {gen+1}/{num_generations}, Mejor Fitness: {best_fitness:.4f}, Features: {num_selected}/{total_features}, Tiempo: {elapsed_time:.2f}s")

    end_time = time.time()
    print("\nGA finalizado.")
    print(f"Tiempo total de ejecución GA: {end_time - start_time:.2f}s")

    return best_individual, best_fitness, history_best_fitness

# --- 5. Evaluación del Mejor Resultado Final (Reutilizamos la función anterior) ---
# La función evaluate_final_model ya está definida y funciona con el vector binario del mejor individuo

# --- Main Execution ---

if __name__ == "__main__":
    # IMPORTANT: Download the datasets first
    train_file = 'Tarea 4/UNSW_NB15_training-set.csv' # <--- ACTUALIZA ESTA RUTA
    test_file = 'Tarea 4/UNSW_NB15_testing-set.csv'   # <--- ACTUALIZA ESTA RUTA

    # Verificar si los archivos existen
    import os
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print(f"ERROR: Archivos de dataset no encontrados en '{train_file}' y '{test_file}'.")
        print("Por favor, descarga los datasets UNSW-NB15 (training-set.csv y testing-set.csv)")
        print("y actualiza las rutas de los archivos en el código.")
    else:
        # Carga y pre-procesamiento
        X_train, X_test, y_train, y_test, original_features, categorical_features = load_and_preprocess_data(train_file, test_file)

        # Número total de características después del pre-procesamiento
        total_features_count = X_train.shape[1]

        # Parámetros para el Algoritmo Genético (GA)
        ga_pop_size = 50          # Tamaño de la población
        ga_num_generations = 100  # Número de generaciones
        ga_crossover_rate = 0.9   # Probabilidad de cruzamiento
        ga_mutation_rate = 0.01   # Probabilidad de mutación (por bit)
        ga_tournament_size = 5    # Tamaño del torneo para selección
        ga_num_elite = 1          # Número de élite a preservar

        # Ejecutar GA para encontrar el mejor subconjunto de características
        best_features_binary_vector, final_best_fitness, fitness_history = run_ga(
            X_train, X_test, y_train, y_test,
            ga_pop_size, ga_num_generations, ga_crossover_rate, ga_mutation_rate,
            ga_tournament_size, ga_num_elite,
            total_features_count
        )

        # Evaluar el modelo final con el mejor subconjunto
        evaluate_final_model(best_features_binary_vector, X_train, X_test, y_train, y_test, original_features, categorical_features)

        # Graficar la convergencia del fitness
        plt.figure(figsize=(10, 6))
        plt.plot(fitness_history)
        plt.xlabel('Generación')
        plt.ylabel('Mejor Fitness (Accuracy Penalizada)')
        plt.title('Convergencia de GA')
        plt.grid(True)
        plt.show()