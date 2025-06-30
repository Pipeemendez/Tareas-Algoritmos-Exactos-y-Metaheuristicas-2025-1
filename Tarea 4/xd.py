import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split # Aunque usaremos splits fijos
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import random
import time
import math
import warnings

# Ignorar warnings para mayor claridad en la salida
warnings.filterwarnings('ignore')

# --- 1. Carga y Pre-procesamiento del Dataset ---
def load_and_preprocess_data(train_path, test_path):
    print("Cargando datos...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print("Datos cargados.")

    # Combinar para asegurar el mismo pre-procesamiento
    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    # Identificar características y la variable objetivo (label binaria)
    # Según el paper, el label binario es 'label' (0 para normal, 1 para ataque)
    # Eliminamos 'id' y 'attack_cat' ya que 'label' es la objetivo binaria
    X = combined_df.drop(['id', 'attack_cat', 'label'], axis=1)
    y = combined_df['label']

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

    # Separar de nuevo en conjuntos de entrenamiento y prueba usando los índices originales
    # Asumiendo que los archivos train_df y test_df originales mantenían el orden
    train_size = len(train_df)
    X_train_processed = X_processed[:train_size]
    X_test_processed = X_processed[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]

    print("Datos pre-procesados y separados (train/test).")
    return X_train_processed, X_test_processed, y_train, y_test, X.columns, preprocessor.transformers_[1][1].get_feature_names_out(categorical_features)

# --- 2. Definición de la Fitness Function (Wrapper) ---

def fitness_function(individual, X_train, X_test, y_train, y_test):
    # Individual es un vector binario (numpy array)
    selected_features_indices = np.where(individual == 1)[0]
    num_selected_features = len(selected_features_indices)
    total_features = len(individual)

    # Penalizar si no se selecciona ninguna característica o si se seleccionan todas
    if num_selected_features == 0 or num_selected_features == total_features:
        return 0.0 # Fitness muy bajo

    # Seleccionar solo las características indicadas
    X_train_selected = X_train[:, selected_features_indices]
    X_test_selected = X_test[:, selected_features_indices]

    # Entrenar y evaluar el clasificador (Random Forest)
    # Parametros de RF tunneados en el paper (Table 6), usaremos valores razonables por simplicidad
    # n_estimators=100 (por defecto), max_depth=4, min_samples_leaf=1, min_samples_split=2
    # Vamos a usar n_estimators=50, max_depth=4 para una evaluación más rápida dentro del fitness
    classifier = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42, n_jobs=-1)
    classifier.fit(X_train_selected, y_train)
    y_pred = classifier.predict(X_test_selected)

    # Calcular la métrica de rendimiento (Accuracy)
    accuracy = accuracy_score(y_test, y_pred)

    # Calcular el fitness (Maximizar Accuracy, Minimizar Features)
    # w1=1, w2=0.05 (penalización pequeña por característica)
    w1 = 1.0
    w2 = 0.05
    fitness = w1 * accuracy - w2 * (num_selected_features / total_features)

    return fitness

# --- 3. Implementación del Algoritmo Binary Differential Evolution (BDE) ---

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def run_bde(X_train, X_test, y_train, y_test, pop_size, num_generations, F, CR, total_features):
    print(f"Ejecutando Binary Differential Evolution (BDE) con {pop_size} individuos por {num_generations} generaciones...")

    # Inicialización: Crear una población de individuos binarios aleatorios
    population = np.random.randint(0, 2, size=(pop_size, total_features))
    fitness_scores = np.array([fitness_function(ind, X_train, X_test, y_train, y_test) for ind in population])

    best_individual = population[np.argmax(fitness_scores)].copy()
    best_fitness = np.max(fitness_scores)
    history_best_fitness = [best_fitness]

    start_time = time.time()

    for gen in range(num_generations):
        new_population = np.zeros((pop_size, total_features), dtype=int)
        new_fitness_scores = np.zeros(pop_size)

        for i in range(pop_size):
            # Seleccionar tres individuos distintos aleatorios (a, b, c)
            indices = [idx for idx in range(pop_size) if idx != i]
            a, b, c = population[random.sample(indices, 3)]

            # Mutación y Crossover (Adaptado para binario/probabilidad)
            # Inspirado en BDE - combina elementos de DE con transformación binaria
            trial_vector = np.zeros(total_features)
            for j in range(total_features):
                # Crossover: Decide si usar el vector mutante o el original
                if random.random() < CR:
                    # Mutación continua conceptual: a + F * (b - c)
                    # Interpretación probabilística: sigmoid de una "diferencia"
                    # Simplificado: Influencia de los padres a, b, c en la probabilidad de ser 1
                    v_j = sigmoid(a[j] + F * (b[j] - c[j])) # Resultado continuo
                    # Transformación a binario: Convertir la probabilidad en un valor binario
                    if random.random() < v_j:
                        trial_vector[j] = 1
                    else:
                        trial_vector[j] = 0
                else:
                    trial_vector[j] = population[i, j] # Usar el valor original

            # Asegurar que el vector trial sea binario (puede haber floats por la sigmoide si no se discretiza bien)
            # En esta implementación, trial_vector ya es binario {0, 1} por la transformación probabilística

            # Evaluación del vector trial
            trial_fitness = fitness_function(trial_vector, X_train, X_test, y_train, y_test)

            # Selección: Reemplazar si el vector trial es mejor
            if trial_fitness > fitness_scores[i]:
                new_population[i] = trial_vector
                new_fitness_scores[i] = trial_fitness
            else:
                new_population[i] = population[i]
                new_fitness_scores[i] = fitness_scores[i]

        population = new_population
        fitness_scores = new_fitness_scores

        # Actualizar el mejor individuo global
        current_best_fitness = np.max(fitness_scores)
        current_best_individual = population[np.argmax(fitness_scores)].copy()

        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = current_best_individual.copy()

        history_best_fitness.append(best_fitness)

        if (gen + 1) % 10 == 0:
            elapsed_time = time.time() - start_time
            print(f"Generación {gen+1}/{num_generations}, Mejor Fitness: {best_fitness:.4f}, Tiempo: {elapsed_time:.2f}s")

    end_time = time.time()
    print("\nBDE finalizado.")
    print(f"Tiempo total de ejecución BDE: {end_time - start_time:.2f}s")

    return best_individual, best_fitness, history_best_fitness

# --- 4. Evaluación del Mejor Resultado Final ---

def evaluate_final_model(best_individual, X_train, X_test, y_train, y_test, original_features, cat_feature_names):
    selected_features_indices = np.where(best_individual == 1)[0]
    num_selected_features = len(selected_features_indices)
    total_features = len(best_individual)

    print("\n--- Evaluación del Mejor Subconjunto de Características ---")
    print(f"Características seleccionadas: {num_selected_features}/{total_features}")

    if num_selected_features == 0:
        print("No se seleccionó ninguna característica. No se puede evaluar el modelo final.")
        return

    # Seleccionar solo las características del mejor individuo
    X_train_selected = X_train[:, selected_features_indices]
    X_test_selected = X_test[:, selected_features_indices]

    # Entrenar el clasificador final con los parámetros tunneados del paper (Table 6)
    # Usaremos los valores de la Table 6 del paper para RF
    final_classifier = RandomForestClassifier(n_estimators=500, # Más robusto que en fitness
                                            max_depth=4,
                                            min_samples_leaf=1,
                                            min_samples_split=2,
                                            random_state=42,
                                            n_jobs=-1)

    print("Entrenando clasificador final...")
    final_classifier.fit(X_train_selected, y_train)
    print("Evaluando clasificador final en el test set...")
    y_pred = final_classifier.predict(X_test_selected)

    # Calcular métricas de evaluación
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Para métricas como DR, PR, F1, FPR necesitamos TP, TN, FP, FN
    # En clasificación binaria (0, 1):
    # TN: Actual 0, Predicted 0
    # FP: Actual 0, Predicted 1
    # FN: Actual 1, Predicted 0
    # TP: Actual 1, Predicted 1

    TN, FP, FN, TP = cm.ravel()

    dr = recall_score(y_test, y_pred) # DR (Detection Rate) = Recall = TP / (TP + FN)
    pr = precision_score(y_test, y_pred) # PR (Precision) = TP / (TP + FP)
    f1 = f1_score(y_test, y_pred)     # F1-Score = 2 * (PR * DR) / (PR + DR)

    # FPR (False Positive Rate) = FP / (FP + TN)
    # Manejar división por cero si TN + FP = 0 (caso raro en datasets grandes)
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0

    print("\nMétricas de rendimiento:")
    print(f"Accuracy (ACC): {acc:.4f}")
    print(f"Detection Rate (DR): {dr:.4f}")
    print(f"Precision (PR): {pr:.4f}")
    print(f"F1-Score (F1): {f1:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print(f"Número de características seleccionadas: {num_selected_features}")

    # Opcional: Mostrar los nombres de las características seleccionadas
    # Esto es más complejo porque el preprocesamiento cambia los nombres
    # Podemos obtener los nombres de las columnas del X_processed si construimos un DataFrame
    # O mostrar un subconjunto de los nombres originales (menos preciso post-OHE)

    # Para obtener los nombres post-OHE y scaling es más complicado sin el pipeline completo
    # Simplificaremos imprimiendo los índices de las columnas seleccionadas
    print(f"Índices de las características seleccionadas: {selected_features_indices.tolist()}")


# --- Main Execution ---

if __name__ == "__main__":
    # IMPORTANT: Download the datasets first
    # UNSW-NB15_training-set.csv and UNSW-NB15_testing-set.csv
    # You can find them online, e.g., on Kaggle or the UNSW website
    train_file = 'Tarea 4/UNSW_NB15_training-set.csv' # <--- UPDATE THIS PATH
    test_file = 'Tarea 4/UNSW_NB15_testing-set.csv'   # <--- UPDATE THIS PATH

    # Verificar si los archivos existen
    import os
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print(f"ERROR: Archivos de dataset no encontrados en '{train_file}' y '{test_file}'.")
        print("Por favor, descarga los datasets UNSW-NB15 (UNSW_NB15_training-set.csv y UNSW_NB15_testing-set.csv)")
        print("y actualiza las rutas de los archivos en el código.")
    else:
        X_train, X_test, y_train, y_test, original_features, cat_feature_names = load_and_preprocess_data(train_file, test_file)

        # Parámetros para BDE (ajustar según rendimiento, estos son ejemplos iniciales)
        # El paper no especifica parámetros de BGSA/BGWO, estos son valores típicos para EAs
        bde_pop_size = 30
        bde_num_generations = 100 # Número de generaciones, ajustar para más tiempo de búsqueda
        bde_F = 0.8 # Factor de escalado (influencia de la diferencia)
        bde_CR = 0.9 # Probabilidad de crossover (usar vector mutante)
        total_features_count = X_train.shape[1] # Número total de características después del pre-procesamiento

        # Ejecutar BDE para encontrar el mejor subconjunto de características
        best_features_binary_vector, final_best_fitness, fitness_history = run_bde(
            X_train, X_test, y_train, y_test,
            bde_pop_size, bde_num_generations, bde_F, bde_CR,
            total_features_count
        )

        # Evaluar el modelo final con el mejor subconjunto
        # Pasamos los nombres originales de las características para referencia,
        # aunque los índices seleccionados corresponden al espacio pre-procesado.
        evaluate_final_model(best_features_binary_vector, X_train, X_test, y_train, y_test, original_features, cat_feature_names)

        # Opcional: Graficar la convergencia del fitness (requiere matplotlib)
        try:
            import matplotlib.pyplot as plt
            plt.plot(fitness_history)
            plt.xlabel('Generación')
            plt.ylabel('Mejor Fitness (Accuracy Penalizada)')
            plt.title('Convergencia de BDE')
            plt.grid(True)
            plt.show()
        except ImportError:
            print("\nInstala matplotlib para ver el gráfico de convergencia: pip install matplotlib")