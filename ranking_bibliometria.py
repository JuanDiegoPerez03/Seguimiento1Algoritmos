"""
Universidad del Quindío - Ingeniería de Sistemas y Computación
Análisis de Algoritmos en el Contexto de la Bibliometría
Taller de Seguimiento: Ranking de Términos con Algoritmos No Basados en Comparaciones
"""

import os
import re
import sys
import json
import time
import unicodedata
from pypdf import PdfReader
import matplotlib.pyplot as plt

# Directorio donde residen los 20 artículos
FOLDER = r"C:\Users\juand\OneDrive\Escritorio\Articulos_Algoritmos"

# Directorio de salida
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Lista de palabras vacías (stopwords) en inglés (generales y académicas estándar)
STOPWORDS = set([
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", 
    "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", 
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", 
    "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", 
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", 
    "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", 
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", 
    "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", 
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", 
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", 
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", 
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", 
    "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", 
    "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", 
    "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves",
    # Palabras comunes de redacción de papers
    "paper", "study", "research", "results", "using", "also", "based", "used", "review", 
    "article", "aims", "findings", "purpose", "method", "analysis", "data", "scoping", "systematic"
])

# ==============================================================================
# 1. EXTRACCIÓN Y PREPROCESAMIENTO DE TEXTO
# ==============================================================================

def normalize_text(text: str) -> str:
    """Normaliza texto eliminando ligaduras tipográficas (ej. fi, fl)."""
    return unicodedata.normalize("NFKD", text)

def extract_abstract(text: str) -> str:
    """Extrae el bloque del abstract identificando patrones de inicio y fin."""
    text = normalize_text(text)
    patterns = [
        r'(?:ABSTRACT|Abstract)\s*[\:\.\—\-]?\s*(.*?)(?=(?:(?:1\.?\s*)?Introduction|Keywords|Key words|INDEX TERMS|Index Terms|\n[0-9]+\.\s+[A-Z]))',
        r'A\s*B\s*S\s*T\s*R\s*A\s*C\s*T\s*(.*?)(?=(?:(?:1\.?\s*)?Introduction|\n[0-9]+\.\s+[A-Z]))',
        r'(?:abstract|a b s t r a c t)\s*[:\.\-]?\s*(.*?)(?=(?:\n\s*1\s+introduction|\n\s*1\.\s+introduction|\n\s*introduction))'
    ]
    for p in patterns:
        m = re.search(p, text, re.DOTALL | re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            # Limpiar posibles restos de tabla inicial
            candidate = re.sub(r'^.*?Keywords:.*?\n\n?', '', candidate, flags=re.DOTALL | re.IGNORECASE).strip()
            if len(candidate) > 100:
                return candidate
    return ""

def clean_and_tokenize(text: str):
    """Limpia el texto, elimina puntuación, convierte a minúsculas y filtra stopwords."""
    text = text.lower()
    text = normalize_text(text)
    # Extraer palabras de 3 o más caracteres
    words = re.findall(r'\b[a-z]{3,}\b', text)
    return [w for w in words if w not in STOPWORDS]

def load_corpus(folder_path: str):
    """Carga los 20 artículos y extrae los abstracts y metadatos."""
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".pdf")])
    corpus_metadata = []
    all_abstract_words = []

    print(f"[*] Leyendo {len(files)} artículos desde: {folder_path}...")
    for i, filename in enumerate(files):
        pdf_path = os.path.join(folder_path, filename)
        reader = PdfReader(pdf_path)
        first_pages_text = ""
        for page in reader.pages[:min(2, len(reader.pages))]:
            first_pages_text += page.extract_text() or ""
        
        abstract = extract_abstract(first_pages_text)
        words = clean_and_tokenize(abstract)
        all_abstract_words.extend(words)

        corpus_metadata.append({
            "id": i + 1,
            "filename": filename,
            "num_pages": len(reader.pages),
            "abstract_word_count": len(words)
        })

    # Frecuencia global de términos
    word_counts = {}
    for w in all_abstract_words:
        word_counts[w] = word_counts.get(w, 0) + 1

    freq_list = list(word_counts.items())
    return freq_list, corpus_metadata

# ==============================================================================
# 2. ALGORITMOS DE ORDENAMIENTO NO BASADOS EN COMPARACIONES
# ==============================================================================

def counting_sort_descending(arr):
    """
    Counting Sort para pares (palabra, frecuencia) en orden descendente.
    Complejidad: O(n + k), donde n = len(arr) y k = max_frecuencia.
    Algoritmo estable.
    """
    if not arr:
        return []
    
    n = len(arr)
    max_freq = max(item[1] for item in arr)
    
    # 1. Arreglo de conteo inicializado en ceros
    count = [0] * (max_freq + 1)
    output = [None] * n
    
    # 2. Conteo de frecuencias
    for item in arr:
        count[item[1]] += 1
        
    # 3. Acumulación inversa para orden DESCENDENTE
    # count[i] guardará el número de elementos con frecuencia >= i
    for i in range(max_freq - 1, -1, -1):
        count[i] += count[i + 1]
        
    # 4. Construcción estable (de derecha a izquierda)
    for i in range(n - 1, -1, -1):
        item = arr[i]
        freq = item[1]
        pos = count[freq] - 1
        output[pos] = item
        count[freq] -= 1
        
    return output


def _counting_sort_by_digit_descending(arr, exp):
    """
    Subrutina de Counting Sort sobre un dígito específico (base 10)
    acumulando de mayor a menor para mantener estabilidad y orden descendente.
    """
    n = len(arr)
    output = [None] * n
    count = [0] * 10  # Dígitos del 0 al 9 en base 10
    
    # 1. Contar ocurrencias del dígito actual
    for item in arr:
        digit = (item[1] // exp) % 10
        count[digit] += 1
        
    # 2. Acumulación inversa (descendente)
    # count[i] almacenará cuántos elementos tienen dígito >= i
    for i in range(8, -1, -1):
        count[i] += count[i + 1]
        
    # 3. Construcción estable (de derecha a izquierda)
    for i in range(n - 1, -1, -1):
        item = arr[i]
        digit = (item[1] // exp) % 10
        pos = count[digit] - 1
        output[pos] = item
        count[digit] -= 1
        
    return output

def radix_sort_descending(arr):
    """
    Radix Sort (LSD) descendente para pares (palabra, frecuencia).
    Ordena dígito por dígito desde las unidades hasta el dígito más significativo.
    Complejidad: O(d * (n + b)), donde d = num_dígitos(k) y b = base (10).
    Algoritmo estable.
    """
    if not arr:
        return []
    
    max_freq = max(item[1] for item in arr)
    exp = 1
    current = list(arr)
    
    # Iterar sobre cada posición decimal (unidades, decenas, centenas...)
    while max_freq // exp > 0:
        current = _counting_sort_by_digit_descending(current, exp)
        exp *= 10
        
    return current


def bucket_sort_descending(arr):
    """
    Bucket Sort por distribución directa para pares (palabra, frecuencia).
    Al ser las frecuencias enteros positivos discretos en rango [1..k],
    cada frecuencia tiene su cubeta directa, eliminando cualquier necesidad
    de comparaciones internas.
    Complejidad: O(n + k).
    """
    if not arr:
        return []
    
    max_freq = max(item[1] for item in arr)
    
    # 1. Crear cubetas (una por cada valor de frecuencia posible de 0 a k)
    buckets = [[] for _ in range(max_freq + 1)]
    
    # 2. Distribuir elementos en sus cubetas correspondientes
    for item in arr:
        buckets[item[1]].append(item)
        
    # 3. Concatenar cubetas en orden descendente (de k hacia abajo)
    output = []
    for freq in range(max_freq, -1, -1):
        for item in buckets[freq]:
            output.append(item)
            
    return output


# ==============================================================================
# 3. MEDICIÓN EMPÍRICA Y VALIDACIÓN
# ==============================================================================

def benchmark_algorithm(sort_fn, data, runs=50):
    """Mide el tiempo real promedio de ejecución en milisegundos (ms)."""
    # Calentamiento inicial
    sort_fn(list(data))
    
    times = []
    for _ in range(runs):
        data_copy = list(data)
        t_start = time.perf_counter()
        res = sort_fn(data_copy)
        t_end = time.perf_counter()
        times.append((t_end - t_start) * 1000.0) # Convertir a milisegundos
        
    avg_time = sum(times) / len(times)
    min_time = min(times)
    return avg_time, min_time, res

def main():
    print("=" * 70)
    print("   UNIVERSIDAD DEL QUINDÍO - ANÁLISIS DE ALGORITMOS EN BIBLIOMETRÍA   ")
    print("=" * 70)

    # 1. Cargar y preprocesar
    freq_list, metadata = load_corpus(FOLDER)
    n = len(freq_list)
    k = max(item[1] for item in freq_list)

    print(f"\n[+] Estadísticas del Corpus:")
    print(f"    - Total de artículos: {len(metadata)}")
    print(f"    - Vocabulario único (n): {n} términos")
    print(f"    - Frecuencia máxima (k): {k}")

    # Guardar metadatos en JSON para futuros cruces de información
    meta_file = os.path.join(OUTPUT_DIR, "corpus_metadatos.json")
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    print(f"    - Metadatos guardados en: {meta_file}")

    # 2. Ejecución y Medición Empírica
    print(f"\n[+] Ejecutando algoritmos de ordenamiento (50 repeticiones cada uno)...")
    
    t_cs_avg, t_cs_min, res_cs = benchmark_algorithm(counting_sort_descending, freq_list)
    t_rs_avg, t_rs_min, res_rs = benchmark_algorithm(radix_sort_descending, freq_list)
    t_bs_avg, t_bs_min, res_bs = benchmark_algorithm(bucket_sort_descending, freq_list)

    print("\n" + "=" * 55)
    print(f"{'Algoritmo':<20} | {'Tiempo Promedio (ms)':<16} | {'Mínimo (ms)':<12}")
    print("-" * 55)
    print(f"{'Counting Sort':<20} | {t_cs_avg:<16.4f} | {t_cs_min:<12.4f}")
    print(f"{'Radix Sort':<20} | {t_rs_avg:<16.4f} | {t_rs_min:<12.4f}")
    print(f"{'Bucket Sort':<20} | {t_bs_avg:<16.4f} | {t_bs_min:<12.4f}")
    print("=" * 55)

    # 3. Validación de Resultados: Top 15 términos
    top15_cs = res_cs[:15]
    top15_rs = res_rs[:15]
    top15_bs = res_bs[:15]

    print("\n[+] Validación de consistencia del Top 15:")
    print(f"    - Counting vs Radix : {'COINCIDEN EXACTAMENTE' if top15_cs == top15_rs else 'DISCREPANCIA'}")
    print(f"    - Counting vs Bucket: {'COINCIDEN EXACTAMENTE' if top15_cs == top15_bs else 'DISCREPANCIA'}")

    print("\n" + "=" * 50)
    print(f"{'Rango':<6} | {'Término':<25} | {'Frecuencia':<12}")
    print("-" * 50)
    for rank, (word, freq) in enumerate(top15_cs, 1):
        print(f"#{rank:<5} | {word:<25} | {freq:<12}")
    print("=" * 50)

    # 4. Generación de Gráfico con Matplotlib
    chart_file = os.path.join(OUTPUT_DIR, "top15_terminos.png")
    words_top15 = [w for w, _ in top15_cs]
    freqs_top15 = [f for _, f in top15_cs]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(words_top15[::-1], freqs_top15[::-1], color="#2b5b84", edgecolor="#1a364f")
    plt.xlabel("Frecuencia Absoluta en Abstracts", fontsize=11, fontweight="bold")
    plt.title("Top 15 Términos Más Frecuentes - Corpus IA Generativa\n(Universidad del Quindío)", fontsize=13, fontweight="bold", pad=15)
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    # Agregar etiqueta numérica al final de cada barra
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.8, bar.get_y() + bar.get_height()/2, f"{int(width)}", ha="left", va="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(chart_file, dpi=300)
    plt.close()
    print(f"\n[+] Gráfico guardado exitosamente en: {chart_file}")

if __name__ == "__main__":
    main()
