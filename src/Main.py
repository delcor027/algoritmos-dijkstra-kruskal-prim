import heapq
import time
from collections import defaultdict
import pandas as pd
from tabulate import tabulate

# Função para medir tempo de execução
def measure_time(func, *args):
    start = time.time()
    result = func(*args)
    end = time.time()
    return result, end - start

# Ajustar a função de Dijkstra para retornar a soma total dos caminhos mínimos
def dijkstra_total_cost(graph, start, n):
    dist = [float('inf')] * n
    dist[start] = 0
    pq = [(0, start)]  # Min-heap
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, weight in graph[u]:
            if dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                heapq.heappush(pq, (dist[v], v))
    return sum(dist) if all(x < float('inf') for x in dist) else float('inf')

# Algoritmo de Kruskal
def kruskal(edges, n):
    parent = list(range(n))
    rank = [0] * n

    def find(u):
        if parent[u] != u:
            parent[u] = find(parent[u])
        return parent[u]

    def union(u, v):
        root_u, root_v = find(u), find(v)
        if root_u != root_v:
            if rank[root_u] > rank[root_v]:
                parent[root_v] = root_u
            elif rank[root_u] < rank[root_v]:
                parent[root_u] = root_v
            else:
                parent[root_v] = root_u
                rank[root_u] += 1

    mst_cost = 0
    mst_edges = []
    edges.sort(key=lambda x: x[2])  # Sort edges by weight
    for u, v, weight in edges:
        if find(u) != find(v):
            union(u, v)
            mst_cost += weight
            mst_edges.append((u, v, weight))
    return mst_cost, mst_edges

# Algoritmo de Prim
def prim(graph, n):
    visited = [False] * n
    min_heap = [(0, 0)]  # (peso, vértice)
    mst_cost = 0
    while min_heap:
        weight, u = heapq.heappop(min_heap)
        if visited[u]:
            continue
        mst_cost += weight
        visited[u] = True
        for v, w in graph[u]:
            if not visited[v]:
                heapq.heappush(min_heap, (w, v))
    return mst_cost

def process_instance(n, m, edges):
    # Construir grafo
    graph = defaultdict(list)
    for u, v, weight in edges:
        graph[u].append((v, weight))
        graph[v].append((u, weight))  # Grafo não direcionado

    # Algoritmo de Dijkstra
    dijkstra_cost, dijkstra_time = measure_time(dijkstra_total_cost, graph, 0, n)

    # Algoritmo de Kruskal
    kruskal_result, kruskal_time = measure_time(kruskal, edges, n)

    # Algoritmo de Prim
    prim_cost, prim_time = measure_time(prim, graph, n)

    # Tabela de resultados
    return {
        "n": n,
        "m": m,
        "Custo_CM": dijkstra_cost,
        "Tempo_CM": dijkstra_time,
        "Custo_AGM_Kruskal": kruskal_result[0],
        "Tempo_AGM_Kruskal": kruskal_time,
        "Custo_AGM_Prim": prim_cost,
        "Tempo_AGM_Prim": prim_time
    }

n, m = 5, 7
edges = [
    (0, 1, 10),
    (0, 2, 20),
    (1, 2, 30),
    (1, 3, 50),
    (2, 3, 20),
    (3, 4, 10),
    (2, 4, 30)
]
result = process_instance(n, m, edges)
print(result)

# Processar uma lista de instâncias e gerar uma tabela de resultados
def generate_table(instances):
    results = []
    for instance in instances:
        n, m, edges = instance
        result = process_instance(n, m, edges)
        results.append(result)
    # Converter resultados para DataFrame
    df = pd.DataFrame(results)
    return df

# Definir instâncias de exemplo
instances = [
    (5, 7, [
        (0, 1, 10),
        (0, 2, 20),
        (1, 2, 30),
        (1, 3, 50),
        (2, 3, 20),
        (3, 4, 10),
        (2, 4, 30)
    ]),
    # Adicione outras instâncias aqui
    (4, 5, [
        (0, 1, 15),
        (1, 2, 25),
        (2, 3, 35),
        (0, 3, 45),
        (1, 3, 10)
    ])
]

# Gerar tabela
df = generate_table(instances)

# Exibir a tabela no console
print(tabulate(df, headers='keys', tablefmt='grid'))
