import numpy as np
from itertools import combinations
import time

n = 0
points = []
points_map = np.zeros((21,21))

def get_distance(city_1, city_2):
    X = abs(points[city_1][0] - points[city_2][0])
    Y = abs(points[city_1][1] - points[city_2][1])
    return round(np.sqrt(X ** 2 + Y ** 2))

def makeMap():
    for i in range(n):
        for j in range(n):
            points_map[i][j] = get_distance(i , j)

from itertools import combinations

def held_karp():
    dp = {}
    for k in range(1, n):
        dp[(1 << k, k)] = (points_map[0][k], 0)
    for subset_size in range(2, n):
        for subset in combinations(range(1, n), subset_size):
            bits = 0
            for bit in subset:
                bits |= 1 << bit
            for k in subset:
                prev = bits & ~(1 << k)
                res = []
                for m in subset:
                    if m == k:
                        continue
                    res.append((dp[(prev, m)][0] + points_map[m][k], m))
                dp[(bits, k)] = min(res)
    
    bits = (2 ** n - 1) - 1
    res = []
    for k in range(1, n):
        res.append((dp[(bits, k)][0] + points_map[k][0], k))
    opt, parent = min(res)

    path = [1]
    last = parent
    cur_bits = bits
    for _ in range(n - 1):
        print(last)
        path.append(last + 1)
        next_bits = cur_bits & ~(1 << last)
        last = dp[(cur_bits, last)][1]
        cur_bits = next_bits
    path.append(1)
    return opt, path


with open('special2.txt', 'r', encoding='utf-8') as f:
    for line in f:
        numbers = list(map(float, line.strip().split()))
        points.append([numbers[1],numbers[2]])
        n = int(numbers[0])

makeMap()

start = time.perf_counter()
total_distance, path = held_karp()
end = time.perf_counter()
print(path)
print(total_distance)
print(end - start)




