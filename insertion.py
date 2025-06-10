import numpy as np
import time
n = 0
points = ["blank"]
points_map = np.zeros((10001,10001))
visited = np.zeros(10001)

def get_distance(city_1, city_2):
    X = abs(points[city_1][0] - points[city_2][0])
    Y = abs(points[city_1][1] - points[city_2][1])
    return round(np.sqrt(X ** 2 + Y ** 2))

def makeMap():
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            points_map[i][j] = get_distance(i , j)

def insertion():
    for i in range(3, n + 1):
        min = 999999999
        idx = 0
        path_len = len(path) - 1
        for j in range(path_len):
            v = points_map[path[j]][i] + points_map[i][path[j + 1]]
            if min > v:
                min = v
                idx = j + 1
        path.insert(idx, i)

def get_total():
    total = 0
    for i in range(n):
        total += points_map[path[i]][path[i+1]]

    return total

with open('input3.txt', 'r', encoding='utf-8') as f:
    for line in f:
        numbers = list(map(float, line.strip().split()))
        points.append([numbers[1],numbers[2]])
        n = int(numbers[0])

makeMap()
path = [1, 2, 1]
start = time.perf_counter()
insertion()
end = time.perf_counter()

print(path)
print(get_total())
print(end - start)


