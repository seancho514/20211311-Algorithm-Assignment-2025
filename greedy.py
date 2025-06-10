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

def greedy(c):
    cnt = 0
    cur = c
    while True:
        cnt += 1
        path.append(cur)
        visited[cur] = 1
        if cnt == n:
            return
        min = 99999999999
        idx = 0
        for i in range(1, n + 1):
            if visited[i] == 0 and min > points_map[cur][i]:
                min = points_map[cur][i]
                idx = i
        cur = idx

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

path = []
makeMap()
start = time.perf_counter()
greedy(1)
end = time.perf_counter()
path.append(1)
print(path)
print(get_total())
print(end - start)