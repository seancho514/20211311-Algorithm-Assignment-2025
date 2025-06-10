import numpy as np
import time
n = 0
points = ["blank"]
points_map = np.zeros((10001,10001))
mst_map = np.zeros((10001,10001))
visited = np.zeros(10001)

class MinHeap:
    def __init__(self):
        self.heap = []

    def push(self, x):
        self.heap.append(x)
        idx = len(self.heap) - 1

        while idx > 0:
            parent = (idx - 1) // 2
            if self.heap[parent][0] > self.heap[idx][0]:
                self.heap[parent], self.heap[idx] = self.heap[idx], self.heap[parent]
                idx = parent
            else:
                break
    def pop(self):
        if not self.heap:
            return None
        if len(self.heap) == 1:
            return self.heap.pop()
        root = self.heap[0]
        self.heap[0] = self.heap.pop()
        idx = 0
        N = len(self.heap)
        while True:
            left = 2 * idx + 1
            right = 2 * idx + 2
            smallest = idx
            if left < N and self.heap[left][0] < self.heap[smallest][0]:
                smallest = left
            if right < N and self.heap[right][0] < self.heap[smallest][0]:
                smallest = right
            if smallest == idx:
                break
            self.heap[smallest], self.heap[idx] = self.heap[idx], self.heap[smallest]
            idx = smallest
        return root
    
    def isempty(self):
        if not self.heap:
            return True
        return False

def get_distance(city_1, city_2):
    X = abs(points[city_1][0] - points[city_2][0])
    Y = abs(points[city_1][1] - points[city_2][1])
    return round(np.sqrt(X ** 2 + Y ** 2))

def makeMap():
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            points_map[i][j] = get_distance(i , j)

def prim():
    mst = []
    heap = MinHeap()

    for i in range(2, n + 1):
        heap.push([points_map[1][i], 1 , i])
    visited[1] = 1

    while heap.isempty() == False:
        line = heap.pop()
        if visited[line[2]] == 0:
            visited[line[2]] = 1
            mst.append(line)
            for i in range(1, n + 1):
                if visited[i] == 0:
                    heap.push([points_map[line[2]][i], line[2], i])

    return mst

def make_mst_map(mst):
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            mst_map[i][j] = -1

    for line in mst:
        mst_map[line[2]][line[1]] = line[0]
        mst_map[line[1]][line[2]] = line[0]

def find_path():
    cur = 1
    cnt = 0
    while True:
        path.append(cur)
        check[cur] = 1
        cnt = cnt + 1
        if cnt == n:
            return
        
        for i in range(1, n + 1):
            if mst_map[cur][i] != -1 and check[i] == 0:
                cur = i

        if check[cur] == 1:
            min = 9999999999999
            idx = cur
            for i in range(1, n + 1):
                if cur != i and check[i] == 0:
                    if points_map[cur][i] < min:
                        idx = i
                        min = points_map[cur][i]
            cur = idx

def get_total():
    total = 0
    for i in range(n):
        total += points_map[path[i]][path[i+1]]

    return total

with open('input2.txt', 'r', encoding='utf-8') as f:
    for line in f:
        numbers = list(map(float, line.strip().split()))
        points.append([numbers[1],numbers[2]])
        n = int(numbers[0])

start = time.perf_counter()
makeMap()
make_mst_map(prim())
path = []
check = np.zeros(10001)
find_path()
path.append(1)
print(path)
print(get_total())
end = time.perf_counter()
print(end - start)