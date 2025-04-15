import numpy as np
import math
import time
import sys

# 재귀함수 깊이 늘리기
sys.setrecursionlimit(1000000)

def selectionsort(A):
    n = len(A)
    for i in range(n):
        min = A[i]
        idx = i
        for j in range(i+1,n):
            if min > A[j]:
                min = A[j]
                idx = j
        if idx != i:
            A[idx] = A[i]
            A[i] = min

    return A

def insertionsort(A):
    n = len(A)
    for j in range(1, n):
        key = A[j]
        i = j - 1
        while i >= 0 and A[i] > key:
            A[i+1] = A[i]
            i = i - 1
        A[i+1] = key

    return A
def merge(A, p , q, r):
    n1 = q - p + 1
    n2 = r - q
    L = np.zeros(n1 + 1)
    R = np.zeros(n2 + 1)

    for i in range(n1):
        L[i] = A[p + i]
    for i in range(n2):
        R[i] = A[q + i + 1]

    L[n1] = np.inf
    R[n2] = np.inf

    i = 0
    j = 0
    k = p
    while k <= r:
        if L[i] <= R[j]:
            A[k] = L[i]
            i = i + 1
        else:
            A[k] = R[j]
            j = j + 1
        k = k + 1
    
def mergesort_func(A, p, r):
    if p < r:
        q = int((p + r) / 2)
        mergesort_func(A, p, q)
        mergesort_func(A, q + 1, r)
        merge(A, p, q, r)
    return A
def mergesort(A):
    n = len(A)
    return mergesort_func(A, 0, n - 1)
def bubblesort(A):
    n = len(A)
    for i in range(n):
        for j in range(n-i-1):
            if A[j] > A[j+1]:
                temp = A[j]
                A[j] = A[j+1]
                A[j+1] = temp
    return A
def partition(A, p, r):
    x = A[r]
    i = p - 1
    j = p
    while j <= r - 1:
        if A[j] <= x:
            i = i + 1
            temp = A[i]
            A[i] = A[j]
            A[j] = temp
        j = j + 1
    temp = A[i+1]
    A[i+1] = A[r]
    A[r] = temp

    return i + 1
def quicksort_fuc(A, p ,r):
    if p < r:
        q = partition(A, p, r)
        quicksort_fuc(A, p, q - 1)
        quicksort_fuc(A, q + 1, r)

    return A
def quicksort(A):
    n = len(A)
    return quicksort_fuc(A, 0, n - 1)
def maxheapify(A, i, heapsize):
    L = i * 2 + 1
    R = i * 2 + 2
    largest = i
    if L < heapsize and A[L] > A[i]:
        largest = L
    if R < heapsize and A[R] > A[largest]:
        largest = R
    if largest != i:
        temp = A[i]
        A[i] = A[largest]
        A[largest] = temp
        A = maxheapify(A, largest, heapsize)
    return A

def buildmaxheap(A):
    n = len(A) - 1
    i = int(n / 2)
    heapsize = n
    while i >= 0:
        A = maxheapify(A, i, heapsize)
        i = i - 1
    return A

def heapsort(A):
    A = buildmaxheap(A)
    n = len(A) - 1
    heapsize = n
    while n >= 1:
        temp = A[0]
        A[0] = A[n]
        A[n] = temp
        heapsize = heapsize - 1
        A = maxheapify(A, 0, heapsize)
        n = n - 1
    return A

def rebalancing(A, v):
    N = 2 * v
    new_A = [None] * N
    cur = 1
    for i in range(len(A)):
        if A[i] is not None:
            new_A[cur] = A[i]
            cur += 2
    return new_A

def find_pos(A, x, count):
    if count == 0:
        return 0
    left, right = 0, len(A) - 1
    while left <= right:
        mid = (left + right) // 2
        if A[mid] is None:
            l, r = mid - 1, mid + 1
            while True:
                if l >= left and A[l] is not None:
                    mid = l
                    break
                if r <= right and A[r] is not None:
                    mid = r
                    break
                l -= 1
                r += 1
                if l < left and r > right:
                    return mid
        if A[mid] is None or A[mid] < x:
            left = mid + 1
        else:
            right = mid - 1
    return left

def librarysort(arr):
    n = len(arr)
    A = [None] * (2 * n)
    count = 0

    for i in range(n):
        x = arr[i]
        pos = find_pos(A, x, count)

        if pos >= len(A) or A[pos] is not None:
            A = rebalancing(A, count)
            pos = find_pos(A, x, count)

        A[pos] = x
        count += 1

    return [x for x in A if x is not None]
def insertion_sort_for_tim(A, p, r):
    for i in range(p + 1, r + 1):
        key = A[i]
        j = i - 1
        while j >= p and A[j] > key:
            A[j + 1] = A[j]
            j = j - 1
        A[j + 1] = key

    return A
def timsort(A):
    n = len(A)
    min_run = 32

    for start in range(0, n, min_run):
        end = min(start + min_run - 1, n - 1)
        insertion_sort_for_tim(A, start, end)

    size = min_run
    while size < n:
        for left in range(0, n, 2 * size):
            mid = min(n - 1, left + size - 1)
            right = min(n - 1, left + 2 * size - 1)

            if mid < right:
                A = merge(A, left, mid, right)

        size *= 2
    return A
def cocktail_shakersort(A):
    n = len(A)
    p = -1
    r = n
    while p < r:
        for i in range(p + 1, r - 1):
            if A[i] > A[i + 1]:
                temp = A[i]
                A[i] = A[i + 1]
                A[i + 1] = temp
        p = p + 1
        for i in range(r - 1, p + 1, -1):
            if A[i] < A[i - 1]:
                temp = A[i]
                A[i] = A[i - 1]
                A[i - 1] = temp
        r = r - 1

    return A
def combsort(A):
    n = len(A)
    gap = len(A)
    shrink_factor = 1.3
    gap = int(gap / shrink_factor)
    while gap > 0:
        i = 0
        while i + gap < n:
            if A[i] > A[i + gap]:
                temp = A[i]
                A[i] = A[i + gap]
                A[i + gap] = temp
            i = i + 1
        gap = gap - 1
    return A

def tournament(A, p, r):
    if p == r:
        return r,A[r]
    q = int((p + r) / 2)
    left_tournament_winner, left_tournament_winner_v = tournament(A, p, q)
    right_tournament_winner, right_tournament_winner_v = tournament(A, q + 1, r)
    if left_tournament_winner_v > right_tournament_winner_v:
        return right_tournament_winner, right_tournament_winner_v
    return left_tournament_winner, left_tournament_winner_v

def tournamentsort(A):
    n = len(A)
    new_A = np.zeros(n)
    for i in range(n):
        idx, min = tournament(A, 0, n - 1)
        new_A[i] = min
        A[idx] = np.inf  
    return new_A

def introsort_util(A, start, end, maxdepth):
    size = end - start + 1
    if size <= 16:
        A = insertion_sort_for_tim(A, start, end)
    elif maxdepth == 0:
        A[start : end + 1] = heapsort(A[start : end + 1].copy())
    else:
        p = partition(A, start, end)
        A = introsort_util(A, start, p - 1, maxdepth - 1)
        A = introsort_util(A, p + 1, end, maxdepth - 1)
    return A

def introsort(A):
    maxdepth = int(math.log2(len(A))) * 2
    return introsort_util(A, 0, len(A) - 1, maxdepth)

def make_sorted(n):
    array = np.random.uniform(low = 0.00001, high = 100000, size = n)
    return sorted(array)

def make_opp_sorted(n):
    array = np.random.uniform(low = 0.00001, high = 100000, size = n)
    return sorted(array, reverse = True)

def make_random_array(n):
    array = np.random.uniform(low = 0.00001, high = 100000, size = n)
    return array

def make_partial_sorted(n):
    N = int(n / 3)
    pos = np.random.randint(3, n - N - 2)

    array = np.random.uniform(low = 0.00001, high = 100000, size = n)
    array[pos : pos + N + 1] = sorted(array[pos : pos + N + 1])

    return array
def check_time(N, sort, data):
    mean_time = 0
    for i in range(10):
        arr = data(N)
        start = time.perf_counter()
        sort(arr)
        end = time.perf_counter()
        mean_time = mean_time + end - start
    mean_time = mean_time / 10
    print(f"{N}개 10회 실행 시간 평균 : {mean_time:.6f}초")
def analysis_sort(sort):
    print("1. Random Data")
    check_time(1000, sort, make_random_array)
    check_time(2000, sort, make_random_array)
    check_time(5000, sort, make_random_array)
    check_time(10000, sort, make_random_array)
    #check_time(50000, sort, make_random_array)
    #check_time(100000, sort, make_random_array)

    print("2. Sorted Data")
    check_time(1000, sort, make_sorted)
    check_time(2000, sort, make_sorted)
    check_time(5000, sort, make_sorted)
    check_time(10000, sort, make_sorted)
    #check_time(50000, sort, make_sorted)
    #check_time(100000, sort, make_sorted)

    print("3. Reverse Sorted Data")
    check_time(1000, sort, make_opp_sorted)
    check_time(2000, sort, make_opp_sorted)
    check_time(5000, sort, make_opp_sorted)
    check_time(10000, sort, make_opp_sorted)
    #check_time(50000, sort, make_opp_sorted)
    #check_time(100000, sort, make_opp_sorted)

    print("4. Partially Sorted Data")
    check_time(1000, sort, make_partial_sorted)
    check_time(2000, sort, make_partial_sorted)
    check_time(5000, sort, make_partial_sorted)
    check_time(10000, sort, make_partial_sorted)
    #check_time(50000, sort, make_partial_sorted)
    #check_time(100000, sort, make_partial_sorted)

def checksort(A, B):
    return np.array_equal(A, B)

#print(array)
#print(mergesort(array,0,size-1))
#print(insertionsort(array, size))
#print(bubblesort(array, size))
#print(quicksort(array, 0, size-1))
#print(heapsort(array))
#print(librarysort(array))

#print(checksort(sorted(array), librarysort(array)))
#print(timsort(array))
#print(cocktail_shakersort(array))
#print(checksort(sorted(array), cocktail_shakersort(array)))
#print(checksort(sorted(array), timsort(array)))
#print(combsort(array))
#print(checksort(sorted(array),combsort(array)))
#print(tournamentsort(array))
#print(checksort(sorted(array), tournamentsort(array)))

"""
print("Selection Sort")
analysis_sort(selectionsort)

print("Insertion Sort")
analysis_sort(insertionsort)

print("Bubble Sort")
analysis_sort(bubblesort)

print("Heap Sort")
analysis_sort(heapsort)

print("Merge Sort")
analysis_sort(mergesort)

print("Quick Sort")
analysis_sort(quicksort)
"""

#analysis_sort(quicksort)
#check_time(1000000, quicksort, make_random_array)

#analysis_sort(timsort)
print("cocktail_shakersort")
analysis_sort(cocktail_shakersort)
print("combsort")
analysis_sort(combsort)
print("tournamentsort")
analysis_sort(tournamentsort)
print("introsort")
analysis_sort(introsort)