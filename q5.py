import random
import statistics
numbers = [random.randint(100, 150) for _ in range(100)]
mean = statistics.mean(numbers)
median= statistics.median(numbers)
mode= statistics.mode(numbers)

print("mean", mean)
print("Median", median)
print("Mode", mode)
