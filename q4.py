rows = int(input("Enter number of rows"))
cols = int(input("Enter number of columns"))
matrix = []
for i in range(rows):
    row = []
    for j in range(cols):
        element = int(input(f"Enter element at position [{i}][{j}]: "))
        row.append(element)
    matrix.append(row)

print("Matrix Transpose")
for i in range(rows):
    for j in range(cols):
        print(matrix[j][i], end=" ")
    print()
