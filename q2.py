rowA = int(input("Enter rows"))
colA = int(input("Enter columns"))

rowB = int(input("Enter rows"))
colB = int(input("Enter columns"))

if colA != rowB:
    print("Matrix multiplication not possible")
else:
    A = []
    B = []
    print("matrix A")
    for i in range(rowA):
        row = []
        for j in range(colA):
            element = int(input(f"Enter element [{i}][{j}]: "))
            row.append(element)
        A.append(row)
    print("matrix B")
    for i in range(rowB):
        row = []
        for j in range(colB):
            element = int(input(f"Enter element [{i}][{j}]: "))
            row.append(element)
        B.append(row)

    result = [[0 for _ in range(colB)] for _ in range(rowA)]

    for i in range(rowA):
        for j in range(colB):
            for k in range(colA):
                result[i][j] += A[i][k] * B[k][j]

    print("Product Matrix:")
    for row in result:
        print(row)
