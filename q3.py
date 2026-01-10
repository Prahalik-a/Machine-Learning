list1 = list(map(int, input("Enter elements l1").split()))
list2 = list(map(int, input("Enter elements l2 ").split()))

common = set(list1) & set(list2)
print("Common element are " , list(common))
