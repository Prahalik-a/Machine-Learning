inp = input("Enter the string: ")

vow = 0
con = 0

for ch in inp:
        if ch in 'aeiou':
            vow += 1
        else:
            con += 1

print(vow)
print(con)
