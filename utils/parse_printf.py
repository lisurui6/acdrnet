printf_string = """
face 0, axis 0, d0 32, d1 32, d2 32, grad_alpha -2.000000
"""
# print(printf_string)
lines = printf_string.split("\n")
xs = []
ys = []
zs = []

for line in lines:
    if line:
        print(line)
        parts = line.split(",")
        axis = int(parts[1].split()[1])
        d0 = int(parts[2].split()[1])
        d1 = int(parts[3].split()[1])
        d2 = int(parts[4].split()[1])
        ds = [d0, d1, d2]
        x = ds[(3 - axis) % 3]
        y = ds[(4 - axis) % 3]
        z = ds[(5 - axis) % 3]
        xs.append(x)
        ys.append(y)
        zs.append(z)
print("xx")
print(xs)
print("yy")
print(ys)
print("zz")
print(zs)
