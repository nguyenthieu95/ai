file = open("test.txt", "w")

file.write("Hello world\n")
file.write("This is {0}\n".format(12))
b = None
file.write("End. {0}".format(b))

w = [12, 0.4, 9.1, -3.5]

file.write("w = {0}".format(w))

file.close()

