batch_size = 128
shape = 10000
num_batches = 3

for i in range(101):
    offset = ((i % num_batches) * batch_size) % (shape - batch_size)
##    offset = (i * batch_size) % (shape - batch_size)
    print(offset)
