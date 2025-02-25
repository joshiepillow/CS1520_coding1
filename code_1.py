import numpy, time, random

# read in data
t = time.time()
file = open("documents", "r")
n, k, q = [int(i) for i in file.readline().split(" ")]

shingles = {} # map from k-shingles to N
documents = [] # map from i to the set of shingles of document i

for i in range(n):
    if i % 50 == 0:
        print(f"Document {i}/{n}. {len(shingles)} distinct shingles", end="\r", flush=True)
    l = file.readline()
    documents.append(set())
    
    shingle = ""
    for c in l:
        if c.isalpha(): # ignore nonalphabetic characters
            shingle += c
        if len(shingle) > k: # maintain shingle length
            shingle = shingle[1:]
        if len(shingle) == k: 
            if not shingle in shingles:
                shingles[shingle] = len(shingles) # map new shingle to next available integer
            documents[-1].add(shingles[shingle])

print("Time taken for document reading: ", time.time() - t)

# convert to numpy array
t = time.time()
doc_array = numpy.zeros((n, len(shingles)))
for i in range(len(documents)):
    for n in documents[i]:
        doc_array[i][n] = 1

print("Time taken for array conversion: ", time.time() - t)

# random hashing
#random_dict = {}
#def hash_n(n):
#    r = 0
#    if n in random_dict:
#        r = random_dict[n]
#    else: 
#        random.seed(n)
#        r = random_dict[n] = random.getrandbits(32)