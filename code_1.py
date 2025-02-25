import numpy, time, random, multiprocessing, functools, itertools, os

os.environ['PYTHONHASHSEED'] = '0'

def get_shingles(line, k):
    shingles = set()
    for i in range(len(line) - k + 1):
        shingles.add(hash(line[i:i+k]))
    return shingles

LIMIT = 100 # for testing

def read_documents():
    # read in data
    t = time.time()
    with open("documents", "r") as file:
        _, k, q = [int(i) for i in file.readline().split(" ")]
        documents = []
        
        with multiprocessing.Pool(8) as pool:
            documents = pool.map(functools.partial(get_shingles, k=k), itertools.islice(file, LIMIT), 8)
            
        print("Time taken for document reading: ", time.time() - t)
        return q, documents

@functools.cache
def hash_n(n):
    random.seed(n)
    mask = random.getrandbits(64)
    return lambda x: mask ^ hash(x)

def get_sim_matrix(documents):
    def inner(n):
        h = hash_n(n)
        l = len(documents)
        lsh = [max([h(x) for x in doc]) for doc in documents]
        return numpy.fromfunction(lambda i, j: i != j and lsh[i] == lsh[j], (l, l), dtype=bool)
    return functools.lru_cache()(inner)

if __name__ == '__main__':
    q, documents = read_documents()
    nth_sim_matrix = get_sim_matrix(documents)
    

# convert to numpy array
# t = time.time()
# doc_array = numpy.zeros((n, len(shingles)))
# for i in range(len(documents)):
#     for n in documents[i]:
#         doc_array[i][n] = 1

# print("Time taken for array conversion: ", time.time() - t)

# random hashing
#random_dict = {}
#def hash_n(n):
#    r = 0
#    if n in random_dict:
#        r = random_dict[n]
#    else: 
#        random.seed(n)
#        r = random_dict[n] = random.getrandbits(32)