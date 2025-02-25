import numpy, time, random, multiprocessing, functools, itertools, os, pickle

os.environ['PYTHONHASHSEED'] = '0'

def get_shingles(line, k):
    shingles = set()
    for i in range(len(line) - k + 1):
        shingles.add(hash(line[i:i+k]))
    return shingles

LIMIT = 100000000 # for testing

def read_documents():
    # read in data
    t = time.time()
    with open("documents", "r") as file:
        _, k, q = [int(i) for i in file.readline().split(" ")]
        documents = []
        
        with multiprocessing.Pool(8) as pool:
            documents = pool.map(functools.partial(get_shingles, k=k), itertools.islice(file, LIMIT), 8)
            
        print(f"Read {len(documents)} in {time.time() - t} time.")
        return q, documents

@functools.cache
def hash_n(n):
    random.seed(n)
    mask = random.getrandbits(64)
    return lambda x: mask ^ hash(x)

def next_sim_matrix(documents):
    i = 0
    while True:
        h = hash_n(i)
        lsh = [max([h(x) for x in doc]) for doc in documents]
        yield numpy.array(lsh)[:, numpy.newaxis] == numpy.array(lsh)[numpy.newaxis, :]
        i += 1

if __name__ == '__main__':
    q, documents = 0, []
    try:
        t = time.time()
        with open("cache.pkl", "rb") as file:
            q, documents = pickle.load(file)
        print(f"Loaded {len(documents)} documents from pickle file in {time.time() - t} time.")
    except FileNotFoundError:
        q, documents = read_documents()
        t = time.time()
        with open("cache.pkl", "wb") as file:
            pickle.dump((q, documents), file)
        print(f"Wrote {len(documents)} documents to pickle file in {time.time() - t} time.")

    t = time.time()
    matrix_iter = next_sim_matrix(documents)
    num_ands = 1
    or_matrices = [next(matrix_iter)]
    print(f"Computed first matrix in {time.time() - t} time.")

    for i in range(1):
        sim_matrix = numpy.logical_or.reduce(or_matrices)
        count = numpy.count_nonzero(sim_matrix) - len(documents)

        print(f"Target: {q} Count: {count} ORs: {len(or_matrices)} ANDs: {num_ands}")
        if count > q:
            or_matrices = [a & next(matrix_iter) for a in or_matrices]
            num_ands += 1
        elif count <= q:
            or_matrices += [numpy.logical_and.reduce([next(matrix_iter) for _ in range(num_ands)]) for _ in range(len(or_matrices))]