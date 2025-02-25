import numpy, time, random, multiprocessing, functools, itertools, os, sys

os.environ['PYTHONHASHSEED'] = '0'

def get_shingles(line, k):
    shingles = set()
    for i in range(len(line) - k + 1):
        shingles.add(hash(line[i:i+k]))
    return shingles

LIMIT = 2000 # for testing

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

def hash_n(n):
    random.seed(n)
    mask = random.getrandbits(64)
    def inner(x): 
        return mask ^ hash(x)
    return inner

def next_sim_matrix(documents):
    i = 0
    while True:
        h = hash_n(i)
        def sad(): 
            return [max([h(x) for x in doc]) for doc in documents]
        lsh = sad()
        yield numpy.array(lsh)[:, numpy.newaxis] == numpy.array(lsh)[numpy.newaxis, :]
        i += 1

def main():
    q, documents = read_documents()

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

if __name__ == '__main__':
    import cProfile
    # if check avoids hackery when not profiling
    # Optional; hackery *seems* to work fine even when not profiling, it's just wasteful
    if sys.modules['__main__'].__file__ == cProfile.__file__:
        import code_1  # Imports you again (does *not* use cache or execute as __main__)
        globals().update(vars(code_1))  # Replaces current contents with newly imported stuff
        sys.modules['__main__'] = code_1  # Ensures pickle lookups on __main__ find matching version
    main()  # Or series of statements