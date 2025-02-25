import numpy, time, random, multiprocessing, functools, itertools, os, sys

os.environ['PYTHONHASHSEED'] = '0'

def get_shingles(line, k):
    shingles = set()
    for i in range(len(line) - k + 1):
        shingles.add(hash(line[i:i+k]))
    return list(shingles)

LIMIT = 2000000000 # for testing

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
        lsh = [max([h(x) for x in doc]) for doc in documents]
        yield numpy.array(lsh)[:, numpy.newaxis] == numpy.array(lsh)[numpy.newaxis, :]
        i += 1

def compute_true_similarity(documents, i, j):
    i_set = set(documents[i])
    j_set = set(documents[j])
    return len(i_set.intersection(j_set))/len(i_set.union(j_set))

def analyze_true_similarity(documents, all_i, all_j):
    t = time.time()
    sims = [compute_true_similarity(documents, i, j) for i, j in zip(all_i, all_j)]
    print(f"Min: {min(sims)} Avg: {sum(sims)/len(sims)} Time: {time.time() - t}")
    return sims

def main():
    q, documents = read_documents()
    n = len(documents)
    print(f"Max: {max([len(doc) for doc in documents])} Avg: {sum([len(doc) for doc in documents])/n}")

    relevant_documents = list(documents)
    matrix_iter = next_sim_matrix(relevant_documents)
    matrix = numpy.fromfunction(lambda i, j: i < j, (n, n))
    
    while True:
        t = time.time()
        new = matrix & next(matrix_iter)
        i, j = numpy.nonzero(new)
        count = len(i)
        relevant = set(i).union(set(j))
        for index in range(n):
            if not index in relevant:
                relevant_documents[index] = [0]
        
        print(f"Pairs: {count} Relevant: {len(relevant)} Time: {time.time() - t}")

        if count <= q:
            remainder = matrix ^ new 
            i_rem, j_rem = numpy.nonzero(remainder)
            i = list(numpy.append(i, i_rem))
            j = list(numpy.append(j, j_rem))

            analyze_true_similarity(documents, i, j)

            with open("lsh_ans", "w") as file:
                for k in range(q):
                    file.write(f"{i[k]} {j[k]}\n")
            break
        else:
            matrix = new

    #num_ands = 1
    #or_matrices = [next(matrix_iter)]
    #print(f"Computed first matrix in {time.time() - t} time.")

    #for i in range(1):
    #    sim_matrix = numpy.logical_or.reduce(or_matrices)
    #    count = numpy.count_nonzero(sim_matrix) - len(documents)

    #    print(f"Target: {q} Count: {count} ORs: {len(or_matrices)} ANDs: {num_ands}")
    #    if count > q:
    #        or_matrices = [a & next(matrix_iter) for a in or_matrices]
    #        num_ands += 1
    #    elif count <= q:
    #        or_matrices += [numpy.logical_and.reduce([next(matrix_iter) for _ in range(num_ands)]) for _ in range(len(or_matrices))]

if __name__ == '__main__':
    import cProfile
    # if check avoids hackery when not profiling
    # Optional; hackery *seems* to work fine even when not profiling, it's just wasteful
    if sys.modules['__main__'].__file__ == cProfile.__file__:
        import code_1  # Imports you again (does *not* use cache or execute as __main__)
        globals().update(vars(code_1))  # Replaces current contents with newly imported stuff
        sys.modules['__main__'] = code_1  # Ensures pickle lookups on __main__ find matching version
    main()  # Or series of statements