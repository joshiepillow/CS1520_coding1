import numpy, time, random, multiprocessing, functools, itertools, os, sys

# necessary for the hash function to be fixed on different threads.
os.environ['PYTHONHASHSEED'] = '0'

# String, Int -> List[Int]
# computes the list of hash values of the k shingles on a given line of input
def get_shingles(line, k):
    shingles = set()
    for i in range(len(line) - k + 1):
        shingles.add(hash(line[i:i+k]))
    return list(shingles)

# for testing purposes on small sections of the input
LIMIT = 10**9 

# None -> Int, List[List[Int]]
# read in the data
def read_documents():
    t = time.time()
    with open("documents", "r") as file:
        _, k, q = [int(i) for i in file.readline().split(" ")]
        
        # distribute lines in input to different threads to compute shingles faster
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
    print()

    relevant_documents = list(documents)
    matrix_iter = next_sim_matrix(relevant_documents)
    matrix = numpy.fromfunction(lambda i, j: i < j, (n, n))
    
    while True:
        t = time.time()
        new = matrix & next(matrix_iter)

        all_i, all_j = numpy.nonzero(new)
        count = len(all_i)

        relevant = set(all_i).union(set(all_j))
        for index in range(n):
            if not index in relevant:
                relevant_documents[index] = [0]
        
        print(f"Pairs: {count} Relevant: {len(relevant)} Time: {time.time() - t}")

        if count <= q:
            print()
            all_i, all_j = numpy.nonzero(matrix)
            
            sims = analyze_true_similarity(documents, all_i, all_j)

            t = time.time()
            best = sorted(zip(sims, all_i, all_j), reverse=True)[:q]
            new_sims, _, _ = zip(*best)
            print(f"Min: {min(new_sims)} Avg: {sum(new_sims)/len(new_sims)} Time: {time.time() - t}")

            with open("lsh_ans", "w") as file:
                for _, i, j in best:
                    file.write(f"{i} {j}\n")
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