import numpy, time, random, multiprocessing, functools, itertools, os, sys

# necessary for the hash function to return the same outputs on different threads.
os.environ['PYTHONHASHSEED'] = '0'

# String, Int -> List[Int]
# computes the list of hash values of the k shingles on a given line of input
def get_shingles(line, k):
    shingles = set()
    for i in range(len(line) - k + 1):
        shingles.add(hash(line[i:i+k])) 
        # there are a total of ~10^7 unique shingles. the expected number of collisions is (10^7)^2/2^65 << 1. 
        # on a 32-bit os the expected number of collisions is (10^7)^2/2^33 = ~10^4
    return list(shingles)

# only for testing purposes to only read a small section of the input
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

        # averages ~20s to read in all of the data
        # my original naive approach w/o multithreading and using a dictionary to convert from string to int took ~300s
        print(f"Read {len(documents)} in {time.time() - t} time.")
        print(f"Max: {max([len(doc) for doc in documents])} Avg: {sum([len(doc) for doc in documents])/len(documents)}")
        print()
        return q, documents

# Int, Int -> Int
# hashes and applies a mask
def hash_mask(x, mask):
    return mask ^ hash(x)

# Int -> Int -> Int
# returns a hash function dependent on n
# i tried other hash functions (universal hashing, xxhash, 32-bit hash) but they were all slower
def hash_n(n):
    random.seed(n)
    mask = random.getrandbits(64) # i am on a 64-bit os
    return functools.partial(hash_mask, mask=mask)

# List[Int], (Int -> Int) -> Int
# computes a locality-sensitive hash of the set of document shingles
def compute_lsh(document, h):
    return max([h(x) for x in document]) # very slow. using numpy arrays actually makes this even slower.

# List[List[Int]] -> Iterator[Array[Bool]]
# returns an iterator that gives similarity matrices m, where m[i,j] = True iff document i,j hash to the same value
def next_sim_matrix(documents):
    i = 0
    while True:
        h = hash_n(i)

        # multithreading barely helps
        lsh = []
        with multiprocessing.Pool(8) as pool:
            lsh = pool.map(functools.partial(compute_lsh, h=h), documents, 16)

        yield numpy.array(lsh)[:, numpy.newaxis] == numpy.array(lsh)[numpy.newaxis, :]
        i += 1

# List[List[Int]], Int, Int -> Number
# computes the exact jaccard similarity of two documents
def compute_true_similarity(documents, i, j):
    i_set = set(documents[i])
    j_set = set(documents[j])
    return len(i_set.intersection(j_set))/len(i_set.union(j_set))

# List[List[Int]], List[Int], List[Int] -> List[Number]
# computes the list of true similarities of the document pairs given by all_i and all_j
def analyze_true_similarity(documents, all_i, all_j):
    t = time.time()
    sims = [compute_true_similarity(documents, i, j) for i, j in zip(all_i, all_j)]
    print(f"Min: {min(sims)} Avg: {sum(sims)/len(sims)} Time: {time.time() - t}")
    return sims

# Int, List[List[Int]] -> None
# estimates q similar documents based on jaccard similarity
def find_most_similar(q, documents):
    n = len(documents)
    relevant_documents = list(documents) # make a copy of documents that will be changed 
    matrix_iter = next_sim_matrix(relevant_documents) # get the matrix iterator
    matrix = numpy.fromfunction(lambda i, j: i < j, (n, n)) # initialize matrix to only consider each pair of distinct documents once

    while True:
        t = time.time()
        new = matrix & next(matrix_iter)

        all_i, all_j = numpy.nonzero(new)
        count = len(all_i)

        # if a document is not in any of the potential pairs we are considering, set its shingles to 0 so that we don't waste computation on its lsh value in the future
        relevant = set(all_i).union(set(all_j))
        for index in range(n):
            if not index in relevant:
                relevant_documents[index] = [0]
        
        print(f"Pairs: {count} Relevant: {len(relevant)} Time: {time.time() - t}")

        # if we have narrowed down to less than 200000 pairs, we can manually compute the similarities in ~3m
        if count <= 200000:
            print()
            all_i, all_j = numpy.nonzero(new)
            
            sims = analyze_true_similarity(documents, all_i, all_j)

            t = time.time()
            best = sorted(zip(sims, all_i, all_j), reverse=True)[:q] # take the q pairs with highest similarity
            new_sims, _, _ = zip(*best)
            print(f"Min: {min(new_sims)} Avg: {sum(new_sims)/len(new_sims)} Time: {time.time() - t}")

            with open("lsh_ans", "w") as file:
                for _, i, j in best:
                    file.write(f"{i} {j}\n")
            break
        else:
            # otherwise refine the set of potential pairs again in the next iteration with the next hash
            matrix = new

def main():
    q, documents = read_documents()
    find_most_similar(q, documents)

    
    
    

if __name__ == '__main__':
    ### a hack i stole from stackoverflow to fix cProfile when multithreading. has no impact on the program when not profiling ###
    import cProfile
    # if check avoids hackery when not profiling
    # Optional; hackery *seems* to work fine even when not profiling, it's just wasteful
    if sys.modules['__main__'].__file__ == cProfile.__file__:
        import code_1  # Imports you again (does *not* use cache or execute as __main__)
        globals().update(vars(code_1))  # Replaces current contents with newly imported stuff
        sys.modules['__main__'] = code_1  # Ensures pickle lookups on __main__ find matching version
    main()  # Or series of statements