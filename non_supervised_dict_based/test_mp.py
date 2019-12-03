from multiprocessing import Pool

d = {1:2, 3:4, 5:6}

def job(x):
    return x[0] + x[1]

if __name__ == "__main__":
    pool = Pool(processes=4)
    res = pool.map(job, d.items())
    print(res)