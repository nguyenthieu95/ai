import time
from urllib.request import urlopen
# from multiprocessing import Pool                        # processes
from multiprocessing.dummy import Pool as ThreadPool  # threads

urls = [
    'http://www.python.org',
    'http://www.python.org/about/',
    'http://www.onlamp.com/pub/a/python/2003/04/17/metaclasses.html',
    'http://www.python.org/doc/',
    'http://www.python.org/download/',
    'http://www.python.org/getit/',
    'http://www.python.org/community/',
    'https://wiki.python.org/moin/',
    'http://planet.python.org/',
    'https://wiki.python.org/moin/LocalUserGroups',
    'http://www.python.org/psf/',
    'http://docs.python.org/devguide/',
    'http://www.python.org/community/awards/'
    # etc..
]

#pool = ThreadPool(4)  # Sets the pool size to 4
# Number of workers, default = number of Cores in the machines

# Open the urls in their own threads and return the results
#results = pool.map(urlopen, urls)
#pool.close()       # close the pool and wait for the work to finish
#pool.join()


s1 = time.time()
results1 = []
for url in urls:
  result = urlopen(url)
  results1.append(result)
t1 = time.time() - s1

# # ------- VERSUS ------- #



# # ------- 4 Pool ------- #
s2 = time.time()
pool4 = ThreadPool(2)
results2 = pool4.map(urlopen, urls)
t2 = time.time() - s2



# # ------- 8 Pool ------- #
s3 = time.time()
pool8 = ThreadPool(4)
results3 = pool8.map(urlopen, urls)
t3 = time.time() - s3



# # ------- 13 Pool ------- #
s4 = time.time()
pool13 = ThreadPool(8)
results4 = pool13.map(urlopen, urls)
t4 = time.time() - s4


print("t1 = {}, t2 = {}, t3 = {}, t4 = {}".format(t1, t2, t3, t4))

















