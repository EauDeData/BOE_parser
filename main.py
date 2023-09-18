from src.workload.pooler import ProcessPooler

process_pooler = ProcessPooler('prima', gpu_list=[1, 2, 3], nthreads= 5)
process_pooler.process('/data2fast/users/amolina/BOE/carlos_ii/')

