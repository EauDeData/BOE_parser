from src.workload.pooler import ProcessPooler, mp

if __name__ == '__main__':
    mp.set_start_method('spawn')

    process_pooler = ProcessPooler('prima', gpu_list=[0, 1, 2, 3, 4], nthreads= 32)
    process_pooler.process('/data2fast/users/amolina/BOE/')

