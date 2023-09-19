import multiprocessing as mp
from multiprocessing import Process, Manager
import torch
import fitz
import os
import json
import warnings
import time
import random
from tqdm import tqdm
warnings.filterwarnings("ignore")

from src.vision.visutils import load_model
from src.io.ioutils import save_numpy, read_img
from src.text.textutils import extract_text_with_position

class DeviceManager:
    result = None
    def __init__(self) -> None:
        pass

    def lock(self):
        self.result = None
        self.free = False
    
    def unlock(self):
        self.free = True
        return self.result

class CudaDeviceManager(DeviceManager):
    def __init__(self, model_name, gpu_idx) -> None:

        self.gpu = f'cuda:{gpu_idx}'

        self.model_name = model_name
        self.model = None 
        self.free = True
        self.result = None

    def launch_and_capture(self, model_name, file, fname, images, output):

        print(f"{self.gpu} GPU Manager Says: I'm handling a process.")
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu.split(':')[-1]
        model = load_model(model_name, device = 'cuda')

        json_gt = {
                "file": file, 
                "path": fname,
                "pages": {}
                }
        
        save_numpy(images, fname)
        for idx_image in images:
            json_gt["pages"][idx_image] = []
            image = images[idx_image]
            with torch.no_grad():
                detection = model.detect(image)

            for item in detection:
                
                json_gt["pages"][idx_image].append(
                            {"type": item.type, "bbox": [int(x) for x in item.coordinates], 'conf': item.score}
                        )
        print(f'GPU device {self.gpu} says: Finished with {len(json_gt["pages"]["0"])} boxes found.')
        output.append(json_gt)
        model = None

        
    def launch_process(self, file, fname, images, manager):

        self.result = manager.list()
        process = Process(target = self.launch_and_capture, args=(self.model_name, file, fname, images, self.result))
        process.start()


class CPUDeviceManager(DeviceManager):
    def __init__(self, cpu_idx) -> None:
        self.thread_num = cpu_idx
        self.free = True
        self.process = None
    
    def parallelize_cpu_computing(self, gpu_manager, images, fname):
        while not len(gpu_manager.result): continue
        print(f"CPU Thread {self.thread_num} says: I received output from GPU {gpu_manager.gpu} with state free = {self.free}.")

        json_gt = gpu_manager.unlock()[0]
        print(f"CPU Thread {self.thread_num} says: I unlocked GPU {gpu_manager.gpu}, current state is free = {self.free}. Received {len(json_gt['pages']['0'])} boxes.")
        margin = 15

        doc =  fitz.open(fname)
        for idx_image, page in zip(json_gt["pages"], doc):

            max_x, max_y = page.mediabox_size
            
            for box_number, box in enumerate(json_gt["pages"][idx_image]):

                x,y,w,h = box['bbox']
                x, y = x-margin, y - margin  
                w,h = w+margin, h+margin

                json_gt["pages"][idx_image][box_number]['ocr'] = extract_text_with_position(fname, idx_image, images[idx_image], max_x, max_y, x, y, w, h)

        outname = fname.replace('images', 'jsons').replace('.pdf', '.json')
        os.makedirs(os.path.dirname(outname), exist_ok=True)
        json.dump(json_gt, open(outname, 'w'))
        self.unlock()
        print(f"CPU Thread {self.thread_num} says: I finished my process. I'm chilling now setting my state to free = {self.free}")
        
        

    def launch_process(self, images, fname, gpu_manager):

        self.working = True
        self.process = Process(target = self.parallelize_cpu_computing, args=(gpu_manager, images, fname))
        self.process.start()
        

    
def _control_results(processes):
    base_string = "\tDevice GPU {} has {} result len and state free = {}\n"
    while True:
        total_string = ''
        for process in processes:
            total_string = total_string + base_string.format(process.gpu, process.result, process.free)
        
        print(total_string)
        time.sleep(10)
        
class ProcessPooler:
    def __init__(self, model_name, gpu_list = [0, 1, 5, 3], nthreads = 16) -> None:
        
        self.manager = Manager()
        self.gpu_process =  self.manager.list([
            CudaDeviceManager(model_name, idx) for idx in gpu_list
        ])

        self.cpu_process = self.manager.list([
            CPUDeviceManager(idx) for idx in range(nthreads)
        ])
                
            
    def acquire_device(self, device_name = 'cpu', idx = 0):
        if device_name == 'cpu': devices = self.cpu_process
        else: devices = self.gpu_process

        already_told = False
        while True:
            device = devices[idx]
            if device.free: 

                print(f'Pooler says: Free {device_name} device found. Acquiring...')
                device.lock()
                return device


            idx += 1
            idx = idx % len(devices)
            already_told -= 1
            if idx == len(devices) -1 and already_told == 0:
                print(f'Pooler says: Free {device_name} not found. Waiting...')
                print(f'Devices:\n\t')
                for idx_debug, dev in enumerate(devices):
                    print(f"{device_name} device number {idx_debug} has state free = {dev.free}")
                already_told = 100

    def acquire_gpu(self):
        return self.acquire_device(device_name = 'cuda')
                
    def acquire_cpu(self):
        return self.acquire_device(device_name = 'cpu')
    
    def process(self, base_folder_with_pdfs):
        
        file_extensions = ['.pdf',]

        number_of_files = 0
        filenames_total = []

        for root, _, files in tqdm(os.walk(base_folder_with_pdfs)):
            for file in files:
                if os.path.splitext(file)[1].lower() in file_extensions:
                    filenames_total.append((root, file))
        random.shuffle(filenames_total)
        for root, file in filenames_total:
            fname = os.path.join(root, file)
            images = read_img(fname)

            gpu_device = self.acquire_gpu()
            gpu_device.launch_process(file, fname, images, self.manager)

            cpu_device = self.acquire_cpu()

            cpu_device.launch_process(images, fname, gpu_device)
            number_of_files += 1

            print(f'Pooler says: Launched job {number_of_files} with GPU {gpu_device.gpu} and thread_id {cpu_device.thread_num}...')
                                    

