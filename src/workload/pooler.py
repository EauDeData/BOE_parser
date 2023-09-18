from multiprocessing import Process, Manager
import torch
import fitz
import os
import json

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
        self.model = load_model(model_name, device = gpu_idx)
        self.free = True

    def launch_and_capture(self, model, file, fname, images, output):

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
                item = model.detect(image)

            json_gt["pages"][idx_image].append(
                        {"type": item.type, "bbox": [int(x) for x in item.coordinates], 'conf': item.score}
                    )
            
            output.append(json_gt)



    def launch_process(self, file, fname, images, manager):

        self.result = manager.list()
        process = Process(target = self.launch_and_capture, args=(self.model, file, fname, images, self.result))
        process.start()


class CPUDeviceManager(DeviceManager):
    def __init__(self, cpu_idx) -> None:
        self.thread_num = cpu_idx
        self.free = True
    
    def parallelize_cpu_computing(self, gpu_manager, images, fname):
        while not len(gpu_manager.result): continue
        print(f"CPU Thread {self.thread_num} says: I received output from GPU {gpu_manager.gpu} with state free = {self.free}.")

        json_gt = gpu_manager.unlock()[0]
        print(f"CPU Thread {self.thread_num} says: I unlocked GPU {gpu_manager.gpu}, current state is free = {self.free}.")
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
        json.dump(json_gt, open(outname, 'w'))
        self.unlock()
        print(f"CPU Thread {self.thread_num} says: I finished my process. I'm chilling now setting my state to free = {self.free}")
        
        

    def launch_process(self, images, fname, gpu_manager):

        process = Process(target = self.parallelize_cpu_computing, args=(gpu_manager, images, fname))
        process.start()

    
    
class ProcessPooler:
    def __init__(self, model_name, gpu_list = [0, 1, 5, 3], nthreads = 16) -> None:
        
        self.manager = Manager()
        self.gpu_process = [
            CudaDeviceManager(model_name, idx) for idx in gpu_list
        ] 

        self.cpu_process = [
            CPUDeviceManager(idx) for idx in range(nthreads)
        ]

    def acquire_device(self, device = 'cpu', idx = 0):
        if device == 'cpu': devices = self.cpu_process
        else: devices = self.gpu_process

        already_told = False
        while True:
            device = devices[idx]
            if device.free: 

                print(f'Pooler says: Free {device} device found. Acquiring...')
                device.lock()
                return device


            idx += 1
            idx = idx % len(devices)
            if idx == len(devices) -1 and not already_told:
                print(f'Pooler says: Free {device} not found. Waiting...')
                already_told = True

    def acquire_gpu(self):
        return self.acquire_device(device = 'cuda')
                
    def acquire_cpu(self):
        return self.acquire_device(device = 'cpu')
    
    def process(self, base_folder_with_pdfs):
        
        file_extensions = ['.pdf',]

        number_of_files = 0
        for root, _, files in os.walk(base_folder_with_pdfs):
            for file in files:
                if os.path.splitext(file)[1].lower() in file_extensions:

                    fname = os.path.join(root, file)
                    images = read_img(fname)

                    gpu_device = self.acquire_gpu()
                    cpu_device = self.acquire_cpu()

                    gpu_device.launch_process(file, fname, images, self.manager)
                    cpu_device.launch_process(images, fname, gpu_device)

                    print(f'Pooler says: Launched job {number_of_files} with GPU {gpu_device.gpu} and thread_id {cpu_device.thread_num}...')
                    

