from jetson_utils import cudaAllocMapped, cudaToNumpy

import numpy as np


class segmentationBuffers:
    def __init__(self, net, args):
        self.net = net
        self.mask = None
        self.overlay = None
        self.composite = None
        self.class_mask = None
        
        self.use_stats = True
        self.use_mask = "mask" in args.visualize
        self.use_overlay = "overlay" in args.visualize
        self.use_composite = self.use_mask and self.use_overlay
        
        if not self.use_overlay and not self.use_mask:
            raise Exception("invalid visualize flags - valid values are 'overlay' 'mask' 'overlay,mask'")
             
        self.grid_width, self.grid_height = net.GetGridSize()	
        self.num_classes = net.GetNumClasses()

    @property
    def output(self):
        if self.use_overlay and self.use_mask:
            return self.composite
        elif self.use_overlay:
            return self.overlay
        elif self.use_mask:
            return self.mask
            
    def Alloc(self, shape, format):
        if self.overlay is not None and self.overlay.height == shape[0] and self.overlay.width == shape[1]:
            return

        if self.use_overlay:
            self.overlay = cudaAllocMapped(width=shape[1], height=shape[0], format=format)

        if self.use_mask:
            mask_downsample = 2 if self.use_overlay else 1
            self.mask = cudaAllocMapped(width=shape[1]/mask_downsample, height=shape[0]/mask_downsample, format=format) 

        if self.use_composite:
            self.composite = cudaAllocMapped(width=self.overlay.width+self.mask.width, height=self.overlay.height, format=format) 

        if self.use_stats:
            self.class_mask = cudaAllocMapped(width=self.grid_width, height=self.grid_height, format="gray8")
            self.class_mask_np = cudaToNumpy(self.class_mask)
            
    def ComputeStats(self):
        if not self.use_stats:
            return
            
        # get the class mask (each pixel contains the classID for that grid cell)
        self.net.Mask(self.class_mask, self.grid_width, self.grid_height)
        mask_data = cudaToNumpy(self.class_mask)
        # compute the number of times each class occurs in the mask
        class_histogram, _ = np.histogram(self.class_mask_np, bins=self.num_classes, range=(0, self.num_classes-1))

        print('grid size:   {:d}x{:d}'.format(self.grid_width, self.grid_height))
        print('num classes: {:d}'.format(self.num_classes))

        print('-----------------------------------------')
        print(' ID  class name        count     %')
        print('-----------------------------------------')

        
        Unsafe = "Cars"

        class_indices = {
            "background": 0,
            "aeroplane": 1,
            "bicycle": 2,
            "bird": 3,
            "boat": 4,
            "bottle": 5,
            "bus": 6,
            "car": 7,
            "cat": 8,
            "chair": 9,
            "cow": 10,
            "diningtable": 11,
            "dog": 12,
            "horse": 13,
            "motorbike": 14,
            "person": 15,
            "pottedplant": 16,
            "sheep": 17,
            "sofa": 18,
            "train": 19,
            "tvmonitor": 20,
            }

        classes_to_detect = ["motorbike","bus","car","bicycle"]  # Modify as per the class you want to detect

        # Prepare a dictionary to store counts for each class
        class_counts = {class_name: 0 for class_name in class_indices}

        # Count occurrences of each class in the mask
        for class_name, class_index in class_indices.items():
            class_counts[class_name] = np.sum(mask_data == class_index)

        # Print statistics for each class
        for n, class_name in enumerate(class_counts):
            count = class_counts[class_name]
            percentage = float(count) / float(self.grid_width * self.grid_height)
            print(f' {n:2d}  {class_name:<18s} {count:3d}   {percentage:f}')

        # Check if person is detected
        detected_classes = [class_name for class_name in classes_to_detect if class_counts.get(class_name, 0) > 0]
        if detected_classes:
            print(f"Detected {Unsafe}, do not cross!")
        else:
            print("It's safe to cross.")