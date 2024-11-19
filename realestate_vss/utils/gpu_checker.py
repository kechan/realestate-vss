import pynvml
from datetime import datetime
import torch

class GPUAvailabilityChecker:
    def __init__(self, gpu_index=0, utilization_threshold=5, high_priority_window=600):
        """
        Initialize the GPU availability checker.

        Args:
            gpu_index (int): Index of the GPU to monitor (only applicable for CUDA).
            utilization_threshold (int): Maximum GPU utilization (%) to consider it available.
            high_priority_window (int): Minimum seconds after the hour to allow secondary tasks.
        """
        self.gpu_index = gpu_index
        self.utilization_threshold = utilization_threshold
        self.high_priority_window = high_priority_window
        self.device_type = self._detect_device_type()

        if self.device_type == "CUDA":
            pynvml.nvmlInit()

    def _detect_device_type(self):
        """
        Detect whether the environment is using CUDA or MPS.

        Returns:
            str: "CUDA" if CUDA is available, "MPS" if using Apple Silicon, "None" otherwise.
        """
        if torch.cuda.is_available():
            return "CUDA"
        elif torch.backends.mps.is_available():
            return "MPS"
        else:
            return "None"

    def get_gpu_metrics(self):
        """
        Retrieve current GPU utilization and memory usage (only for CUDA).

        Returns:
            dict: A dictionary containing utilization (%) and memory used (MiB).
        """
        if self.device_type != "CUDA":
            return {"utilization": 0, "memory_used": 0, "total_memory": 0}

        handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            "utilization": utilization.gpu,
            "memory_used": memory_info.used / 1024**2,  # Convert to MiB
            "total_memory": memory_info.total / 1024**2,
        }

    def is_gpu_available(self):
        """
        Check if the GPU is available for secondary tasks.

        Returns:
            bool: True if the GPU is available or running on MPS, False otherwise.
        """
        # If not using CUDA (e.g., on MPS), always return True
        if self.device_type != "CUDA":
            return True

        # Check time since the hour
        now = datetime.now()
        seconds_past_hour = now.minute * 60 + now.second
        if seconds_past_hour < self.high_priority_window:
            return False  # Still within the high-priority task window

        # Check GPU utilization
        metrics = self.get_gpu_metrics()
        if metrics["utilization"] > self.utilization_threshold:
            return False  # GPU is in use

        return True

    def shutdown(self):
        """Clean up NVML resources (only for CUDA)."""
        if self.device_type == "CUDA":
            pynvml.nvmlShutdown()
