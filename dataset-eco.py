"""
LED/Switch Classification Dataset
Optimized for microcontroller deployment with 8-image dataset
4 LED states + 4 Switch positions = 2 classes
"""

import tensorflow as tf
import numpy as np
import os
import json
from typing import Tuple, List, Dict, Optional
import base64
from pathlib import Path

class LEDSwitchDataset:
    """
    Specialized dataset handler for LED/Switch classification
    Optimized for 160x120 resolution, ~1.5-2KB per image
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (160, 120),
        batch_size: int = 1,  # Microcontroller optimized
        validation_split: float = 0.25,  # 2 images per class for validation
        cache_dir: Optional[str] = None,
        seed: int = 42
    ):
        self.image_size = image_size
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.cache_dir = cache_dir
        self.seed = seed
        
        # LED states mapping
        self.led_classes = {
            0: "led_low",
            1: "led_2medium", 
            2: "led3_high",
            3: "led4_none"
        }
        
        # Switch states mapping
        self.switch_classes = {
            0: "switch_off",
            1: "switch_position_1",
            2: "switch_position_2", 
            3: "switch_position_3"
        }
        
        self.class_names = ["led_states", "switch_states"]
        
    def create_sample_dataset(self, output_dir: str = "sample_data") -> str:
        """Generate sample 8-image dataset for testing"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create directories
        (output_path / "led_states").mkdir(exist_ok=True)
        (output_path / "switch_states").mkdir(exist_ok=True)
        
        # Generate synthetic images
        dataset_info = {
            "total_images": 8,
            "classes": {
                "led_states": {"count": 4, "examples": list(self.led_classes.values())},
                "switch_states": {"count": 4, "examples": list(self.switch_classes.values())}
            },
            "image_size": self.image_size,
            "format": "jpg",
            "optimized_for": "microcontroller"
        }
        
        # Create sample images (synthetic data for demonstration)
        for class_name in self.class_names:
            class_dir = output_path / class_name
            for i in range(4):
                # Create synthetic image data
                if class_name == "led_states":
                    # LED patterns - simple colored rectangles
                    img_data = self._create_led_pattern(i)
                else:
                    # Switch patterns - simple geometric shapes
                    img_data = self._create_switch_pattern(i)
                
                # Save as JPEG
                img_path = class_dir / f"{class_name}_{i}.jpg"
                tf.io.write_file(str(img_path), tf.image.encode_jpeg(img_data, quality=85))
        
        # Save dataset info
        with open(output_path / "dataset_info.json", "w") as f:
            json.dump(dataset_info, f, indent=2)
            
        return str(output_path)
    
    def _create_led_pattern(self, led_type: int) -> tf.Tensor:
        """Create synthetic LED pattern image"""
        height, width = self.image_size
        image = tf.zeros([height, width, 3], dtype=tf.uint8)
        
        # LED colors based on state
        colors = [
            [255, 0, 0],      # red (low)
            [255, 165, 0],    # orange (medium)
            [255, 255, 0],    # yellow (high)  
            [128, 128, 128],  # gray (none)
        ]
        
        color = colors[led_type]
        # Create LED indicator
        y_start = height // 3
        y_end = 2 * height // 3
        x_start = width // 3
        x_end = 2 * width // 3
        
        image = tf.tensor_scatter_nd_update(
            image,
            [[y, x, c] for y in range(y_start, y_end) 
             for x in range(x_start, x_end) 
             for c in range(3)],
            [color[c] for _ in range(y_end - y_start) 
             for _ in range(x_end - x_start) 
             for c in range(3)]
        )
        
        return image
    
    def _create_switch_pattern(self, switch_pos: int) -> tf.Tensor:
        """Create synthetic switch pattern image"""
        height, width = self.image_size
        image = tf.ones([height, width, 3], dtype=tf.uint8) * 255
        
        # Switch base (gray rectangle)
        y_start = height // 4
        y_end = 3 * height // 4
        x_start = width // 4
        x_end = 3 * width // 4
        
        # Draw switch base
        for c in range(3):
            image = tf.tensor_scatter_nd_update(
                image,
                [[y, x, c] for y in range(y_start, y_end) 
                 for x in range(x_start, x_end)],
                [128 for _ in range((y_end - y_start) * (x_end - x_start))]
            )
        
        # Switch lever position
        lever_y = int(y_start + (y_end - y_start) * (switch_pos / 3))
        lever_x = x_start - 10
        
        # Draw lever
        for c in range(3):
            image = tf.tensor_scatter_nd_update(
                image,
                [[lever_y, x, c] for x in range(lever_x, x_start)],
                [0 for _ in range(x_start - lever_x)]
            )
        
        return image
    
    def load_from_directory(self, data_dir: str) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Load dataset from directory structure"""
        data_dir = Path(data_dir)
        
        # Get image paths and labels
        image_paths = []
        labels = []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*.jpg"):
                    image_paths.append(str(img_path))
                    labels.append(class_idx)
        
        # Convert to TensorFlow dataset
        path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
        label_ds = tf.data.Dataset.from_tensor_slices(labels)
        
        # Load and preprocess images
        def load_image(path):
            image = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, self.image_size)
            image = tf.cast(image, tf.float32) / 255.0
            return image
        
        image_ds = path_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = tf.data.Dataset.zip((image_ds, label_ds))
        
        # Split into train/validation
        total_size = len(image_paths)
        val_size = int(total_size * self.validation_split)
        train_size = total_size - val_size
        
        # Shuffle and split
        dataset = dataset.shuffle(buffer_size=total_size, seed=self.seed)
        train_ds = dataset.take(train_size)
        val_ds = dataset.skip(train_size)
        
        # Batch and optimize for microcontroller
        train_ds = train_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        return train_ds, val_ds
    
    def export_for_microcontroller(self, model, export_path: str = "microcontroller_model"):
        """Export model in microcontroller-friendly format"""
        export_path = Path(export_path)
        export_path.mkdir(exist_ok=True)
        
        # Save model in TensorFlow Lite format
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Microcontroller optimizations - fix quantization settings
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        # For full integer quantization, we need to use float32 input/output
        # and let the model handle the quantization internally
        
        # Quantization with representative dataset
        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(1, *self.image_size, 3).astype(np.float32)
                yield [data]
        
        converter.representative_dataset = representative_dataset
        
        try:
            tflite_model = converter.convert()
            
            # Save quantized model
            with open(export_path / "model_quantized.tflite", "wb") as f:
                f.write(tflite_model)
                
            # Save metadata
            metadata = {
                "input_shape": list(self.image_size) + [3],
                "output_classes": len(self.class_names),
                "class_names": self.class_names,
                "model_size_bytes": len(tflite_model),
                "optimized_for": "microcontroller",
                "image_format": "RGB",
                "input_type": "float32",
                "input_range": [0.0, 1.0]
            }
            
            with open(export_path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
                
            return str(export_path)
            
        except ValueError as e:
            print(f"Warning: Standard quantization failed, trying without quantization: {e}")
            
            # Fallback to simple conversion without quantization
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            tflite_model = converter.convert()
            
            # Save model
            with open(export_path / "model.tflite", "wb") as f:
                f.write(tflite_model)
                
            # Save metadata
            metadata = {
                "input_shape": list(self.image_size) + [3],
                "output_classes": len(self.class_names),
                "class_names": self.class_names,
                "model_size_bytes": len(tflite_model),
                "optimized_for": "microcontroller",
                "image_format": "RGB",
                "input_type": "float32",
                "input_range": [0.0, 1.0],
                "quantization": "none"
            }
            
            with open(export_path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
                
            return str(export_path)
    
    def create_roboflow_format(self, dataset_dir: str) -> str:
        """Create Roboflow-compatible dataset structure"""
        roboflow_dir = Path(dataset_dir) / "roboflow_export"
        roboflow_dir.mkdir(exist_ok=True)
        
        # Create train/valid/test splits
        for split in ["train", "valid", "test"]:
            for class_name in self.class_names:
                (roboflow_dir / split / class_name).mkdir(parents=True, exist_ok=True)
        
        # Copy images to splits
        data_dir = Path(dataset_dir)
        all_images = []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = data_dir / class_name
            if class_dir.exists():
                images = list(class_dir.glob("*.jpg"))
                all_images.extend([(img, class_idx, class_name) for img in images])
        
        # Shuffle and split
        np.random.seed(self.seed)
        np.random.shuffle(all_images)
        
        n_total = len(all_images)
        n_train = int(0.6 * n_total)
        n_valid = int(0.2 * n_total)
        
        splits = {
            "train": all_images[:n_train],
            "valid": all_images[n_train:n_train+n_valid], 
            "test": all_images[n_train+n_valid:]
        }
        
        # Copy files
        for split_name, split_images in splits.items():
            for img_path, class_idx, class_name in split_images:
                dest = roboflow_dir / split_name / class_name / img_path.name
                tf.io.gfile.copy(str(img_path), str(dest))
        
        # Create data.yaml for Roboflow
        data_yaml = f"""
train: train
val: valid
test: test

nc: {len(self.class_names)}
names: {self.class_names}

roboflow:
  workspace: microcontroller-datasets
  project: led-switch-classification
  version: 1
  license: MIT
  url: https://universe.roboflow.com/microcontroller-datasets/led-switch-classification
"""
        
        with open(roboflow_dir / "data.yaml", "w") as f:
            f.write(data_yaml.strip())
        
        return str(roboflow_dir)


# Usage example
if __name__ == "__main__":
    # Create dataset instance
    dataset = LEDSwitchDataset()
    
    # Generate sample dataset
    sample_dir = dataset.create_sample_dataset()
    print(f"Sample dataset created at: {sample_dir}")
    
    # Load dataset
    train_ds, val_ds = dataset.load_from_directory(sample_dir)
    print(f"Training batches: {len(list(train_ds))}")
    print(f"Validation batches: {len(list(val_ds))}")
    
    # Create Roboflow format
    roboflow_path = dataset.create_roboflow_format(sample_dir)
    print(f"Roboflow format exported to: {roboflow_path}")
