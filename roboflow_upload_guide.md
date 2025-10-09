# Roboflow Upload Guide for Microcontroller ML Dataset

## Dataset Overview
- **Total Images**: 8 (4 per class)
- **Image Dimensions**: 160x120 pixels (optimized for microcontrollers)
- **Format**: JPEG with 85% quality
- **Classes**: 2 distinct classes for binary classification

## Class Descriptions

### Class 1: LED Status Indicators
- `led_off_1.jpg` - LED in off state (dark)
- `led_low_2.jpg` - LED at low brightness (dim green)
- `led_medium_3.jpg` - LED at medium brightness (moderate green)
- `led_high_4.jpg` - LED at high brightness (bright green with glow)

### Class 2: Switch/Button States
- `switch_off_1.jpg` - Switch in OFF position (left)
- `switch_on_2.jpg` - Switch in ON position (right)
- `switch_neutral_3.jpg` - Switch in neutral/center position
- `switch_toggle_4.jpg` - Toggle switch in up position

## Roboflow Upload Steps

1. **Create New Project**
   - Go to app.roboflow.com
   - Click "Create New Project"
   - Choose "Object Detection" or "Classification" based on your needs

2. **Upload Dataset**
   - Zip the entire `microcontroller_dataset` folder
   - Upload the zip file to Roboflow
   - Or upload images individually maintaining folder structure

3. **Annotation (if needed)**
   - For object detection: Draw bounding boxes around LEDs/switches
   - For classification: Images are already properly organized by class

4. **Dataset Configuration**
   - Train/Validation/Test split: Recommend 70/20/10
   - With only 8 images, consider 6/1/1 split
   - Enable data augmentation for better training

## Recommended Augmentations for Microcontroller Deployment
- **Brightness**: ±15% (simulates different lighting conditions)
- **Contrast**: ±10% (accounts for camera variations)
- **Noise**: Add slight Gaussian noise (simulates sensor noise)
- **Rotation**: ±5° (accounts for mounting variations)
- **Avoid**: Heavy distortions that don't reflect real deployment conditions

## Training Recommendations
- **Model**: Use lightweight models (YOLOv8n, MobileNet, or custom CNN)
- **Input Size**: Keep at 160x120 or similar small resolution
- **Epochs**: Start with 100-200 epochs given small dataset
- **Batch Size**: 4-8 (adjust based on available memory)
- **Learning Rate**: 0.001 initial, with decay

## Deployment Considerations
- **Memory**: Images optimized for <32KB each
- **Processing**: Simple features for fast inference
- **Power**: Designed for low-power microcontroller operation
- **Real-time**: Suitable for real-time classification tasks

## Next Steps After Upload
1. Train initial model with current dataset
2. Test model performance on validation set
3. Deploy to microcontroller for field testing
4. Collect additional real-world images to expand dataset
5. Retrain with expanded dataset for improved accuracy

## Technical Specifications
- **Color Space**: RGB
- **Bit Depth**: 8-bit per channel
- **Compression**: JPEG (85% quality)
- **File Size**: ~1.5-2KB per image
- **Memory Footprint**: Suitable for ESP32, Arduino with camera modules