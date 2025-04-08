# American Sign Computer Language (ASCL) Recognizer

An interactive tool for creating and recognizing custom hand gestures, supporting both dynamic (movement-based) and static poses.

## Overview

This project implements a versatile hand gesture recognition system using:
- Jackknife.py - Time series pattern recognition algorithm
- Machete.py - A segmentation technique 
- MediaPipe - Hand tracking and landmark detection
- OpenCV - Computer vision and video processing

## Key Features

- Real-time gesture recognition for both static poses and dynamic movements
- Custom gesture template creation and management
- Multi-threaded processing architecture  
- 3D hand landmark tracking with high precision
- Configurable gesture matching parameters
- Support for both quick poses and complex movement sequences

## Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Launch the main recognition system:
```
python Scripts/main.py
```

4. For recording new gesture templates:
```
python Scripts/TemplateCrafter.py
```

## Usage

### Real-time Recognition:

- Position your hand in front of the camera
- For static gestures: Hold the pose for recognition
- For dynamic gestures: Allow ~3 seconds for the gesture buffer to fill
- Perform gestures naturally
- Recognition results appear in the console

### Template Creation:

- Use TemplateCrafter.py to record new gestures
- Choose between static pose or dynamic movement recording
- Review recordings with frame-by-frame playback
- Save templates for recognition training

## Dependencies

- OpenCV (opencv-python, opencv-contrib-python) - Video processing
- MediaPipe - Hand tracking
- NumPy - Numerical processing
- Pillow - Image processing

## Development Status

Currently in active development with focus on:

- GUI 3.0 implementation
- Expanded gesture recognition set
- Performance optimization
- Template management improvements

See checklist.md for detailed development status.

## Technical Details

The system uses:

- Dynamic Time Warping (DTW) for gesture matching
- Position-based matching for static poses
- MediaPipe hand landmark detection
- Multi-threaded gesture processing pipeline
- Rate-limited recognition output
- Configurable gesture confidence thresholds

## References

- [Jackknife Repository](https://github.com/ISUE/Jackknife)
- [Machete Repository](https://github.com/ISUE/Machete)
- [MediaPipe Documentation](https://ai.google.dev/edge/mediapipe/solutions/guide)
- [OpenCV Documentation](https://opencv.org/) 

## Known Limitations

- Template recording requires separate window to main application
- Recognition requires consistent lighting/quality conditions
- Limited to single-hand gestures currently

## Future Developments

- Two-handed gesture support
- Improved template management system
- Improved static/dynamic gesture discrimination
