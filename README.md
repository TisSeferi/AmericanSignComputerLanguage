# American Sign Computer Language (ASCL) Recognizer

A real-time American Sign Language recognition system that combines advanced gesture recognition algorithms with computer vision technologies.

## Overview

This project implements a gesture recognition system for American Sign Language using:
- Jackknife.py - Time series pattern recognition algorithm
- Machete.py - A segmentation technique 
- [MediaPipe](https://mediapipe.dev/) - Hand tracking and landmark detection
- [OpenCV](https://opencv.org/) - Computer vision and video processing

## Key Features

- Real-time ASL gesture recognition
- Multi-threaded processing architecture  
- 3D hand landmark tracking
- Template recording and management
- Configurable gesture matching parameters

## Supported ASL Gestures

Currently recognizes the following ASL signs:
- "Forget" 
- "Thank you"
- "Like"
- "No"
- "Need"

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
- Allow ~3 seconds for the gesture buffer to fill
- Perform ASL gestures naturally
- Recognition results appear in the console

### Template Creation:

- Use TemplateCrafter.py to record new gestures
- Review recordings with frame-by-frame playback
- Save templates for recognition training

## Dependencies

- OpenCV (opencv-python, opencv-contrib-python) - Video processing
- MediaPipe - Hand tracking
- NumPy - Numerical processing
- Pillow - Image processing

## Development Status

Currently in active development with focus on:

- GUI 2.0 implementation
- Expanded gesture recognition set
- Performance optimization
- Template management improvements

See checklist.md for detailed development status.

## Technical Details

The system uses:

- Dynamic Time Warping (DTW) for gesture matching
- MediaPipe hand landmark detection
- Multi-threaded gesture processing pipeline
- Rate-limited recognition output
- Configurable gesture confidence thresholds

## References

- Jackknife Repository [text](https://github.com/ISUE/Jackknife)
- Machete Repository [text](https://github.com/ISUE/Machete)
- MediaPipe Documentation

## Known Limitations

- Template recording requires manual frame selection
- Recognition requires consistent lighting conditions
- Limited to single-hand gestures currently

## Future Developments

- Two-handed gesture support
- Improved template management system
- Automated gesture segmentation
- Extended ASL vocabulary support
