# imagefilter
This API is for image processor.
You can get the information of image you input by using this API. 

## Features

- **Lighting Analysis**: Classify images as natural light or indoor light using deep learning models
- **Scene Classification**: Identify scene types using ResNet-based models  
- **Object Detection**: YOLO-based object detection and analysis
- **Image Processing**: Comprehensive image analysis pipeline with exposure and histogram analysis
- **Batch Processing**: Process multiple images from CSV files or API endpoints

## Project Structure

```
/app/
├── app.py                 # Main FastAPI application
├── requirements.txt       # Python dependencies
├── docker-compose.yml     # Docker configuration
├── dockerfile            # Docker build instructions
├── tools/                # Core processing modules
│   ├── lighting_processor.py   # Lighting classification
│   ├── scene_processor.py      # Scene classification  
│   ├── object_processor.py     # Object detection
│   ├── image_processor.py      # Image processing utilities
│   ├── exposure_processor.py   # Exposure analysis
│   ├── data_processor.py       # Main data processing pipeline
│   └── csv_processor.py        # CSV file handling
├── utils/                # Utility modules
│   └── api.py            # API utilities
├── data/                 # Data storage
│   ├── results.json      # Processing results
│   └── logs/             # Application logs
└── test/                 # Test files and models
    ├── best_udcsit_model.pth    # Lighting classification model
    ├── resnet18_places365.pth.tar  # Scene classification model
    └── lighting_labels.txt      # Lighting class labels
```

## Installation

### Using Docker (Recommended)

```bash
docker-compose up --build
```

### Manual Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env` file:
```env
CSV_FILE=path/to/your/data.csv
RESULT_FILE=path/to/results.json
SCENE_MODEL_WEIGHT=test/resnet18_places365.pth.tar
LIGHTING_MODEL_PATH=test/best_udcsit_model.pth
LIGHTING_LABEL_FILE=test/lighting_labels.txt
```

3. Run the application:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## API Usage

### Process Images from URLs

```bash
POST /process_images
Content-Type: application/json

{
  "Data": [
    {
      "row": 1,
      "key": "image1",
      "url": "https://example.com/image1.jpg"
    }
  ]
}
```

### Health Check

```bash
GET /health
```

## Core Components

### LightingProcessor
Classifies images into lighting conditions:
- `natural_light`: Outdoor/natural lighting
- `indoor_light`: Indoor/artificial lighting

### SceneProcessor  
Identifies scene types using Places365 dataset classification.

### ObjectProcessor
Performs object detection using YOLO models.

### ImageProcessor
Handles image preprocessing, format conversion, and basic analysis.

## Model Requirements

- **Lighting Model**: PyTorch model trained for lighting classification
- **Scene Model**: ResNet18 model trained on Places365 dataset
- **Object Detection**: YOLO11 model for object detection

## Development

### Running Tests
```bash
python -m pytest test/
```

### Adding New Processors
1. Create processor class in `tools/` directory
2. Implement required methods following existing patterns
3. Integrate with `DataProcessor` in `data_processor.py`

## Docker Deployment

The application includes Docker support for easy deployment:

```bash
# Build and run with docker-compose
docker-compose up -d

# Or build manually
docker build -t image-analysis-api .
docker run -p 8000:8000 image-analysis-api
```

## Dependencies

- **FastAPI**: Web framework for API development
- **PyTorch/Torchvision**: Deep learning models
- **OpenCV**: Image processing
- **Pillow**: Image handling
- **Ultralytics**: YOLO object detection
- **Pandas**: Data manipulation
- **Pydantic**: Data validation

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
