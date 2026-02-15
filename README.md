# Lung Disease Classifier

This project provides a simple deep learning model to classify lung diseases from medical images using YOLOv8.

## Project Structure

```
lung-disease-classifier/
├── example_image.jpg     # Example image for prediction
├── models/               # Trained YOLOv8 classification model (best.pt)
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
├── predict.py            # Script to run predictions
└── train.py              # Script to train the model (placeholder)
```

## Setup

1.  **Clone the repository (if applicable) or create the project structure**:

    ```bash
    mkdir lung-disease-classifier
    cd lung-disease-classifier
    mkdir models
    ```

2.  **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3.  **Place the trained model and example image**:
    *   Move `best.pt` into the `models/` directory.
    *   Place an example image (e.g., `example_image.jpg`) directly into the `lung-disease-classifier/` root directory.

## Usage (Prediction)

To run a prediction on the default example image:

```bash
python predict.py
```

To run a prediction on a specific image:

```bash
python predict.py --image_path path/to/your/image.jpg
```

This will output the classification results and save the predicted image (if configured).

## Usage (Training)

To train the model (using the placeholder script):

```bash
python train.py --data_path /path/to/your/dataset --epochs 50 --imgsz 224 --batch 32
```

**Note**: The `train.py` script is a placeholder and requires actual dataset integration and potentially more detailed configuration.
