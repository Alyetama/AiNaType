# AiNaType

AiNaType is an open-source Python project for automated image classification, primarily focused on identifying wildlife signs such as live animals, dead specimens, scat, and tracks using deep learning models. The repository leverages the Ultralytics YOLO framework and provides both API and Gradio app interfaces for prediction and demonstration.

## Features

- **YOLO-based Classification**: Utilizes YOLO models to classify images into categories like dead, live animal, scat, and tracks.
- **API Interface**: FastAPI-based endpoint (`/predict`) for programmatic image classification, supporting custom model weights and versioning.
- **Gradio Demo App**: User-friendly web app for interactive image uploads and quick predictions.
- **Annotation Utilities**: Scripts for exporting and processing annotation tasks with Label Studio integration.
- **Batch Prediction**: SQLite-based tracking of image predictions for bulk processing.
- **Jupyter Notebooks**: Data preparation and analysis notebooks for tasks and merging prediction results.
- **Flexible Configuration**: Command-line options for specifying model weights, versions, host, and port.

## Model Weights

Pre-trained model weights for AiNaType can be downloaded from the [Releases page](https://github.com/Alyetama/AiNaType/releases). Download the appropriate weights files and specify their path in your API or batch prediction scripts.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Alyetama/AiNaType.git
    cd AiNaType
    ```

2. Install requirements:
    ```bash
    pip install -r requirements.txt
    ```

3. (Optional) Set up environment variables for annotation export:
    ```
    LABEL_STUDIO_URL=<your_label_studio_url>
    API_KEY=<your_api_key>
    ```

## Usage

### Gradio Demo

Run the demo app for interactive classification:

```bash
python gradio-app-demo.py
```
Visit the displayed URL to upload images and view predictions.

### API Server

Start the FastAPI server:

```bash
python api-ainatype.py -w <path_to_weights> -m <model_version>
```

Send a POST request to `/predict` with JSON payload containing the image URL and project information.

### Batch Prediction

Process a directory of images and store results in a SQLite database:

```bash
python predict.py -d <db_path> -m <model_path> -i <images_dir>
```

## Annotation Export

Export annotations from Label Studio via shell script:

```bash
bash misc/export_annotations.sh
```

## Project Structure

- `api-ainatype.py`: FastAPI server implementation.
- `gradio-app-demo.py`: Gradio demo web app.
- `predict.py`: Batch prediction script with SQLite tracking.
- `misc/get_annotations.py`: Python script to fetch and save annotations.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Author

Developed by [Mohammad Alyetama](https://github.com/Alyetama).
