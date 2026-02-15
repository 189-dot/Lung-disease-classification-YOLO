
import argparse
import os
from ultralytics import YOLO
# from IPython.display import Image, display # Commented out as it's for IPython environments

def main():
    parser = argparse.ArgumentParser(description="Run inference on an image using the Lung Disease Classifier model.")
    parser.add_argument('--image_path', type=str, default='data/images.jpg', help='Path to the input image for prediction (default: data/images.jpg).')
    args = parser.parse_args()

    # Load the trained model
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'best.pt')
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    model = YOLO(model_path)

    # Construct the full path to the image, relative to the script's directory
    full_image_path = os.path.join(os.path.dirname(__file__), args.image_path)
    
    # Check if the image path exists
    if not os.path.exists(full_image_path):
        print(f"Error: Image not found at {full_image_path}")
        return

    # Run inference
    print(f"Running prediction on: {full_image_path}")
    results = model.predict(
        source=full_image_path,
        conf=0.4, # Minimum confidence threshold
        save=True # Save results to 'runs/classify/predict' directory
    )

    # Display results (if running in an environment that supports it, like Colab)
    if results:
        result = results[0]
        print("\nPrediction Results:")
        for prob_idx, prob in enumerate(result.probs.data):
            class_name = result.names[prob_idx]
            print(f"  {class_name}: {prob:.2f}")

        # Optionally display the saved image with predictions
        if result.save_dir and result.path:
            output_image_path = os.path.join(result.save_dir, os.path.basename(result.path))
            if os.path.exists(output_image_path):
                print(f"\nPrediction image saved to: {output_image_path}")
                # The display function is for IPython environments, won't work in a standalone script
                # In a real script, you'd likely just view the saved image.
                # try:
                #    display(Image(filename=output_image_path))
                # except Exception as e:
                #    print(f"Could not display image: {e}. View it manually at {output_image_path}")


if __name__ == '__main__':
    main()
