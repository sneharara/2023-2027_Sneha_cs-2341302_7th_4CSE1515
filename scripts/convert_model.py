import os
import sys

def convert():
    # Helper to check imports
    try:
        import tensorflow as tf
        import tensorflowjs as tfjs
    except ImportError as e:
        print("\n❌ Error: Missing required dependencies to convert the model.")
        print("Please install them by running: ")
        print("   pip install tensorflow tensorflowjs\n")
        sys.exit(1)

    # Resolve paths relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # We look for the model inside backend/model/
    model_dir = os.path.join(project_root, "backend", "model")
    h5_path = os.path.join(model_dir, "fire_smoke_model.h5")
    
    # Check if the .h5 model file exists
    if not os.path.exists(h5_path):
        print(f"\n❌ Error: Keras model file not found at: {h5_path}")
        print("Please follow these steps:")
        print("  1. Download your trained 'fire_smoke_model.h5' model from Google Colab.")
        print("  2. Create a folder named 'model' inside 'smoke-fire-app/backend/'.")
        print("  3. Paste the 'fire_smoke_model.h5' file inside that folder.")
        print(f"  4. Re-run this script: python scripts/convert_model.py\n")
        sys.exit(1)

    output_dir = os.path.join(project_root, "frontend", "static", "model")
    os.makedirs(output_dir, exist_ok=True)

    print(f"🔄 Loading Keras model from {h5_path}...")
    try:
        # Load the model
        model = tf.keras.models.load_model(h5_path)
        print("Model loaded successfully.")
        
        print(f"🔄 Converting model to TensorFlow.js Layers format...")
        print(f"Saving to directory: {output_dir}...")
        
        # Save Keras model as TF.js model.json + binary shard files
        tfjs.converters.save_keras_model(model, output_dir)
        
        print("\n✅ Success! Model converted and saved to frontend/static/model/")
        print("Generated files:")
        for file in os.listdir(output_dir):
            print(f"  - {file}")
            
    except Exception as e:
        print(f"\n❌ Error occurred during model loading/conversion: {e}")
        print("If you encounter keras configuration issues, make sure your python tensorflow version matches the one in Colab.")
        sys.exit(1)

if __name__ == "__main__":
    convert()
