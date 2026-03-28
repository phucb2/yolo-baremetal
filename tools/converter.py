import torch
import struct
import argparse
import os

def export_yolo26_to_bin(model_path, output_path):
    print(f"Loading weights from {model_path}...")
    
    # Load raw checkpoint without instantiating the model class
    # This avoids "AttributeError: Can't get attribute 'C3k2'"
    ckpt = torch.load(model_path, map_location='cpu')
    
    # Ultralytics models usually store the state_dict in 'model' or 'ema'
    if 'model' in ckpt:
        model = ckpt['model']
    elif 'ema' in ckpt:
        model = ckpt['ema']
    else:
        model = ckpt
        
    # Check if it's a state_dict or a full model object
    if hasattr(model, 'state_dict'):
        state_dict = model.state_dict()
        nc = getattr(model, 'nc', 80) # Default to 80 if not found
    else:
        state_dict = model
        nc = 80 # We might need to pass this as an argument if not in state_dict
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        # 1. NC (Number of classes)
        f.write(struct.pack('i', nc))
        
        # 2. Total parameters count
        f.write(struct.pack('i', len(state_dict)))
        
        # 3. Export weights in state_dict order
        for name, param in state_dict.items():
            # Ensure we only export tensors
            if not isinstance(param, torch.Tensor):
                continue
                
            # Convert to float32 and flatten
            data = param.detach().cpu().numpy().astype('float32').flatten()
            
            # Metadata: Name length, Name, Dims count, Dims, Data
            name_bytes = name.encode('ascii')
            f.write(struct.pack('i', len(name_bytes)))
            f.write(name_bytes)
            
            f.write(struct.pack('i', len(param.shape)))
            for d in param.shape:
                f.write(struct.pack('i', d))
                
            f.write(data.tobytes())
            print(f"Exported: {name:.<60} {list(param.shape)}")

    print(f"\nSuccess! Weights saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to .pt model")
    parser.add_argument("--output", type=str, default="weights/yolo26.bin", help="Output .bin path")
    args = parser.parse_args()
    
    export_yolo26_to_bin(args.model, args.output)
