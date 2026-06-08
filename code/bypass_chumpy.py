import os
import pickle
import re

def force_clean_file(src_path, dest_path):
    print(f"Processing: {src_path}")
    
    # 1. Read the entire file into memory as raw bytes
    with open(src_path, 'rb') as f:
        raw_bytes = f.read()
    
    # 2. Globally replace chumpy object initializers with numpy array initializers
    # This edits the pickle assembly instructions directly in the byte stream
    cleaned_bytes = raw_bytes.replace(b'cchumpy.ch\nCh\n', b'cnumpy\narray\n')
    cleaned_bytes = cleaned_bytes.replace(b'cchumpy\nCh\n', b'cnumpy\narray\n')
    
    # 3. Handle specific internal structural changes across different SMPL versions
    # This prevents the 4-item tuple/dict layout error entirely
    cleaned_bytes = re.sub(b'cchumpy\.[a-zA-Z_\n]+', b'cnumpy\narray\n', cleaned_bytes)

    # 4. Safely safely load the modified byte stream 
    try:
        data = pickle.load(os.fspath(src_path) if hasattr(os, 'fspath') else torch_io_fallback(cleaned_bytes), encoding='latin1')
    except Exception:
        # Fallback using standard io bytes loading if direct interpretation acts up
        import io
        data = pickle.load(io.BytesIO(cleaned_bytes), encoding='latin1')

    # 5. Re-save as a pure, clean modern dictionary
    clean_data = {}
    for k, v in data.items():
        # Strip away any lingering custom dictionary types left over by ancient scipy
        if type(v).__name__ in ['Ch', 'ch', 'csc_matrix']:
            continue 
        clean_data[k] = v

    with open(dest_path, 'wb') as f:
        pickle.dump(clean_data, f, protocol=4)
        
    print(f"Successfully converted and saved: {dest_path}\n")

# Point to your SMPL directory
model_dir = './code/nets/'

for gender in ['FEMALE', 'MALE', 'NEUTRAL']:
    file_name = f'SMPL_{gender}.pkl'
    full_path = os.path.join(model_dir, file_name)
    
    if os.path.exists(full_path):
        force_clean_file(full_path, full_path)
    else:
        print(f"Could not find {file_name}")
