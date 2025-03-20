import pickle

def save_checkpoint(data, filename="../myOutputs/calculateL_checkpoint.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✅ Checkpoint saved successfully to {filename}")

# Example data to test
test_data = {"key1": "value1", "key2": [1, 2, 3]}  # Replace with your actual data
save_checkpoint(test_data)

# import pickle

# checkpoint_path = "../myOutputs/calculateL_checkpoint.pkl"

# def load_valid_data():
#     valid_data = []
#     try:
#         with open(checkpoint_path, "rb") as f:
#             while True:
#                 try:
#                     data = pickle.load(f)  # Load available data
#                     valid_data.append(data)  # Store valid parts
#                 except EOFError:
#                     print("⚠️ End of file reached (truncated). Recovering valid parts...")
#                     break  # Stop loading further
#     except Exception as e:
#         print(f"❌ Error: {e}")
    
#     if valid_data:
#         print(f"✅ Successfully recovered {len(valid_data)} valid entries.")
#     else:
#         print("❌ No recoverable data found.")
#     return valid_data

# # Attempt to recover valid data
# recovered_data = load_valid_data()


# import os
# import pickle

# checkpoint_path = "../myOutputs/calculateL_checkpoint.pkl"

# if not os.path.exists(checkpoint_path):
#     print(f"❌ Error: Checkpoint file does not exist: {checkpoint_path}")
# elif os.path.getsize(checkpoint_path) == 0:
#     print(f"❌ Error: Checkpoint file is empty: {checkpoint_path}")
# else:
#     print(f"✅ Checkpoint file exists and has size: {os.path.getsize(checkpoint_path)} bytes")
# def load_checkpoint():
#     checkpoint_path = "../myOutputs/calculateL_checkpoint.pkl"
#     with open(checkpoint_path, "rb") as f:
#         data = pickle.load(f)  # Load the data
#         print("Loaded checkpoint content:")
#         print(data)  # Print the content
#         return data  # Return the data for further use

# # Call the function
# checkpoint_data = load_checkpoint()

# def load_checkpoint():
#     checkpoint_path = "../myOutputs/calculateL_checkpoint.pkl"
#     if not os.path.exists(checkpoint_path):
#         raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
#     if os.path.getsize(checkpoint_path) == 0:
#         raise EOFError(f"Checkpoint file is empty: {checkpoint_path}")
    
#     with open(checkpoint_path, "rb") as f:
#         return pickle.load(f)