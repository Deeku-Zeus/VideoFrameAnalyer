import hashlib
import time
import os

def generate_unique_hash():
    # Combine a unique identifier with current time and random bytes
    unique_string = f"{time.time()}-{os.urandom(16)}"
    
    # Create a SHA-256 hash
    hash_object = hashlib.sha256(unique_string.encode('utf-8'))
    unique_hash = hash_object.hexdigest()
    
    return unique_hash

# Example usage
print(generate_unique_hash())
