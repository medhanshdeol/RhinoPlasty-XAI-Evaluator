import urllib.request
import os

def download_dummy_data():
    """Downloads a public domain face image for MediaPipe testing."""
    os.makedirs("test_images", exist_ok=True)
    
    # Download a sample face from unsplash source (using a specific ID to ensure it's a face)
    url = "https://images.unsplash.com/photo-1542909168-82c3e7fdca5c?w=500&q=80"
    target_path = "test_images/dummy_pre.jpg"
    
    print(f"Downloading test image to {target_path}...")
    try:
        urllib.request.urlretrieve(url, target_path)
        print("Download complete.")
        
        # We will just duplicate it for 'post' for structural testing
        target_path_post = "test_images/dummy_post.jpg"
        urllib.request.urlretrieve(url, target_path_post)
        print(f"Created duplicate at {target_path_post}")
        
    except Exception as e:
        print(f"Failed to download image: {e}")

if __name__ == "__main__":
    download_dummy_data()
