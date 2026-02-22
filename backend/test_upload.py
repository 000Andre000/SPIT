import requests
import sys

video_path = r"D:\SPIT\frontend\public\videos\input_video.mp4"
endpoint = "http://localhost:8000/predict/video"

print(f"Testing upload to: {endpoint}")
print(f"Video file: {video_path}")

try:
    with open(video_path, "rb") as f:
        files = {"file": ("test_video.mp4", f, "video/mp4")}
        headers = {"X-Request-ID": "test-request-123"}
        
        print("\nSending request...")
        response = requests.post(endpoint, files=files, headers=headers, timeout=300)
        
        print(f"\nStatus Code: {response.status_code}")
        print(f"Content-Type: {response.headers.get('content-type')}")
        
        if response.status_code == 200:
            print(f"Success! Response size: {len(response.content)} bytes")
            output_path = "test_output_video.mp4"
            with open(output_path, "wb") as out:
                out.write(response.content)
            print(f"Saved to: {output_path}")
        else:
            print(f"\nError Response:")
            print(response.text[:500])
            
except Exception as e:
    print(f"\nException: {e}")
    import traceback
    traceback.print_exc()
