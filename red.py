# redownload_facenet.py
import os, time, traceback
from facenet_pytorch import InceptionResnetV1
import torch

cache_dir = os.path.join(os.path.expanduser("~"), ".facenet_pytorch", "models")
os.makedirs(cache_dir, exist_ok=True)
cached_file = os.path.join(cache_dir, "vggface2.pt")

def remove_if_exists(path):
    try:
        if os.path.exists(path):
            os.remove(path)
            print("Removed:", path)
    except Exception as e:
        print("Could not remove file:", e)

def try_load(pretrained='vggface2', device='cpu', max_attempts=4):
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        print(f"Attempt {attempt}/{max_attempts} to load model...")
        try:
            model = InceptionResnetV1(pretrained=pretrained).eval().to(device)
            print("Model loaded successfully.")
            return model
        except RuntimeError as e:
            # specific check for corrupted file / EOF
            msg = str(e).lower()
            print("RuntimeError:", msg[:200])
            if 'unexpected eof' in msg or 'truncated' in msg or 'corrupt' in msg:
                print("Detected corrupted download. Removing cached file and retrying...")
                remove_if_exists(cached_file)
            else:
                # unknown runtime error — re-raise after showing trace
                traceback.print_exc()
                raise
        except Exception as e:
            print("Other exception:", e)
            traceback.print_exc()
        # small backoff
        time.sleep(3 * attempt)
    raise RuntimeError("Failed to load model after retries.")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # قبل التشغيل: أوقف مؤقتًا antivirus/VPN لتحسين ثبات التحميل
    m = try_load(device=device)
    print("Done. Now try running your main script.")
