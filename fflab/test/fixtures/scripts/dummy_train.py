import json, os, sys, time
out = os.environ["DUMMY_OUT"]
os.makedirs(out, exist_ok=True)
steps = int(os.environ["DUMMY_STEPS"])
for i in range(steps):
    print(f"step {i + 1}/{steps} fixed={os.environ.get('DUMMY_FIXED')}", flush=True)
    time.sleep(0.05)
if os.environ["DUMMY_FAIL"] == "1":
    print("Traceback (most recent call last): simulated failure", flush=True)
    sys.exit(3)
json.dump({"steps": steps}, open(os.path.join(out, "result.json"), "w"))
open(os.path.join(out, "ckpt.pt"), "wb").write(b"x" * 10)
os.makedirs(os.path.join(out, "sub"), exist_ok=True)
open(os.path.join(out, "sub", "fly.mp4"), "wb").write(b"video")
print("[DONE]", flush=True)
