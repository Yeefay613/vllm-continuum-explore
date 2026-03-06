import requests, time, json

URL = "http://localhost:8100/v1/chat/completions"
payload = {
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "messages": [{"role": "user", "content": "Write a 3-sentence summary of the moon."}],
  "max_tokens": 128,
  "temperature": 0,
}
t0 = time.time()
n = 10
for i in range(n):
    r = requests.post(URL, json=payload)
    r.raise_for_status()
dt = time.time() - t0
print(f"{n} requests in {dt:.2f}s -> {n/dt:.2f} req/s")