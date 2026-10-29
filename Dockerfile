FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir numpy scipy matplotlib
COPY ddpm ./ddpm
VOLUME ["/app/runs", "/app/data", "/app/docs"]
ENTRYPOINT ["python", "-m"]
CMD ["ddpm.train", "--steps", "5000", "--out", "runs/mnist"]
