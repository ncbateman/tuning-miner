FROM winglian/axolotl-cloud:main-latest

ARG HUGGINGFACE_TOKEN

ARG WANDB_TOKEN

ARG AXOLOTL_PORT

WORKDIR /workspace

COPY ./docker/requirements.txt /workspace/requirements.txt

RUN pip install -r requirements.txt

RUN mkdir -p /workspace/axolotl/outputs /workspace/axolotl/data

ENV CONFIG_DIR="/workspace/config"

ENV OUTPUT_DIR="/workspace/axolotl/outputs"

RUN huggingface-cli login --token "$HUGGINGFACE_TOKEN" --add-to-git-credential
RUN wandb login "$WANDB_TOKEN"

COPY asgi.py /workspace/asgi.py
COPY endpoints/ /workspace/endpoints
COPY utilities/ /workspace/utilities
COPY models/ /workspace/models

CMD ["sh", "-c", "uvicorn --host 0.0.0.0 --port $AXOLOTL_PORT asgi:factory --factory"]
