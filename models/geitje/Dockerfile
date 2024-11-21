FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04

RUN apt-get update && \
    apt-get install -y python3.9 python3.9-distutils curl && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9 && \
    ln -s /usr/bin/python3.9 /usr/bin/python3

WORKDIR /app

COPY geitje_model.py /app/geitje_model.py
COPY model /app/geitje-model

RUN pip install flask transformers sentencepiece blobfile protobuf \
    && pip install torch --extra-index-url https://download.pytorch.org/whl/cu116

RUN if [ ! -d "/app/geitje-model" ]; then \
    python3 -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
                model_name = 'Rijgersberg/GEITje-7B'; \
                AutoTokenizer.from_pretrained(model_name).save_pretrained('/app/geitje-model'); \
                AutoModelForCausalLM.from_pretrained(model_name).save_pretrained('/app/geitje-model')"; \
    else \
        echo "Using local model files"; \
    fi

EXPOSE 8002

CMD ["python3", "geitje_model.py"]
