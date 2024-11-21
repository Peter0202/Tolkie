FROM python:3.9
WORKDIR /app

COPY fietje_model.py /app/fietje_model.py

RUN pip install flask 
RUN pip install torch 
RUN pip install transformers

RUN python3 -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
                model_name = 'BramVanroy/fietje-2'; \
                AutoTokenizer.from_pretrained(model_name).save_pretrained('/app/fietje-model'); \
                AutoModelForCausalLM.from_pretrained(model_name).save_pretrained('/app/fietje-model')"

EXPOSE 8003
CMD ["python", "fietje_model.py"]
