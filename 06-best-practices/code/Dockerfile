FROM public.ecr.aws/lambda/python:3.9

RUN pip install -U pip==24.0
RUN pip install pipenv

COPY [ "Pipfile", "Pipfile.lock", "./" ]

# Use pip directly instead of pipenv to avoid the metadata issue
#RUN pipenv install --system --deploy
# Export BOTH production and dev dependencies
RUN pipenv requirements > requirements.txt && \
    pipenv requirements --dev > requirements-dev.txt && \
    pip install -r requirements.txt && \
    pip install -r requirements-dev.txt

COPY [ "lambda_function.py", "model.py", "./" ]
COPY ["tests/", "./tests/"]

CMD [ "lambda_function.lambda_handler" ]