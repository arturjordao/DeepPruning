FROM python:3.9
LABEL authors="Vitor Sasaki"

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir numpy>=1.16.5 --force-reinstall
RUN pip install --no-cache-dir scipy>=1.3.1 --force-reinstall
RUN pip install --no-cache-dir scikit-learn>=0.21.3 --force-reinstall
RUN pip install --no-cache-dir tqdm>=4.32.2 --force-reinstall
RUN pip install --no-cache-dir tensorflow==2.10.1 --force-reinstall
RUN pip install --no-cache-dir keras==2.10.0 --force-reinstall

ENTRYPOINT ["/bin/bash"]