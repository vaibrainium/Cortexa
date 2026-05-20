FROM ghcr.io/walkerlab/docker-pytorch-cuda:cuda-11.8.0-pytorch-2.6.0-torchvision-0.21.0-torchaudio-2.6.0-ubuntu-22.04

LABEL maintainer='vaibrainium (vaibhavt459@gmail.com)'


COPY . /src/

RUN apt-get update -y
RUN apt-get upgrade -y

EXPOSE 8811 7860
