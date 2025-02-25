
## Docker Build Setup EC2
```
sudo apt update && sudo apt upgrade -y
sudo apt install docker.io -y
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
newgrp docker
docker --version
```

### EC2 NVIDIA-DRIVER install g4dn.xlarge
```sh
sudo apt update && sudo apt upgrade -y
sudo apt install -y gcc make
sudo apt install -y nvidia-headless-535-server nvidia-utils-535-server
sudo reboot
nvidia-smi
```
### EC2 docker `--gpu all` flag fix

- https://stackoverflow.com/questions/75118992/docker-error-response-from-daemon-could-not-select-device-driver-with-capab
- https://documentation.ubuntu.com/aws/en/latest/aws-how-to/instances/install-nvidia-drivers/


```sh
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey |sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
&& curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
&& sudo apt-get update

sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Run Docker image
```sh
docker run -it -p 8888:8888 --gpus all tarune14/sam2-dino:1.0 bash
```
```sh
jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root
```
