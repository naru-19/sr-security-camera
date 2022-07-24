FROM nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04

ARG USERNAME=izubuchi
ARG UID
ARG GROUPNAME
ARG GID
ARG USE_SHELL=/bin/bash
ARG ROOT_PASSWORD=admin

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get -y install apt-utils software-properties-common
RUN apt-get update
RUN apt-get -y install nano vim wget curl unzip openssh-server sudo git iputils-ping net-tools netcat
RUN apt-get update && apt-get install --no-install-recommends -y \
    libgl1-mesa-dev \
    x11-apps \
    libgl1 \
    libgomp1 \
    libusb-1.0-0 \
    python3-pip \
      && rm -rf /var/lib/apt/lists/*

RUN echo 'root:admin' | chpasswd
RUN useradd -u $UID $USERNAME -m
RUN echo 'izubuchi:admin' | chpasswd
RUN usermod -aG sudo $USERNAME
RUN chsh -s /bin/bash $USERNAME

RUN mkdir /var/run/sshd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i 's/#PasswordAuthentication/PasswordAuthentication/' /etc/ssh/sshd_config
# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

RUN pip install --upgrade pip
RUN pip install git+https://github.com/sabjorn/NumpySocket.git
RUN pip install torch torchvision torchsummary torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
COPY ./requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt

# 超解像系
RUN pip install git+https://github.com/naru-19/Myutils
RUN pip install basicsr
RUN pip install facexlib
RUN pip install git+https://github.com/TencentARC/GFPGAN
RUN pip install realesrgan


WORKDIR /home/${USERNAME}
RUN mkdir /home/${username}/.jupyter && \
    chmod -R 777 /home/${username} && \
    echo "c.NotebookApp.password='$(python3 -c "from notebook.auth import passwd; print(passwd('admin'))")'" \
    >> /home/${username}/.jupyter/jupyter_notebook_config.py


EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
# 起動後にsudo /usr/sbin/sshd
