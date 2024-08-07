# CI setup

## gitlab runner
Docker is the recommended excecutor of the VCM-RS CI. The docker image can be created using the script and Dockerfile in the `Docker` folder. 

To register a runner, run command `gitlab-runner register` and follow the instructions. After setting the URL and registration token, enter a description for your runner, leave the tags and optional maintenance note empty. For the gitlab-runner executer, enter `docker`, and enter the docker image name, for example `vcm:latest`. 

Note that VCM-RS test code generate a large amount of logs, the following line shall be put to /etc/gitlab-runnder/config.toml to increase the log size. 

```output_limit = 20000```



## Enable CI using docker and GPU


The instructions of enabling CI using docker and GPU can be found at 
   https://www.howtogeek.com/devops/how-to-use-an-nvidia-gpu-with-docker-containers/
   https://docs.gitlab.com/runner/configuration/gpus.html


  - Install nvidia docker using the following commands

```
  distribution=$(. /etc/os-release;echo $ID$VERSION_ID) 
     && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - 
     && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
  apt-get update
  apt-get install -y nvidia-docker2
```

  - restart docker using command `sudo systemctl restart docker`

  - Enable gitlab runner using all GPUs: in /etc/gitlab-runner/config.toml, add the following line
```
  [runners.docker]
      gpus = "all"
```
  

  - To use the docker image in local machine, put the following line to the docker section. 

```
pull_policy = ["if-not-present"]
```




The following is an examle of `config.toml` file

```
oncurrent = 1
check_interval = 0
shutdown_timeout = 0

[session_server]
  session_timeout = 1800

[[runners]]
  name = "vcm_runner"
  url = "xxxxxxxxxxxxx"
  id = 271
  token = "xxxxxxxxxxxxxxx"
  token_obtained_at = 2023-07-04T10:55:42Z
  token_expires_at = 0001-01-01T00:00:00Z
  executor = "docker"
  output_limit = 20000
  [runners.cache]
    MaxUploadedArchiveSize = 0
  [runners.docker]
    tls_verify = false
    image = "vcm:latest"
    privileged = false
    disable_entrypoint_overwrite = false
    oom_kill_disable = false
    disable_cache = false
    volumes = ["/cache"]
    shm_size = 0
    pull_policy = ["if-not-present"]
    gpus = "all"
```

# Misc

## setting MPEG_CONTENT_PASSWORD

The install.sh of VCM-RS requires the MPEG_CONTENT_PASSWORD to download pretrained models and data from the mpeg content ftp server. To set the password, go to `settings -> CI/CD -> variables`, add variable `GITLAB_PASSWORD` and set the value to the correct mpeg content server password. Note that the value shall be updated after the mpeg password is changed.

## setting maximum job timeout value
Note that the the CI make take a longer time the that maximum job timeout value, you may need to modify `Maximum job timeout` from gitlab UI in `Settings -> CI/CD -> Runner page and click the edit button next to your assigned project runner instance. 

