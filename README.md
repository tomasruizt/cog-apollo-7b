# Apollo-LMMs/Apollo-7B-t32 Cog model

This is an implementation of the Huggingface space [Apollo-LMMs/Apollo-7B-t32](https://huggingface.co/Apollo-LMMs/Apollo-7B-t32) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

Run a prediction:
```bash
cog predict -i video=@astro.mp4 -i messages='[{"role": "user", "content": "Describe this video in detail"}]'
```

## Input

![alt text](astro.gif)

## Output:

    The video features an astronaut in a white spacesuit walking on the moon's surface. The background showcases a large, detailed moon against a starry sky. As the astronaut walks, they begin to run and eventually leap into the air, floating above the moon's rocky terrain. The scene transitions to the astronaut drifting away from the moon, with the lunar landscape and the moon itself visible in the background. The video concludes with the astronaut continuing to float in space, gazing at the moon.


# Build & Run Locally (Docker)
Build the container:
```bash
cog build -t replicate-apollo-7b
```
Download and unpack the model weights:
```bash
mkdir checkpoints
cd checkpoints
wget https://weights.replicate.delivery/default/Apollo-LMMs/Apollo-7B-t32/7b.tar
tar -xvf 7b.tar
```

To get a prediction, run the command `cog predict` command above.
