-build singularity image:
    $ sudo singularity build singularity.sif config.txt
-test singularity:
    $ singularity exec singularity.sif <command> 
-file.slurm will execute singularity on blade
    add --nv flag if gpu is needed