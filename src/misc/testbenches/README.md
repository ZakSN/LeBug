This directory contains ad hoc testbenches used during development.

To run one:

1. start modelsim
```
$ xhost local:`whoami` # or other OS equivalent
$ docker start modelsim
$ docker attach modelsim
[modelsim_container]$ vsim -gui
```

2. copy relevant source files into the docker container (from a new terminal).
For example:
```
docker cp dc_tb_temp.sv modelsim:/src/dc_tb_temp.sv && docker cp delta_compressor.sv modelsim:/src/delta_compressor.sv && docker cp trace_buffer.sv modelsim:/src/trace_buffer.sv && docker cp reconfig_unit.sv modelsim:/src/reconfig_unit.sv && docker cp ../device-specific/ram_dual_port_cycloneV.sv modelsim:/src/ram_dual_port.sv && docker cp ../../simulationBlocks/altera_lnsim.sv modelsim:/src/altera_lnsim.sv && docker cp ../../simulationBlocks/altera_mf.v modelsim:/src/altera_mf.v
```

3. run the simulation in vsim, something like:
```
VSIM > vlog /src/dc_tb_temp
VSIM > run -all
```
