#!/bin/bash
for config in 0.01 0.1 1 10 ; do
	{
		if [ $config == 0.01 ]
		then
			device=0
		elif [ $config == 0.1 ]
		then 
			device=1
		elif [ $config == 1 ]
		then 
			device=2
		elif [ $config == 10 ]
		then 
			device=3
		fi
		CUDA_VISIBLE_DEVICES=$((device + 2)) python dual.py  $config |& tee ./result/$config
	} & 

done
wait
