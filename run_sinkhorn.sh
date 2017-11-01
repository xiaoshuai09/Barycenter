#!/bin/bash
for config in 10 100 300 ; do
	{
		if [ $config == 10 ]
		then
			device=0
		elif [ $config == 100 ]
		then 
			device=1
		elif [ $config == 300 ]
		then 
			device=2
		fi
		CUDA_VISIBLE_DEVICES=$((device + 2)) python sinkhorn.py  $config |& tee ./result/$config
	} & 

done
wait
