for epoch in 5

do

for batch_size in 16

do

for lr in 0.01 

do

python run.py --epoch ${epoch} --batch_size ${batch_size} --lr ${lr}

done

done

done