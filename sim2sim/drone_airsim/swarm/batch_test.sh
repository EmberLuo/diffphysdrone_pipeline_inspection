for i in $(seq 10); do
python eval.py --resume swarm.pth --target_speed 2.5
done
