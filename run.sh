bash submit.sh vitl-full-rank --backbone-training full
bash submit.sh vitl-lora-rank-8 --backbone-training lora --lora-rank 8 
bash submit.sh vitl-lora-rank-16 --backbone-training lora --lora-rank 16
bash submit.sh vitl-full-rank-low-res --low-res --backbone-training full
bash submit.sh vitl-lora-rank-8-low-res --low-res --backbone-training lora --lora-rank 8 
bash submit.sh vitl-lora-rank-16-low-res --low-res --backbone-training lora --lora-rank 16