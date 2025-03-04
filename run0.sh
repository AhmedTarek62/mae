#!/bin/bash
python finetune.py --task positioning --device cuda:0 --epochs 50 --label spects --prefix
python finetune.py --task positioning --device cuda:0 --epochs 50 --label spects --ePrefix

python finetune.py --task csi_sensing --device cuda:0 --epochs 50 --label spects --ePrefix
python finetune.py --task csi_sensing --device cuda:0 --epochs 50 --label spects --ePrefix --lora
python finetune.py --task csi_sensing --device cuda:0 --epochs 50 --label spects --prefix --lora
python finetune.py --task csi_sensing --device cuda:0 --epochs 50 --label spects --prefix
python finetune.py --task csi_sensing --device cuda:0 --epochs 50 --label spects 
python finetune.py --task csi_sensing --device cuda:0 --epochs 50 --label spects --lora

python finetune.py --task csi_sensing --device cuda:0 --epochs 50 --label all_small --ePrefix --checkpoint /home/elsayedmohammed/vit-models/pretrained_all_data.pth --model vit_small_patch16
python finetune.py --task csi_sensing --device cuda:0 --epochs 50 --label all_small --ePrefix --lora --checkpoint /home/elsayedmohammed/vit-models/pretrained_all_data.pth --model vit_small_patch16
python finetune.py --task csi_sensing --device cuda:0 --epochs 50 --label all_small --prefix --lora --checkpoint /home/elsayedmohammed/vit-models/pretrained_all_data.pth --model vit_small_patch16
python finetune.py --task csi_sensing --device cuda:0 --epochs 50 --label all_small --prefix --checkpoint /home/elsayedmohammed/vit-models/pretrained_all_data.pth --model vit_small_patch16
python finetune.py --task csi_sensing --device cuda:0 --epochs 50 --label all_small --checkpoint /home/elsayedmohammed/vit-models/pretrained_all_data.pth --model vit_small_patch16
python finetune.py --task csi_sensing --device cuda:0 --epochs 50 --label all_small --lora --checkpoint /home/elsayedmohammed/vit-models/pretrained_all_data.pth --model vit_small_patch16


# Positioning - all
python finetune.py --task positioning --device cuda:0 --epochs 50 --label small_all --checkpoint /home/elsayedmohammed/vit-models/pretrained_all_data.pth --model vit_small_patch16
python finetune.py --task positioning --device cuda:0 --epochs 50 --label small_all --prefix --checkpoint /home/elsayedmohammed/vit-models/pretrained_all_data.pth --model vit_small_patch16
python finetune.py --task positioning --device cuda:0 --epochs 50 --label small_all --ePrefix --checkpoint /home/elsayedmohammed/vit-models/pretrained_all_data.pth --model vit_small_patch16