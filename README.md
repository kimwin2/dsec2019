## DSEC welding defect classification 

##### Data download 받기
## test

1) 0.1_binarize.py실행하여 label파일 binarize화 한다. 
2) 0.2_sizing.py실행하여 모든 파일 resize시킨다.
   - resize전에 모든 파일 이름을 4자리 숫자(예: 0001.jpg)로 맞춘다. dark namer프로그램 등을 이용하라.
3) 1_make_dataset.py 를 실행하여 pt파일을 생성한다. 생성된 파일은 pt_dataset폴더에 들어있다. 

2_train.py 
학습하는 코드. 

python3 2_train.py --gpu_ids 0 --is_attn --batch_size 2 

결과가 나옴. 

3_visualization.py 
결과를 그림으로 저장 -> display 폴더에. 

