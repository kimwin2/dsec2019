echo '1. Download original dataset (mp4)'
wget ftp://miro.kaist.ac.kr:8048/dsec/raw_dataset.zip 
unzip raw_dataset.zip 
rm raw_dataset.zip

python3 0_convert_images.py 
python3 1_make_dataset.py

#python3 0_convert_raw_data.py

#echo '2. Download inpainted dataset using histogram-based pooling' 
#wget ftp://miro.kaist.ac.kr:8048/ky_opt/pdm_dataset_hp.zip .
#unzip pdm_dataset_hp.zip
#rm pdm_dataset_hp.zip

#echo '3. Download inpainted dataset using ns' 
#wget ftp://miro.kaist.ac.kr:8048/ky_opt/pdm_dataset_ns.zip .
#unzip pdm_dataset_ns.zip
#rm pdm_dataset_ns.zip

#echo '4. Download inpainted dataset using telea' 
#wget ftp://miro.kaist.ac.kr:8048/ky_opt/pdm_dataset_telea.zip .
#unzip pdm_dataset_telea.zip
#rm pdm_dataset_telea.zip
