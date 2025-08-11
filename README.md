
How to Run: 




Install dependencies:
pip install -r requirements.txt



run:
python main.py \
    --rgb roofs/O_3bands_5cm_4010.tif \
    --dsm roofs/DSM_15cm_4010.tif \
    --out output/refined_facets.gpkg \
    --layer roofs



    
Where: 
rgb- path to rgb image
dsm- path to DSM file
out- output path
layer- output layer name
