# EfficientDet-UK-RoadSigns-94classes
Training and detection RoadSigns in UK by EfficientDet

<h2>
EfficientDet UK RoadSigns 94classes (Updated: 2022/07/05)
</h2>

This is a simple python example to train and detect RoadSigns in UK based on 
<a href="https://github.com/google/automl/tree/master/efficientdet">Google Brain AutoML efficientdet</a>.
<li>
Modified to use tensorflow 2.8.0 on Windows11. (2022/06/17)<br>
</li>
<li>
Modified to use the latest <a href="https://github.com/google/automl/tree/master/efficientdet">google/automl/efficientdet</a>.(2022/06/13)<br>
</li>

<li>
Modified to use the bat files in ./projects/UK_RoadSigns/.(2022/06/14)<br>
</li>

<li>
Modified to use mixed_size tfrecord dataset.(2022/07/05)<br>
</li>
<br>
<h2>
1. Installing tensorflow on Windows11
</h2>
We use Python 3.8.10 to run tensoflow 2.8.0 on Windows11.<br>
<h3>1.1 Install Microsoft Visual Studio Community</h3>
Please install <a href="https://visualstudio.microsoft.com/ja/vs/community/">Microsoft Visual Studio Community</a>, 
which can be used to compile source code of 
<a href="https://github.com/cocodataset/cocoapi">cocoapi</a> for PythonAPI.<br>
<h3>1.2 Create a python virtualenv </h3>
Please run the following command to create a python virtualenv of name <b>py38-efficientdet</b>.
<pre>
>cd c:\
>python38\python.exe -m venv py38-efficientdet
>cd c:\py38-efficientdet
>./scripts/activate
</pre>
<h3>1.3 Create a working folder </h3>
Please create a working folder "c:\google" for your repository, and install the python packages.<br>

<pre>
>mkdir c:\google
>cd    c:\google
>pip install cython
>git clone https://github.com/cocodataset/cocoapi
>cd cocoapi/PythonAPI
</pre>
You have to modify extra_compiler_args in setup.py in the following way:
<pre>
   extra_compile_args=[]
</pre>
<pre>
>python setup.py build_ext install
</pre>

<br>

<br>
<h2>
2. Installing EfficientDet-UK-RoadSigns
</h2>
<h3>2.1 clone EfficientDet-UK-RoadSigns-94classes</h3>

Please clone EfficientDet-UK-RoadSigns-94classes in the working folder <b>c:\google</b>.<br>
<pre>
>git clone  https://github.com/atlan-antillia/EfficientDet-UK-RoadSigns-94classes.git<br>
</pre>
You can see the following folder <b>projects</b> in  EfficientDet-UK-RoadSigns-94classes folder of the working folder.<br>

<pre>
EfficientDet-Slightly-Realistic-UK-RoadSigns-94classes
└─projects
    └─UK_RoadSigns
        ├─eval
        ├─saved_model
        │  └─variables
        ├─realistic-test-dataset
        └─realistic-test-dataset_outputs        
</pre>



<br>
<h3>2.2 Install python packages</h3>

Please run the following command to install python packages for this project.<br>
<pre>
>cd ./EfficientDet-UK-RoadSigns-94classes
>pip install -r requirements.txt
</pre>

<h3>2.3 Download TFRecord dataset</h3>
 You can download TRecord_UK_RoadSigns 94classes dataset from 
<a href="https://drive.google.com/file/d/1yyOT8Sy7PJn4QrAmQGxt1LD5uMSB0D5c/view?usp=sharing">UK_RoadSigns_94classes (Updated:2022/07/05)</a>
<br>
The downloaded train and valid dataset must be placed in ./projects/UK_RoadSigns folder.
<pre>
└─projects
    └─UK_RoadSigns
        ├─train
        └─valid
</pre>
The train and valid folders contain the following tfrecord files:<br>
<pre>
    └─UK_RoadSigns
        ├─train
        │  └─mixed_size_train.tfrecord
        │  └─train.tfrecord
        └─valid
           └─mixed_size_valid.tfrecord
           └─valid.tfrecord
</pre>
We have added <b>mixed_size_train.tfrecord</b> and <b>mixed_size_valid.tfrecord</b> to improve inference accuracy for realistic_test_dataset (2022/07/05).<br>
<br>


<h3>2.4 Workarounds for Windows</h3>
As you know or may not know, the efficientdet scripts of training a model and creating a saved_model do not 
run well on Windows environment in case of tensorflow 2.8.0(probably after the version 2.5.0) as shown below:. 
<pre>
INFO:tensorflow:Saving checkpoints for 0 into ./models\model.ckpt.
I0609 06:22:50.961521  3404 basic_session_run_hooks.py:634] Saving checkpoints for 0 into ./models\model.ckpt.
2022-06-09 06:22:52.780440: W tensorflow/core/framework/op_kernel.cc:1745] OP_REQUIRES failed at save_restore_v2_ops.cc:110 :
 NOT_FOUND: Failed to create a NewWriteableFile: ./models\model.ckpt-0_temp\part-00000-of-00001.data-00000-of-00001.tempstate8184773265919876648 :
</pre>

The real problem seems to happen in the original <b> save_restore_v2_ops.cc</b>. The simple workarounds to the issues are 
to modify the following tensorflow/python scripts in your virutalenv folder. 
<pre>
c:\py38-efficientdet\Lib\site-packages\tensorflow\python\training
 +- basic_session_run_hooks.py
 
634    logging.info("Saving checkpoints for %d into %s.", step, self._save_path)
635    ### workaround date="2022/06/18" os="Windows"
636    import platform
637    if platform.system() == "Windows":
638      self._save_path = self._save_path.replace("/", "\\")
639    #### workaround
</pre>

<pre>
c:\py38-efficientdet\Lib\site-packages\tensorflow\python\saved_model
 +- builder_impl.py

595    variables_path = saved_model_utils.get_variables_path(self._export_dir)
596    ### workaround date="2022/06/18" os="Windows" 
597    import platform
598    if platform.system() == "Windows":
599      variables_path = variables_path.replace("/", "\\")
600    ### workaround
</pre>

<br>

<h3>3. Inspect tfrecord</h3>
 Move to ./projects/UK_RoadSigns directory, 
 and run the following bat file to inspect train/train.tfrecord:
<pre>
tfrecord_inspect.bat
</pre>
, which is the following:
<pre>
python ../../TFRecordInspector.py ^
  ./train/*.tfrecord ^
  ./label_map.pbtxt ^
  ./Inspector/train
</pre>
<br>
This will generate annotated images with bboxes and labels from the tfrecord, and cout the number of annotated objects in it.<br>
<br>
<b>TFRecordInspecotr: annotated images in train.tfrecord</b><br>
<img src="./asset/tfrecord_inspector_annotated_images.png">

<br>
<br>
<b>TFRecordInspecotr: objects_count train.tfrecord</b><br>
<img src="./asset/tfrecord_inspector_objects_count.png">
<br>
This bar graph shows that the number of the objects.
<br>
<br>
<br>
<h3>4. Downloading the pretrained-model efficientdet-d0</h3>
Please download an EfficientDet model chekcpoint file <b>efficientdet-d0.tar.gz</b>, and expand it in <b>EfficientDet-UK-RoadSigns</b> folder.<br>
<br>
https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d0.tar.gz
<br>
See: https://github.com/google/automl/tree/master/efficientdet<br>


<h3>5. Training UK RoadSigns Model by using pretrained-model</h3>
Move to the ./projects/UK_RoadSigns directory, and run the following bat file to train roadsigns efficientdet model:
<pre>
1_train.bat
</pre> 
, which is the following:
<pre>
rem 1_train.bat
python ../../ModelTrainer.py ^
  --mode=train_and_eval ^
  --train_file_pattern=./train/*.tfrecord  ^
  --val_file_pattern=./valid/*.tfrecord ^
  --model_name=efficientdet-d0 ^
  --hparams="input_rand_hflip=False,image_size=512x512,num_classes=94,label_map=./label_map.yaml" ^
  --model_dir=./models ^
  --label_map_pbtxt=./label_map.pbtxt ^
  --eval_dir=./eval ^
  --ckpt=../../efficientdet-d0  ^
  --train_batch_size=4 ^
  --early_stopping=map ^
  --patience=10 ^
  --eval_batch_size=1 ^
  --eval_samples=1000  ^
  --num_examples_per_epoch=2000 ^
  --num_epochs=160
</pre>

<table style="border: 1px solid #000;">
<tr>
<td>
--mode</td><td>train_and_eval</td>
</tr>
<tr>
<td>
--train_file_pattern</td><td>./train/train.tfrecord</td>
</tr>
<tr>
<td>
--val_file_pattern</td><td>./valid/valid.tfrecord</td>
</tr>
<tr>
<td>
--model_name</td><td>efficientdet-d0</td>
</tr>
<tr><td>
--hparams</td><td>"input_rand_hflip=False,image_size=512,num_classes=94,label_map=./label_map.yaml"
</td></tr>
<tr>
<td>
--model_dir</td><td>./models</td>
</tr>
<tr><td>
--label_map_pbtxt</td><td>./label_map.pbtxt
</td></tr>

<tr><td>
--eval_dir</td><td>./eval
</td></tr>

<tr>
<td>
--ckpt</td><td>../../efficientdet-d0</td>
</tr>
<tr>
<td>
--train_batch_size</td><td>4</td>
</tr>
<tr>
<td>
--early_stopping</td><td>map</td>
</tr>
<tr>
<td>
--patience</td><td>10</td>
</tr>

<tr>
<td>
--eval_batch_size</td><td>1</td>
</tr>
<tr>
<td>
--eval_samples</td><td>1000</td>
</tr>
<tr>
<td>
--num_examples_per_epoch</td><td>2000</td>
</tr>
<tr>
<td>
--num_epochs</td><td>160</td>
</tr>
</table>
<br>
<br>
<b>label_map.yaml</b>
<pre>
1: 'All_vehicles_prohibited'
2: 'Articulated_vehicles_prohibited'
3: 'Bend_to_right'
4: 'Buses_and_tramcars_only'
5: 'Buses_prohibited'
6: 'Children_going_to_or_from_school'
7: 'Crossroads'
8: 'Cyclists_and_pedestrians_only'
9: 'Double_bend_first_to_the_left'
10: 'End_of_20_miles_per_hour_zone'
11: 'End_of_30_miles_minimum_speed_limit'
12: 'End_of_dual_carriageway'
13: 'Entrance_20_miles_per_hour_speed_limit_zone'
14: 'Falling_or_fallen_rocks'
15: 'Fire_station_ahead'
16: 'Ford_warning'
17: 'Frail_pedestrians_likely_to_cross'
18: 'Give_way'
19: 'Horse_drawn_vehicles_prohibited'
20: 'Hump_bridge'
21: 'Ice_or_snowdrifts'
22: 'Junction_on_a_bend'
23: 'Keep_to_the_left'
24: 'Loose_chippings'
25: 'Low_flying_aircraft_or_sudden_aircraft_noise'
26: 'Low_flying_helicopters_or_sudden_helicopter_noise'
27: 'Maximum_speed_10'
28: 'Maximum_speed_20'
29: 'Maximum_speed_30'
30: 'Maximum_speed_40'
31: 'Maximum_speed_50'
32: 'Maximum_speed_60'
33: 'Minimum_speed_limit_30_miles_per_hour'
34: 'Mini_roundabout'
35: 'Motor_vehicles_except_solo_motorcycles_prohibited'
36: 'Motor_vehicles_prohibited'
37: 'National_speed_limit_applies'
38: 'No_entry_for_vehicular_traffic'
39: 'No_left_turn'
40: 'No_overtaking'
41: 'No_right_turn'
42: 'No_stopping'
43: 'No_u_turns_for_vehicular_traffic'
44: 'No_waiting_or_no_parking'
45: 'One_way_traffic'
46: 'Opening_or_swing_bridge'
47: 'Other_dangers'
48: 'Parking_place'
49: 'Part_time_traffic_signals_ahead'
50: 'Pedal_cycles_only'
51: 'Pedestrians_ahead'
52: 'Pedestrians_prohibited'
53: 'Priority_must_be_given_to_vehicles_from_the_opposite_direction'
54: 'Ridden_or_accompanied_horses_prohibited'
55: 'Riding_of_pedal_cycles_prohibited'
56: 'Riverbank_or_quayside'
57: 'Road_narrows_on_both_sides'
58: 'Road_narrows_on_the_right'
59: 'Road_works'
60: 'Roundabout'
61: 'Side_road'
62: 'Side_winds'
63: 'Slippery_road'
64: 'Slow_moving_military_vehicles_likely_to_be_in_or_crossing_the_road'
65: 'Slow_moving_vehicles'
66: 'Soft_verges'
67: 'Solo_motorcycles_prohibited'
68: 'Speed_camera'
69: 'Staggered_junction'
70: 'Steep_ascent'
71: 'Steep_descent'
72: 'Stop'
73: 'Stop_children_crossing_ahead'
74: 'Stop_or_give_way_ahead'
75: 'Stop_police_ahead'
76: 'Towed_caravans_prohibited'
77: 'Traffic_merges_from_the_left'
78: 'Traffic_merges_onto_the_main_carriageway'
79: 'Traffic_queues_likely_ahead'
80: 'Traffic_signals_ahead'
81: 'Tramcars_only'
82: 'Tramway'
83: 'Try_your_brakes'
84: 'Tunnel'
85: 'Turn_ahead_in_the_direction'
86: 'Turn_left'
87: 'Two_way_traffic'
88: 'Two_way_traffic_on_route_crossing_ahead'
89: 'T_junction'
90: 'Uneven_road'
91: 'Vehicles_carrying_explosives_prohibited'
92: 'Water_course_alongside_road'
93: 'Wild_animals'
94: 'Zebra_crossing'

</pre>

<br>
<b>Training console output at epoch 62</b>
<br>
<img src="./asset/coco_metrics_console_at_epoch65_tf2.8.0_0705.png" width="1024" height="auto">
<br>
<br>
<b><a href="./projects/UK_RoadSigns/eval/coco_metrics.csv">COCO meticss</a></b><br>
<img src="./asset/coco_metrics_at_epoch65_tf2.8.0_0705.png" width="1024" height="auto">
<br>
<br>
<b><a href="./projects/UK_RoadSigns/eval/train_losses.csv">Train losses</a></b><br>
<img src="./asset/train_losses_at_epoch65_tf2.8.0_0705.png" width="1024" height="auto">
<br>
<br>

<b><a href="./projects/UK_RoadSigns/eval/coco_ap_per_class.csv">COCO ap per class</a></b><br>
<img src="./asset/coco_ap_per_class_at_epoch65_tf2.8.0_0705.png" width="1024" height="auto">
<br>
<br>
<h3>
6. Create a saved_model from the checkpoint
</h3>
 Please run the following bat file to create a saved model from a chekcpoint in models folder.
<pre>
2_create_saved_model.bat
</pre>
, which is the following:
<pre>
python ../../SavedModelCreator.py ^
  --runmode=saved_model ^
  --model_name=efficientdet-d0 ^
  --ckpt_path=./models  ^
  --hparams="image_size=512x512,num_classes=94" ^
  --saved_model_dir=./saved_model
</pre>

<table style="border: 1px solid #000;">
<tr>
<td>--runmode</td><td>saved_model</td>
</tr>

<tr>
<td>--model_name </td><td>efficientdet-d0 </td>
</tr>

<tr>
<td>--ckpt_path</td><td>./models</td>
</tr>

<tr>
<td>--hparams</td><td>"image_size=512x512,num_classes=94"</td>
</tr>

<tr>
<td>--saved_model_dir</td><td>./saved_model</td>
</tr>
</table>

<br>
<br>
<h3>
7. Inference UK_RoadSigns by using the saved_model
</h3>
 Please run the following bat file to infer the roadsigns by using the saved_model:
<pre>
</pre>
, which is the following:
<pre>
python ../../SavedModelInferencer.py ^
  --runmode=saved_model_infer ^
  --model_name=efficientdet-d0 ^
  --saved_model_dir=./saved_model ^
  --min_score_thresh=0.4 ^
  --hparams="label_map=./label_map.yaml" ^
  --input_image=./realistic_test_dataset/*.jpg ^
  --classes_file=./classes.txt ^
  --ground_truth_json=./realistic_test_dataset/annotation.json ^
  --output_image_dir=./realistic_test_dataset_outputs
</pre>

<table style="border: 1px solid #000;">
<tr>
<td>--runmode</td><td>saved_model_infer </td>
</tr>

<tr>
<td>--model_name</td><td>efficientdet-d0 </td>
</tr>

<tr>
<td>--saved_model_dir</td><td>./saved_model </td>
</tr>

<tr>
<td>--min_score_thresh</td><td>0.4 </td>
</tr>

<tr>
<td>--hparams</td><td>"label_map=./label_map.yaml"</td>
</tr>

<tr>
<td>--input_image</td><td>./realistic_test_dataset/*.jpg</td>
</tr>

<tr>
<td>--classes_file</td><td>./classes.txt</td>
</tr>
<tr>
<td>--ground_truth_json</td><td>./realistic_test_dataset/annotation.json</td>
</tr>

<tr>
<td>--output_image_dir</td><td>./realistic_test_dataset_outputs</td>
</tr>
</table>
<br>
<h3>
8. Some inference results of UK RoadSigns
</h3>

<img src="./projects/UK_RoadSigns/realistic_test_dataset_outputs/uk_roadsigns_1000.jpg" width="1280" height="auto"><br>
<a href="./projects/UK_RoadSigns/realistic_test_dataset_outputs/uk_roadsigns_1000.jpg_objects.csv">roadsigns_1.jpg_objects.csv</a><br>

<img src="./projects/UK_RoadSigns/realistic_test_dataset_outputs/uk_roadsigns_1020.jpg" width="1280" height="auto"><br>
<a  href="./projects/UK_RoadSigns/realistic_test_dataset_outputs/uk_roadsigns_1020.jpg_objects.csv">roadsigns_2.jpg_objects.csv</a><br>

<img src="./projects/UK_RoadSigns/realistic_test_dataset_outputs/uk_roadsigns_1030.jpg" width="1280" height="auto"><br>
<a  href="./projects/UK_RoadSigns/realistic_test_dataset_outputs/uk_roadsigns_1030.jpg_objects.csv">roadsigns_3.jpg_objects.csv</a><br>

<img src="./projects/UK_RoadSigns/realistic_test_dataset_outputs/uk_roadsigns_1040.jpg" width="1280" height="auto"><br>
<a  href="./projects/UK_RoadSigns/realistic_test_dataset_outputs/uk_roadsigns_1040.jpg_objects.csv">roadsigns_4.jpg_objects.csv</a><br>

<img src="./projects/UK_RoadSigns/realistic_test_dataset_outputs/uk_roadsigns_1050.jpg" width="1280" height="auto"><br>
<a  href="./projects/UK_RoadSigns/realistic_test_dataset_outputs/uk_roadsigns_1050.jpg_objects.csv">roadsigns_5.jpg_objects.csv</a><br>

<img src="./projects/UK_RoadSigns/realistic_test_dataset_outputs/uk_roadsigns_1060.jpg" width="1280" height="auto"><br>
<a  href="./projects/UK_RoadSigns/realistic_test_dataset_outputs/uk_roadsigns_1060.jpg_objects.csv">roadsigns_6.jpg_objects.csv</a><br>

<img src="./projects/UK_RoadSigns/realistic_test_dataset_outputs/uk_roadsigns_1070.jpg" width="1280" height="auto"><br>
<a  href="./projects/UK_RoadSigns/realistic_test_dataset_outputs/uk_roadsigns_1070.jpg_objects.csv">roadsigns_7.jpg_objects.csv</a><br>

<img src="./projects/UK_RoadSigns/realistic_test_dataset_outputs/uk_roadsigns_1080.jpg" width="1280" height="auto"><br>
<a  href="./projects/UK_RoadSigns/realistic_test_dataset_outputs/uk_roadsigns_1080.jpg_objects.csv">roadsigns_8.jpg_objects.csv</a><br>

<img src="./projects/UK_RoadSigns/realistic_test_dataset_outputs/uk_roadsigns_1090.jpg" width="1280" height="auto"><br>
<a  href="./projects/UK_RoadSigns/realistic_test_dataset_outputs/uk_roadsigns_1090.jpg_objects.csv">roadsigns_9.jpg_objects.csv</a><br>

<img src="./projects/UK_RoadSigns/realistic_test_dataset_outputs/uk_roadsigns_1099.jpg" width="1280" height="auto"><br>
<a  href="./projects/UK_RoadSigns/realistic_test_dataset_outputs/uk_roadsigns_1099.jpg_objects.csv">roadsigns_10.jpg_objects.csv</a><br>

<h3>9. COCO metrics of inference result</h3>
The 3_inference.bat computes also the COCO metrics(f, map, mar) to the <b>realistic_test_dataset</b> as shown below:<br>

<a href="./projects/UK_RoadSigns/realistic_test_dataset_outputs/prediction_f_map_mar.csv">prediction_f_map_mar.csv</a>

<br>
<img src="./asset/coco_metrics_console_test_dataset_at_epoch65_tf2.8.0_0705.png" width="740" height="auto"><br>

