#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
extract organ masks from totalseg segmentation results
Index 	TotalSegmentator name
1 	spleen 	
2 	kidney_right 	
3 	kidney_left 	
4 	gallbladder 	
5 	liver 	
6 	stomach 	
7 	pancreas 	
8 	adrenal_gland_right 	
9 	adrenal_gland_left 	
10 	lung_upper_lobe_left 	
11 	lung_lower_lobe_left 	
12 	lung_upper_lobe_right 	
13 	lung_middle_lobe_right 
14 	lung_lower_lobe_right 
15 	esophagus 	
16 	trachea 	
17 	thyroid_gland 	
18 	small_bowel 	small intestine
19 	duodenum 	
20 	colon 	
21 	urinary_bladder 	
22 	prostate 	
23 	kidney_cyst_left 	
24 	kidney_cyst_right 	
25 	sacrum 	
26 	vertebrae_S1 	
27 	vertebrae_L5 	
28 	vertebrae_L4 	
29 	vertebrae_L3 	
30 	vertebrae_L2 	
31 	vertebrae_L1 	
32 	vertebrae_T12 	
33 	vertebrae_T11 	
34 	vertebrae_T10 	
35 	vertebrae_T9 	
36 	vertebrae_T8 	
37 	vertebrae_T7 	
38 	vertebrae_T6 	
39 	vertebrae_T5 	
40 	vertebrae_T4 	
41 	vertebrae_T3 	
42 	vertebrae_T2 	
43 	vertebrae_T1 	
44 	vertebrae_C7 	
45 	vertebrae_C6 	
46 	vertebrae_C5 	
47 	vertebrae_C4 	
48 	vertebrae_C3 	
49 	vertebrae_C2 	
50 	vertebrae_C1 	
51 	heart 	
52 	aorta 	
53 	pulmonary_vein 	
54 	brachiocephalic_trunk 	
55 	subclavian_artery_right 	
56 	subclavian_artery_left 	
57 	common_carotid_artery_right 	
58 	common_carotid_artery_left 	
59 	brachiocephalic_vein_left 	
60 	brachiocephalic_vein_right 	
61 	atrial_appendage_left 	
62 	superior_vena_cava 	
63 	inferior_vena_cava 	
64 	portal_vein_and_splenic_vein 	hepatic portal vein
65 	iliac_artery_left 	common iliac artery
66 	iliac_artery_right 	common iliac artery
67 	iliac_vena_left 	common iliac vein
68 	iliac_vena_right 	common iliac vein
69 	humerus_left 	
70 	humerus_right 	
71 	scapula_left 	
72 	scapula_right 	
73 	clavicula_left 	clavicle
74 	clavicula_right 	clavicle
75 	femur_left 	
76 	femur_right 	
77 	hip_left 	
78 	hip_right 	
79 	spinal_cord 	
80 	gluteus_maximus_left 	gluteus maximus muscle
81 	gluteus_maximus_right 	gluteus maximus muscle
82 	gluteus_medius_left 	gluteus medius muscle
83 	gluteus_medius_right 	gluteus medius muscle
84 	gluteus_minimus_left 	gluteus minimus muscle
85 	gluteus_minimus_right 	gluteus minimus muscle
86 	autochthon_left 	
87 	autochthon_right 	
88 	iliopsoas_left 	iliopsoas muscle
89 	iliopsoas_right 	iliopsoas muscle
90 	brain 	
91 	skull 	
92 	rib_left_1 	
93 	rib_left_2 	
94 	rib_left_3 	
95 	rib_left_4 	
96 	rib_left_5 	
97 	rib_left_6 	
98 	rib_left_7 	
99 	rib_left_8 	
100 	rib_left_9 	
101 	rib_left_10 	
102 	rib_left_11 	
103 	rib_left_12 	
104 	rib_right_1 	
105 	rib_right_2 	
106 	rib_right_3 	
107 	rib_right_4 	
108 	rib_right_5 	
109 	rib_right_6 	
110 	rib_right_7 	
111 	rib_right_8 	
112 	rib_right_9 	
113 	rib_right_10 	
114 	rib_right_11 	
115 	rib_right_12 	
116 	sternum 	
117 	costal_cartilages
"""

import os
join = os.path.join
import numpy as np
from tqdm import tqdm
import nibabel as nib
import multiprocessing as mp


s_total_path = ''
t_path = ''
os.makedirs(t_path, exist_ok=True)

names = os.listdir(s_total_path)
print(f'num of files {len(names)=}')
# for name in tqdm(names):
def process_label(name):
    if not os.path.isfile(join(t_path, name)):
        total_nii = nib.load(join(s_total_path, name))
        total_data = total_nii.get_fdata()
        new_label_data = np.zeros_like(total_data, dtype=np.uint8)
        new_label_data[total_data==2] = 1 # right kidney
        new_label_data[total_data==24] = 1 # right kidney cyst
        new_label_data[total_data==1] = 2 # spleen
        new_label_data[total_data==3] = 3 # left kidney
        new_label_data[total_data==23] = 3 # left kidney cyst
        # new_label_data[total_data==4] = 4 # gallbladder
        new_label_data[total_data==5] = 5 # liver
        new_label_data[total_data==6] = 6 # stomach
        new_label_data[total_data==7] = 7 # pancreas
        # new_label_data[total_data==8] = 8 # right adrenal gland
        # new_label_data[total_data==9] = 8 # left adrenal gland
        new_label_data[total_data==10] = 8 # lef lung
        new_label_data[total_data==11] = 8 # left lung
        new_label_data[total_data==12] = 9 # right lung
        new_label_data[total_data==13] = 9 # right lung
        new_label_data[total_data==14] = 9 # right lung
        # new_label_data[total_data==15] = 10 # esophagus
        # new_label_data[total_data==16] = 10 # trachea
        # new_label_data[total_data==17] = 10 # thyroid gland
        # new_label_data[total_data==18] = 11 # small intestine
        # new_label_data[total_data==19] = 11 # duodenum
        # new_label_data[total_data==20] = 11 # colon
        # new_label_data[total_data==21] = 12 # urinary bladder
        # new_label_data[total_data==22] = 12 # prostate

        # set all the bones (label 25-50) to 13
        # new_label_data[np.logical_and(total_data>=25, total_data<=50)] = 13
        # new_label_data[np.logical_and(total_data>=69, total_data<=79)] = 13
        # new_label_data[total_data>=92] = 13
        # new_label_data[np.logical_and(total_data>=51, total_data<=68)] = 14 # heart and vessels
        # new_label_data[np.logical_and(total_data>=80, total_data<=89)] = 15 # muscles
        

        save_nii = nib.Nifti1Image(new_label_data, total_nii.affine)
        nib.save(save_nii, join(t_path, name))

if __name__ == '__main__':
    with mp.Pool(8) as p:
        r = list(tqdm(p.imap(process_label, names), total=len(names)))
    print('done! Results are saved in', t_path)


