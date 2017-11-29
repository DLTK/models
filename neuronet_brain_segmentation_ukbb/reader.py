import SimpleITK as sitk
import tensorflow as tf
import numpy as np

from dltk.io.augmentation import extract_random_example_array
from dltk.io.preprocessing import whitening

ALL_PROTOCOLS = ['fsl_fast', 'fsl_first', 'spm_tissue', 'malp_em', 'malp_em_tissue']
NUM_CLASSES = [4, 16, 4, 139, 6]

def map_labels(lbl, protocol=None, convert_to_protocol=False):
    """
        Map dataset specific label id protocols to consecutive integer ids for training and back.
        Parameters
        ----------
        lbl : np.array
            a label map
        protocol : str
            a string describing the labeling protocol
        convert_to_protocol : bool
            flag to determine to convert from or to the protocol ids
    """

    """
        SPM tissue ids:
        0 Background
        1 CSF
        2 GM
        3 WM
    """
    spm_tissue_ids = range(4)
    
        
    """
        Fast ids:
        0 Background
        1 CSF
        2 GM
        3 WM
    """
    fast_ids = range(4)
    
    
    """
        First ids:
        0 Background
        10 Left-Thalamus-Proper 40
        11 Left-Caudate 30
        12 Left-Putamen 40
        13 Left-Pallidum 40
        16 Brain-Stem /4th Ventricle 40
        17 Left-Hippocampus 30
        18 Left-Amygdala 50
        26 Left-Accumbens-area 50
        49 Right-Thalamus-Proper 40
        50 Right-Caudate 30
        51 Right-Putamen 40
        52 Right-Pallidum 40
        53 Right-Hippocampus 30
        54 Right-Amygdala 50
        58 Right-Accumbens-area 50 
    """
    first_ids = [0, 10, 11, 12, 13, 16, 17, 18, 26, 49, 50, 51, 52, 53, 54, 58]
    
    
    """
        MALP-EM tissue ids:
        0 Background
        1 Ventricles
        2 Sub-cortical and cerebellum GM
        3 WM
        4 Cortical GM
        5 ?????
    """
    malpem_tissue_ids = range(6)
    
        
    """
        MALP-EM ids:
        0 Background
        1 3rdVentricle
        2 4thVentricle
        3 RightAccumbensArea
        4 LeftAccumbensArea
        5 RightAmygdala
        6 LeftAmygdala
        7 BrainStem
        8 RightCaudate
        9 LeftCaudate
        10 RightCerebellumExterior
        11 LeftCerebellumExterior
        12 RightCerebellumWhiteMatter
        13 LeftCerebellumWhiteMatter
        14 RightCerebralExterior
        15 LeftCerebralExterior
        16 RightCerebralWhiteMatter
        17 LeftCerebralWhiteMatter
        18 CSF
        19 RightHippocampus
        20 LeftHippocampus
        21 RightInfLatVent
        22 LeftInfLatVent
        23 RightLateralVentricle
        24 LeftLateralVentricle
        25 RightPallidum
        26 LeftPallidum
        27 RightPutamen
        28 LeftPutamen
        29 RightThalamusProper
        30 LeftThalamusProper
        31 RightVentralDC
        32 LeftVentralDC
        33 Rightvessel
        34 Leftvessel
        35 OpticChiasm
        36 CerebellarVermalLobulesI-V
        37 CerebellarVermalLobulesVI-VII
        38 CerebellarVermalLobulesVIII-X
        39 LeftBasalForebrain
        40 RightBasalForebrain
        41 RightACg Ganteriorcingulategyrus Right
        42 LeftACg Ganteriorcingulategyrus Left
        43 RightAIns Anteriorinsula Right
        44 LeftAIns Anteriorinsula Left
        45 RightAOrG Anteriororbitalgyrus Right
        46 LeftAOrG Anteriororbitalgyrus Left
        47 RightAnG Angulargyrus Right
        48 LeftAnG Angulargyrus Left
        49 RightCalc Calcarinecortex Right
        50 LeftCalc Calcarinecortex Left
        51 RightCO Centraloperculum Right
        52 LeftCO Centraloperculum Left
        53 RightCun Cuneus Right
        54 LeftCun Cuneus Left
        55 RightEntA Ententorhinalarea Right
        56 LeftEntA Ententorhinalarea Left
        57 RightFO Frontaloperculum Right
        58 LeftFO Frontaloperculum Left
        59 RightFRP Frontalpole Right
        60 LeftFRP Frontalpole Left
        61 RightFuG Fusiformgyrus Right
        62 LeftFuG Fusiformgyrus Left
        63 RightGRe Gyrusrectus Right
        64 LeftGRe Gyrusrectus Left
        65 RightIOG Inferioroccipitalgyrus Right
        66 LeftIOG Inferioroccipitalgyrus Left
        67 RightITG Inferiortemporalgyrus Right
        68 LeftITG Inferiortemporalgyrus Left
        69 RightLiG Lingualgyrus Right
        70 LeftLiG Lingualgyrus Left
        71 RightLOrG Lateralorbitalgyrus Right
        72 LeftLOrG Lateralorbitalgyrus Left
        73 RightMCgG Middlecingulategyrus Right
        74 LeftMCgG Middlecingulategyrus Left
        75 RightMFC Medialfrontalcortex Right
        76 LeftMFC Medialfrontalcortex Left
        77 RightMFG Middlefrontalgyrus Right
        78 LeftMFG Middlefrontalgyrus Left
        79 RightMOG Middleoccipitalgyrus Right
        80 LeftMOG Middleoccipitalgyrus Left
        81 RightMOrG Medialorbitalgyrus Right
        82 LeftMOrG Medialorbitalgyrus Left
        83 RightMPoG Postcentralgyrusmedialsegment Right
        84 LeftMPoG Postcentralgyrusmedialsegment Left
        85 RightMPrG Precentralgyrusmedialsegment Right
        86 LeftMPrG Precentralgyrusmedialsegment Left
        87 RightMSFG Superiorfrontalgyrusmedialsegment Right
        88 LeftMSFG Superiorfrontalgyrusmedialsegment Left
        89 RightMTG Middletemporalgyrus Right
        90 LeftMTG Middletemporalgyrus Left
        91 RightOCP Occipitalpole Right
        92 LeftOCP Occipitalpole Left
        93 RightOFuG Occipitalfusiformgyrus Right
        94 LeftOFuG Occipitalfusiformgyrus Left
        95 RightOpIFG Opercularpartoftheinferiorfrontalgyrus Right
        96 LeftOpIFG Opercularpartoftheinferiorfrontalgyrus Left
        97 RightOrIFG Orbitalpartoftheinferiorfrontalgyrus Right
        98 LeftOrIFG Orbitalpartoftheinferiorfrontalgyrus Left
        99 RightPCgG Posteriorcingulategyrus Right
        100 LeftPCgG Posteriorcingulategyrus Left
        101 RightPCu Precuneus Right
        102 LeftPCu Precuneus Left
        103 RightPHG Parahippocampalgyrus Right
        104 LeftPHG Parahippocampalgyrus Left
        105 RightPIns Posteriorinsula Right
        106 LeftPIns Posteriorinsula Left
        107 RightPO Parietaloperculum Right
        108 LeftPO Parietaloperculum Left
        109 RightPoG Postcentralgyrus Right
        110  LeftPoG Postcentralgyrus Left
        111 RightPOrG Posteriororbitalgyrus Right
        112 LeftPOrG Posteriororbitalgyrus Left
        113 RightPP Planumpolare Right
        114 LeftPP Planumpolare Left
        115 RightPrG Precentralgyrus Right
        116 LeftPrG Precentralgyrus Left
        117 RightPT Planumtemporale Right
        118 LeftPT Planumtemporale Left
        119 RightSCA Subcallosalarea Right
        120 LeftSCA Subcallosalarea Left
        121 RightSFG Superiorfrontalgyrus Right
        122 LeftSFG Superiorfrontalgyrus Left
        123 RightSMC Supplementarymotorcortex Right
        124 LeftSMC Supplementarymotorcortex Left
        125 RightSMG Supramarginalgyrus Right
        126 LeftSMG Supramarginalgyrus Left
        127 RightSOG Superioroccipitalgyrus Right
        128 LeftSOG Superioroccipitalgyrus Left
        129 RightSPL Superiorparietallobule Right
        130 LeftSPL Superiorparietallobule Left
        131 RightSTG Superiortemporalgyrus Right
        132 LeftSTG Superiortemporalgyrus Left
        133 RightTMP Temporalpole Right
        134 LeftTMP Temporalpole Left
        135 RightTrIFG Triangularpartoftheinferiorfrontalgyrus Right
        136 LeftTrIFG Triangularpartoftheinferiorfrontalgyrus Left
        137 RightTTG Transversetemporalgyrus Right
        138 LeftTTG Transversetemporalgyrus Left
    """
    malpem_ids = range(139)
    
    out_lbl = np.zeros_like(lbl)
    
    if protocol == 'fsl_fast':
        ids = fast_ids
    elif protocol == 'fsl_first':
        ids = first_ids
    elif protocol == 'spm_tissue':
        ids = spm_tissue_ids
    elif protocol == 'malp_em':
        ids = malpem_ids
    elif protocol == 'malp_em_tissue':
        ids = malpem_tissue_ids
    else:
        print("Method is not recognised. Exiting.")
        return -1
    
    if convert_to_protocol:
        # map from consecutive ints to protocol labels
        for i in range(len(ids)):
            out_lbl[lbl==i] = ids[i]
    else:
        # map from protocol labels to consecutive ints
        for i in range(len(ids)):
            out_lbl[lbl==ids[i]] = i
            
    return out_lbl

def read_fn(file_references, mode, params=None):
    """A custom python read function for interfacing with nii image files.

    Args:
        file_references (list): A list of lists containing file references,
            such as [['id_0', 'image_filename_0', target_value_0], ...,
             ['id_N', 'image_filename_N', target_value_N]].
        mode (str): One of the tf.estimator.ModeKeys strings: TRAIN, EVAL
            or PREDICT.
        params (dict, optional): A dictionary to parameterise read_fn ouputs
            (e.g. reader_params = {'n_examples': 10, 'example_size':
            [64, 64, 64], 'extract_examples': True}, etc.).

    Yields:
        dict: A dictionary of reader outputs for dltk.io.abstract_reader.
    """

    for f in file_references:

        # Read the image nii with sitk
        img_id = f[0]
        img_fn = f[1]
        img_sitk = sitk.ReadImage(str(img_fn))
        img = sitk.GetArrayFromImage(img_sitk)

        # Normalise volume image
        img = whitening(img)
        
        # Create a 4D image (i.e. [x, y, z, channels])
        img = np.expand_dims(img, axis=-1).astype(np.float32)
        
        if mode == tf.estimator.ModeKeys.PREDICT:
            yield {'features': {'x': img},
                   'labels': None}

        # Read the label nii with sitk for each of the protocols
        lbls = []
        for p in params['protocols']:
            idx = ALL_PROTOCOLS.index(p)
            lbl_fn = f[2 + idx]
            lbl = sitk.GetArrayFromImage(sitk.ReadImage(str(lbl_fn))).astype(np.int32)

            # Map the label ids to consecutive integers
            lbl = map_labels(lbl, protocol=p)
            lbls.append(lbl)

        # Check if the reader is supposed to return training examples or
        # full images   
        if params['extract_examples']:
            # Concatenate into a list of images and labels and extract
            img_lbls_list = [img] + lbls
            img_lbls_list = extract_random_example_array(
                img_lbls_list,
                example_size=params['example_size'],
                n_examples=params['n_examples'])
                        
            # Yield each image example and corresponding label protocols 
            for e in range(params['n_examples']):
                yield {'features': {'x': img_lbls_list[0][e].astype(np.float32)},
                       'labels': {params['protocols'][i]: img_lbls_list[1 + i][e]
                                  for i in range(len(params['protocols']))}
                      }
        else:
            yield {'features': {'x': img},
                   'labels': {params['protocols'][i]: img_lbls_list[1 + i][e]
                                  for i in range(len(params['protocols']))},
                   'sitk': img_sitk,
                   'img_id': img_id}
    return