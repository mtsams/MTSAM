Overview：

MTSAM is a novel glioma diagnosis framework that leverages a customized SAM-Med3D model. It fine-tunes large models using multi-view information to simultaneously perform IDH typing and grading of gliomas. This framework effectively integrates multimodal MRI, handcrafted radiomics (HCR), and clinical features to improve prognostic accuracy, outperforming several glioma diagnosis methods.

Gliomas are the most common malignant brain tumors in adults, and accurate survival risk prediction is crucial for personalized treatment. Although molecular biomarkers have prognostic value, their invasive and costly nature limits their widespread application. MTSAM addresses this issue by utilizing non-invasive MRI data, HCR, and clinical features, combined with advanced deep learning techniques, to achieve robust diagnosis.
Key Contributions：

1) We propose a multi-task network named MTSAM, which explores T2-FLAIR mismatch features and utilizes the customized SAM-Med3D to explore the SAM-Med3D’s prior knowledge learned from large-scale medical data for accurate glioma IDH genotyping and grading.

2) We propose a multi-view adapter called MVAdapter that explores complementary and multi-scale information in multi-view data, including HCR, clinical, and MRI features, to fine-tune SAM-Med3D and uncover deep features for glioma IDH genotyping and grading.

3) We propose a T2-FLAIR mismatch feature extraction block based on the human visual system, named MFEB, which first provides an overview of MRIs through dilated convolutions, and then conducts detailed exploration using convolutions, aiming to capture the complementary information between T2 and FLAIR images and perform weighted subtraction to obtain T2-FLAIR mismatch features.

Datasets：

MTSAM is validated on two publicly available datasets:

UCSF-PDGM Dataset

Access: https://www.cancerimagingarchive.net/collection/ucsf-pdgm/

BraTS2020 Dataset

Access: https://www.med.upenn.edu/cbica/brats2020/data.html

Requirements：

To run MTSAM, the following dependencies are required:

torch>=1.10.0 monai>=0.9.0 torchvision>=0.11.0 scipy>=1.7.0 numpy>=1.21.0 pickle5>=0.0.11

Install dependencies using pip:

pip install torch>=1.10.0 monai>=0.9.0 torchvision>=0.11.0 scipy>=1.7.0 numpy>=1.21.0 pickle5>=0.0.11

Usage：

Data Preparation

Download the UCSF-PDGM or BraTS2020 dataset from the provided links.

Ensure the dataset includes multimodal MRI scans, handcrafted radiomics features, and clinical features (e.g., age, gender).

Training

To train MTSAM on the UCSF-PDGM dataset:

python main.py --dataset ucsf-pdgm

To train on the BraTS2020 dataset:

python main.py --dataset brats2020
