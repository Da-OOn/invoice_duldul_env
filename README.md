# YOLOv5 Custom Dataset Invoice Recognition

This repository contains the `invoice_duldul` environment used for recognizing and processing invoices using a custom YOLOv5 model. 덜덜이가 송장을 인식하기 위한 프로젝트입니다!! 0_<

## Overview

The project aims to detect and recognize invoices, extract text using OCR (Optical Character Recognition), and store the extracted information in a structured format. The main code file for this project is `invoice_main.py`.

## Environment Setup

To recreate the `invoice_duldul` environment, use the provided `environment.yml` file. This environment includes all necessary dependencies for running the invoice recognition system.

### Steps to Set Up the Environment

1. **Clone the repository:**

    ```bash
    git clone https://github.com/Da-OOn/invoice_duldul_env.git
    cd invoice_duldul_env
    ```

2. **Create the conda environment:**

    ```bash
    conda env create -f environment.yml
    ```

3. **Activate the environment:**

    ```bash
    conda activate invoice_duldul
    ```

4. **Install specific numpy version (if necessary):**

    ```bash
    conda install numpy=1.21.2
    ```

## Running the Invoice Recognition System

The main code file `invoice_main.py` handles the detection and OCR processing of the invoices. Below is an example of how to run the script.

### Running the Script

Ensure your webcam is connected and functional, as the script uses it to capture images.

```bash
python invoice_main.py


덜덜이 얼굴 prototype!!!! https://www.figma.com/design/CybQ51lG0dFeS7U9cCacEs/%EB%8D%9C%EB%8D%9C%EC%9D%B4-GUI-design_oon?node-id=0-1&t=Og1gYvbETUT9NLJh-0
