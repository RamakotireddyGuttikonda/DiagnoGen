# Standard Library Imports
import os
import io
import re
import uuid
import json
import shutil
import tempfile
import logging
import traceback
from datetime import datetime
from collections import deque
from pathlib import Path

# Third-Party Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import base64

# RDKit
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Draw
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# Machine Learning & Deep Learning
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import joblib

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import keras.backend as K

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# Tokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE

# Image and Medical Imaging
import cv2
import nibabel as nib
from patchify import patchify

# FastAPI Framework
from fastapi import FastAPI, File, UploadFile, Form, Query, APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Jupyter Display (if applicable)
from IPython.display import display

# Local Module Imports
from modules.vit import load_model_vit, morgan_to_image
from modules.stackRNN import StackAugmentedRNN
from modules.data import GeneratorData
from modules.predictor import QSAR
from modules.chembert import RoBERTaForMaskedLM, chemberta_predict
from modules.docking import (
    download_pdb_and_ligand,
    convert_ligand_to_pdbqt,
    run_plif_and_visualize,
    add_hydrogens_to_protein
)
from modules.predict import predict_smiles
from helper import identity, dice_coef, dice_loss
from io import BytesIO
import sqlite3
import json
from datetime import datetime
from fastapi import HTTPException
import logging