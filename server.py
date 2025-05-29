import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from PIL import Image
import numpy as np
import torch
import base64
from io import BytesIO
from imports import load_model_vit, morgan_to_image  # Your custom imports

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Load model and label encoder
vit, le = load_model_vit("vit_model.pkl")

app = FastAPI()

class InputData(BaseModel):
    SMILES: str

# Function to generate 2D structure image from SMILES
def smiles_to_base64_image(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        AllChem.Compute2DCoords(mol)
        drawer = Draw.MolDraw2DCairo(300, 300)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        png_data = drawer.GetDrawingText()
        img = Image.open(BytesIO(png_data))
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        logger.error(f"Error generating 2D image: {e}")
        return None

@app.post('/predict/vit')
def predict(data: InputData):
    try:
        smiles = data.SMILES.strip()
        logger.info(f"Processing SMILES: {smiles}")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise HTTPException(status_code=400, detail="Invalid SMILES")

        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        fp_array = np.array(fingerprint, dtype=np.float32)
        fp_array = fp_array / (fp_array.max() + 1e-6)
        image = morgan_to_image(fp_array)
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            output = vit(image_tensor)
            predicted_class_idx = output.argmax(1).item()
            predicted_label = le.inverse_transform([predicted_class_idx])[0]

        structure_image = smiles_to_base64_image(smiles)

        return {
            "smiles": smiles,
            "activity": predicted_label,
            "image": structure_image
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Internal server error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 🔥 Dynamic PORT like Node.js style
if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get("PORT", 4000))  # Default to 4000 if PORT not set
    uvicorn.run(app, host="0.0.0.0", port=port)
