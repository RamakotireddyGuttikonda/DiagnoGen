from imports import *

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

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
        # Generate 2D coordinates
        AllChem.Compute2DCoords(mol)
        # Draw the molecule
        drawer = Draw.MolDraw2DCairo(300, 300)  # Use Cairo backend for PNG
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        # Get the PNG data
        png_data = drawer.GetDrawingText()
        # Convert to PIL Image
        img = Image.open(BytesIO(png_data))
        # Convert to base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str
    except Exception as e:
        logger.error(f"Error generating 2D structure for SMILES {smiles}: {e}")
        return None


@app.post('/predict/vit')
def predict(data: InputData):
    try:
        smiles = data.SMILES.strip()
        logger.info(f"Processing SMILES for VIT prediction: {smiles}")
        
        # Convert SMILES to molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.error(f"Invalid SMILES: {smiles}")
            raise HTTPException(status_code=400, detail="Invalid SMILES")
        
        # Generate Morgan fingerprint
        try:
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        except Exception as e:
            logger.error(f"Error generating fingerprint: {e}")
            raise HTTPException(status_code=400, detail="Error during fingerprint generation")
        
        fp_array = np.array(fingerprint, dtype=np.float32)
        fp_array = fp_array / (fp_array.max() + 1e-6)
        
        # Convert to image and predict
        image = morgan_to_image(fp_array)
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            output = vit(image_tensor)
            predicted_class_idx = output.argmax(1).item()
            predicted_label = le.inverse_transform([predicted_class_idx])[0]
        
        # Generate 2D structure image
        structure_image = smiles_to_base64_image(smiles)
        
        result = {
            "smiles": smiles,
            "activity": predicted_label,
            "image": structure_image
        }
        logger.info(f"VIT prediction result: {result}")
        return result

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Internal server error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
    

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)