import pandas as pd
import os
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

# Cargar el modelo preentrenado de Hugging Face
model_name = "human-centered-summarization/financial-summarization-pegasus"
# model_name = "google/pegasus-large"     # más fiel al texto original
# model_name = "google/pegasus-xsum"      # más conciso, estilo noticioso
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(ROOT_DIR, 'data', 'news', 'processed')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'data', 'news', 'summarized')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Listar todos los archivos parquet en el directorio PROCESSED_DIR
parquet_files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith('.parquet')]

print(f"Found {len(parquet_files)} parquet files to process")

# Procesar cada archivo
for filename in parquet_files:
    print(f"\nProcessing: {filename}")
    
    input_path = os.path.join(PROCESSED_DIR, filename)
    df = pd.read_parquet(input_path, engine="pyarrow")
    
    # Crear lista para almacenar los resúmenes
    summaries = []
    
    # Generar resumen para cada fila
    for idx, row in df.iterrows():
        text_to_summarize = row["clean_body"]
        
        # Tokenize our text
        input_ids = tokenizer(text_to_summarize, return_tensors="pt").input_ids
        
        # Generate the output
        output = model.generate(
            input_ids,
            max_length=256,       
            min_length=80,        
            num_beams=5,          
            early_stopping=True
        )
        
        # Decode the summary
        summary = tokenizer.decode(output[0], skip_special_tokens=True)
        summaries.append(summary)
        
        print(f"  Processed row {idx + 1}/{len(df)}")
    
    # Añadir la columna body_summary al dataframe
    df["body_summary"] = summaries
    
    # Guardar con el mismo nombre de archivo
    output_path = os.path.join(OUTPUT_DIR, filename)
    df.to_parquet(output_path, engine="pyarrow", index=False)
    print(f"Saved: {output_path} ({len(df)} rows)")

print("\nAll files processed!")
