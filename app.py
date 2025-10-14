import os
os.environ['TF_KERAS'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import gradio as gr
from prediction import predict
import json

def predict_properties(smiles_input):
    """Wrapper function for Gradio interface"""
    try:
        if not smiles_input.strip():
            return "Please enter a SMILES string"
        
        result = predict(smiles_input)
        
        if "error" in result:
            return f"❌ Error: {result['error']}"
        
        # Format output for display
        output = f"""
🧪 **Polymer Property Predictions**

**Input SMILES:** `{result['smiles']}`

**Predicted Properties:**
• 🌡️  **Glass Transition Temperature (Tg):** {result['properties']['Tg (Glass Transition Temperature)']}
• 🔥  **Crystallization Temperature (Tc):** {result['properties']['Tc (Crystallization Temperature)']}
• ⚡  **Fractional Free Volume (FFV):** {result['properties']['FFV (Fractional Free Volume)']}
• 🏋️  **Density:** {result['properties']['Density']}
• 📏  **Radius of Gyration (Rg):** {result['properties']['Rg (Radius of Gyration)']}

---
*Predictions generated using Graph Neural Networks trained on NeurIPS 2025 competition data*
        """
        
        return output
        
    except Exception as e:
        return f"❌ Prediction failed: {str(e)}"

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="🧪 Polymer Property Predictor", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🧪 Polymer Property Prediction Tool
        
        Enter a polymer SMILES string to predict five key polymer properties using advanced Graph Neural Networks.
        
        **Properties Predicted:**
        - **Tg**: Glass Transition Temperature (°C)
        - **Tc**: Crystallization Temperature (°C)
        - **FFV**: Fractional Free Volume
        - **Density**: Polymer Density (g/cm³)
        - **Rg**: Radius of Gyration (Å)
        """)
        
        with gr.Row():
            with gr.Column():
                smiles_input = gr.Textbox(
                    label="🧬 Polymer SMILES String",
                    placeholder="Enter SMILES (e.g., *CC* for polyethylene)",
                    lines=2
                )
                
                predict_btn = gr.Button("🔬 Predict Properties", variant="primary")
                
                gr.Markdown("""
                **Example SMILES:**
                - `*/C=C/*` - Polyethylene
                - `*CC(C)*` - Polypropylene  
                - `*C*` - Polymethylene
                """)
            
            with gr.Column():
                output = gr.Markdown(
                    label="📊 Prediction Results",
                    value="Enter a SMILES string and click 'Predict Properties' to see results."
                )
        
        predict_btn.click(
            fn=predict_properties,
            inputs=[smiles_input],
            outputs=[output]
        )
        
        gr.Markdown("""
        ---
        **About this Model:**
        This prediction model was developed using Graph Neural Networks for the NeurIPS 2025 Open Polymer Prediction competition.
        
        🔗 [GitHub Repository](https://github.com/Gaurav-Kushwaha-1225/NeurIPS-Open-Polymer-Prediction-2025)
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
