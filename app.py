import streamlit as st
from transformers import pipeline

# Load the models
pii_model = pipeline("ner", model="iiiorg/piiranha-v1-detect-personal-information")
pci_model = pipeline("ner", model="lakshyakh93/deberta_finetuned_pii")
phi_model = pipeline("ner", model="obi/deid_roberta_i2b2")

# Function to run all models and collect entities
def run_all_models(text):
    pii_entities = pii_model(text)
    pci_entities = pci_model(text)
    phi_entities = phi_model(text)

    return {'PII': pii_entities, 'PCI': pci_entities, 'PHI': phi_entities}

# Resolve conflicts between models
def resolve_conflicts(entities):
    resolved_entities = []
    entity_map = {}
    
    for category, entity_list in entities.items():
        for entity in entity_list:
            start = entity['start']
            end = entity['end']
            text = entity['word']
            label = category
            score = entity['score']
            
            if (start, end) not in entity_map:
                entity_map[(start, end)] = {'text': text, 'label': label, 'score': score}
            else:
                # Compare confidence scores and assign the most relevant category
                current_label = entity_map[(start, end)]['label']
                if label == 'PCI' and ('ACCOUNTNUM' in text or 'CREDITCARDNUMBER' in text):
                    entity_map[(start, end)]['label'] = 'PCI'
                elif score > entity_map[(start, end)]['score']:
                    entity_map[(start, end)] = {'text': text, 'label': label, 'score': score}
    
    for (start, end), entity in entity_map.items():
        resolved_entities.append(entity)
    
    return resolved_entities

# Color code the entities based on category
def color_code_entities(text, entities):
    color_map = {
        'PII': '#ADD8E6',  # Light blue
        'PCI': '#FFFFE0',  # Light yellow
        'PHI': '#FFC0CB'   # Light pink
    }
    sorted_entities = sorted(entities, key=lambda x: x['start'])
    colored_text = text
    offset = 0

    for entity in sorted_entities:
        start = entity['start'] + offset
        end = entity['end'] + offset
        label = entity['label']
        entity_text = text[start:end]
        colored_text = colored_text[:start] + f"<span style='background-color:{color_map[label]}'>{entity_text}</span>" + colored_text[end:]
        offset += len(f"<span style='background-color:{color_map[label]}'>") + len("</span>") - (end - start)
    
    return colored_text

# Streamlit app
def ner_app():
    st.title("Multi-model NER: PII, PCI, PHI")
    
    input_text = st.text_area("Enter your text for NER analysis", height=200)
    
    if st.button("Run NER"):
        # Run the NER models
        entities = run_all_models(input_text)
        resolved_entities = resolve_conflicts(entities)
        colored_text = color_code_entities(input_text, resolved_entities)
        
        # Display colored text
        st.markdown(colored_text, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    ner_app()
