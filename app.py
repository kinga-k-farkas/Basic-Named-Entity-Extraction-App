import streamlit as st
import spacy
from spacy import displacy


if 'address_ner' not in st.session_state:
    st.session_state.address_ner= spacy.load("Models/model_best_3_Plus")

if 'drug_ner' not in st.session_state:
    st.session_state.drug_ner = spacy.load("Models/model-best_drug_Plus")

if 'ner' not in st.session_state:
    st.session_state.ner = spacy.load('en_core_web_lg')


def ss(key):
    return st.session_state[key]


address_ner = ss('address_ner')
drug_ner = ss('drug_ner')
ner = ss('ner')

display_dict = {"ADDRESS": "Street Addresses", "PHONE_NUMBER": "Phone Numbers", "DRUG": "Drugs"}


def get_ents( sentence, nlp):
    doc = nlp(sentence)
    ent_dict = {}
    for ent in doc.ents:
        if ent.label_ not in ent_dict:
            ent_dict[ent.label_] = [ent]
        else:
            ent_dict[ent.label_].append(ent)
    return ent_dict

st.subheader("Gathering Basic Information with NER")
st.caption("Extracting Street Addresses, Phone Numbers, Drug Names and Generic Named Entities from Text")
query = st.text_area('Enter your text:')
if len(query) == 0:
    st.write('Entities:')
else:
    address_dict = get_ents(query, address_ner)
    if not address_dict:
        st.write("No street addresses and no phone numbers were found in the text")
    else:
        for key in address_dict.keys():
            st.write(f"{display_dict[key]}: {address_dict[key]}")
    drug_dict = get_ents(query, drug_ner)
    if not drug_dict:
        st.write("No drug names were found in the text.")
    else:
        st.write(f"Drugs: {drug_dict['DRUG']}")


HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

if st.button("Display Generic Entities"):
    docx = ner(query)
    html = displacy.render(docx, style="ent")
    html = html.replace("\n\n", "\n")
    st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)

