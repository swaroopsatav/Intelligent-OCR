import hashlib
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import time
from services.api_client import (
    ApiClientError,
    upload_file,
    preprocess_document,
    run_ocr,
    correct_text,
    classify_document,
    extract_information,
    validate_document,
    ask_question
)

# ----------------------------------
# Page Config
# ----------------------------------

st.set_page_config(
    page_title="Intelligent OCR System",
    page_icon="📄",
    layout="wide"
)

# ----------------------------------
# Title
# ----------------------------------

st.title("📄 Intelligent OCR System")

st.markdown(
    "Upload a PDF or Image document and extract information using OCR and Open Source LLMs."
)

PIPELINE_STATE_KEYS = [
    "upload_response",
    "processed_images",
    "preprocessing_results",
    "ocr_text",
    "ocr_metadata",
    "corrected_text",
    "document_type",
    "classification_confidence",
    "extracted_data",
    "validation_result",
    "qa_history"
]


def clear_pipeline_state():
    for key in PIPELINE_STATE_KEYS:
        st.session_state.pop(
            key,
            None
        )


def clear_session_keys(keys):
    for key in keys:
        st.session_state.pop(
            key,
            None
        )


def clear_after_preprocessing():
    clear_session_keys(
        [
            "ocr_text",
            "ocr_metadata",
            "corrected_text",
            "document_type",
            "classification_confidence",
            "extracted_data",
            "validation_result",
            "qa_history"
        ]
    )


def clear_after_ocr():
    clear_session_keys(
        [
            "corrected_text",
            "document_type",
            "classification_confidence",
            "extracted_data",
            "validation_result",
            "qa_history"
        ]
    )


def clear_after_correction():
    clear_session_keys(
        [
            "document_type",
            "classification_confidence",
            "extracted_data",
            "validation_result",
            "qa_history"
        ]
    )


def clear_after_extraction():
    clear_session_keys(
        [
            "validation_result"
        ]
    )


def show_api_error(error, fallback_message="Something went wrong."):
    messages = {
        "backend_unavailable": (
            "Backend is unavailable. Please start the FastAPI backend "
            "server, then try again."
        ),
        "backend_timeout": (
            "Backend request timed out. The server may still be loading "
            "a model or processing a large document."
        ),
        "invalid_file_format": (
            "Invalid file format. Please upload a PDF, PNG, JPG, or JPEG file."
        ),
        "model_loading_failed": (
            "Model loading failed. Check the configured model name, network "
            "access, Hugging Face token, and available memory."
        ),
        "backend_error": (
            "Backend failed while processing the request. Please check the "
            "backend logs and try again."
        ),
        "ocr_failed": (
            "OCR failed for this document. Try a clearer image or run "
            "preprocessing again."
        )
    }

    st.error(
        messages.get(
            error.error_code,
            fallback_message
        )
    )

    if error.message:
        st.caption(
            f"Details: {error.message}"
        )


def get_file_signature(file):
    file_bytes = file.getvalue()

    return (
        file.name,
        file.type,
        file.size,
        hashlib.md5(
            file_bytes
        ).hexdigest()
    )


def flatten_for_export(data, prefix=""):
    flattened = {}

    for key, value in data.items():
        export_key = (
            f"{prefix}.{key}"
            if prefix
            else key
        )

        if isinstance(
            value,
            dict
        ):
            flattened.update(
                flatten_for_export(
                    value,
                    export_key
                )
            )
        elif isinstance(
            value,
            list
        ):
            flattened[
                export_key
            ] = json.dumps(
                value,
                ensure_ascii=False
            )
        else:
            flattened[
                export_key
            ] = value

    return flattened


def to_json_export(data):
    return json.dumps(
        data,
        indent=2,
        ensure_ascii=False
    )


def to_single_row_csv(data):
    return pd.DataFrame(
        [
            flatten_for_export(
                data
            )
        ]
    ).to_csv(
        index=False
    )


def get_export_basename():
    upload_response = st.session_state.get(
        "upload_response",
        {}
    )

    original_name = upload_response.get(
        "filename",
        "document"
    )

    return Path(
        original_name
    ).stem.replace(
        " ",
        "_"
    )


def render_export_buttons(
    title,
    data,
    json_filename,
    csv_filename=None,
    csv_data=None
):
    st.markdown(
        f"#### {title}"
    )

    export_json_col, export_csv_col = st.columns(2)

    with export_json_col:

        st.download_button(
            "Download JSON",
            data=to_json_export(
                data
            ),
            file_name=json_filename,
            mime="application/json",
            use_container_width=True
        )

    with export_csv_col:

        include_csv = st.checkbox(
            "Include CSV export",
            key=f"{title}_csv_export"
        )

        if include_csv and csv_filename and csv_data is not None:

            st.download_button(
                "Download CSV",
                data=csv_data,
                file_name=csv_filename,
                mime="text/csv",
                use_container_width=True
            )


def load_screenshot_demo_state():
    if st.query_params.get(
        "demo"
    ) != "screenshots":
        return

    st.session_state.setdefault(
        "upload_response",
        {
            "filename": "sample_invoice.pdf",
            "file_path": "data/uploads/sample_invoice.pdf",
            "file_size": 184320,
            "message": "Upload successful"
        }
    )

    st.session_state.setdefault(
        "ocr_text",
        (
            "INVOICE\n"
            "Invoice No: INV-2026-014\n"
            "Date: 24/06/2026\n"
            "Bill To: Swaroop Satav\n"
            "Total Amount: Rs. 12,450.00"
        )
    )

    st.session_state.setdefault(
        "corrected_text",
        (
            "INVOICE\n"
            "Invoice No: INV-2026-014\n"
            "Date: 24/06/2026\n"
            "Bill To: Swaroop Satav\n"
            "Total Amount: Rs. 12,450.00"
        )
    )

    st.session_state.setdefault(
        "document_type",
        "Invoice"
    )

    st.session_state.setdefault(
        "classification_confidence",
        0.96
    )

    st.session_state.setdefault(
        "extracted_data",
        {
            "document_title": "Invoice",
            "document_type": "Invoice",
            "name": "Swaroop Satav",
            "company_name": "Acme Services",
            "document_id": "INV-2026-014",
            "date": "24/06/2026",
            "address": "Pune, Maharashtra",
            "phone": "9876543210",
            "email": "billing@example.com",
            "invoice_number": "INV-2026-014",
            "total_amount": "Rs. 12,450.00",
            "pan": "ABCDE1234F",
            "gst": "27ABCDE1234F1Z5",
            "aadhaar": "",
            "dynamic_fields": {
                "vendor_name": "Acme Services",
                "buyer_name": "Swaroop Satav",
                "payment_status": "Pending"
            }
        }
    )

    st.session_state.setdefault(
        "validation_result",
        {
            "email": True,
            "phone": True,
            "date": True,
            "pan": True,
            "aadhaar": False,
            "gst": True
        }
    )

    st.session_state.setdefault(
        "qa_history",
        [
            {
                "question": "What is the invoice total?",
                "answer": "The invoice total is Rs. 12,450.00."
            }
        ]
    )


load_screenshot_demo_state()

# ==================================
# Upload Section
# ==================================

uploaded_file = st.file_uploader(
    "Choose Document",
    type=["pdf", "png", "jpg", "jpeg"]
)

if uploaded_file:

    current_file_signature = get_file_signature(
        uploaded_file
    )

    if (
        st.session_state.get(
            "selected_file_signature"
        )
        != current_file_signature
    ):
        clear_pipeline_state()

        st.session_state[
            "selected_file_signature"
        ] = current_file_signature

    col1, col2 = st.columns([2, 1])

    with col1:

        st.subheader("Document Preview")

        if uploaded_file.type.startswith("image"):

            st.image(
                uploaded_file,
                use_container_width=True
            )

        else:

            st.info(
                f"📄 PDF Selected: {uploaded_file.name}"
            )

    with col2:

        st.subheader("File Details")

        st.write(
            f"**Name:** {uploaded_file.name}"
        )

        st.write(
            f"**Type:** {uploaded_file.type}"
        )

        st.write(
            f"**Size:** "
            f"{round(uploaded_file.size / 1024, 2)} KB"
        )

    if st.button(
        "Upload Document",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "Uploading document..."
        ):

            try:

                response = upload_file(
                    uploaded_file
                )

                clear_pipeline_state()

                st.session_state[
                    "selected_file_signature"
                ] = current_file_signature

                st.success(
                    "✅ Upload Successful"
                )

                st.session_state[
                    "upload_response"
                ] = response

            except ApiClientError as e:

                show_api_error(
                    e,
                    "Upload failed. Please try again."
                )

            except Exception as e:

                st.error(
                    f"Upload failed: {e}"
                )
else:
    if "selected_file_signature" in st.session_state:
        clear_pipeline_state()

        st.session_state.pop(
            "selected_file_signature",
            None
        )

# ==================================
# Upload Information
# ==================================

if "upload_response" in st.session_state:

    st.divider()

    st.subheader("Upload Information")

    st.json(
        st.session_state[
            "upload_response"
        ]
    )

# ==================================
# Preprocessing Section
# ==================================

if "upload_response" in st.session_state:

    st.divider()

    st.subheader(
        "Image Preprocessing"
    )

    if st.button(
        "Run Preprocessing",
        use_container_width=True
    ):

        try:

            file_path = (
                st.session_state[
                    "upload_response"
                ]["file_path"]
            )

            clear_after_preprocessing()

            with st.spinner(
                "Preprocessing..."
            ):

                response = preprocess_document(
                    file_path
                )

            st.success(
                "✅ Preprocessing Completed"
            )

            st.session_state[
                "processed_images"
            ] = response[
                "processed_images"
            ]

            st.session_state[
                "preprocessing_results"
            ] = response.get(
                "preprocessing_results",
                []
            )

        except ApiClientError as e:

            show_api_error(
                e,
                "Preprocessing failed. Please try again."
            )

        except Exception as e:

            st.error(
                f"Preprocessing failed: {e}"
            )

# ==================================
# Processed Images
# ==================================

if "processed_images" in st.session_state:

    st.divider()

    st.subheader(
        "Before / After Preprocessing"
    )

    preprocessing_results = st.session_state.get(
        "preprocessing_results",
        []
    )

    if preprocessing_results:

        for idx, result in enumerate(
            preprocessing_results,
            start=1
        ):

            st.markdown(
                f"### Page {idx}"
            )

            before_col, after_col = st.columns(2)

            with before_col:

                st.caption(
                    "Original"
                )

                st.image(
                    result.get(
                        "original_image",
                        ""
                    ),
                    use_container_width=True
                )

            with after_col:

                st.caption(
                    "Processed"
                )

                st.image(
                    result.get(
                        "processed_image",
                        ""
                    ),
                    use_container_width=True
                )

            st.caption(
                "Applied steps: "
                + ", ".join(
                    result.get(
                        "applied_steps",
                        []
                    )
                )
            )

    else:

        for idx, image_path in enumerate(
            st.session_state[
                "processed_images"
            ],
            start=1
        ):

            st.markdown(
                f"### Page {idx}"
            )

            st.image(
                image_path,
                use_container_width=True
            )

# ==================================
# OCR Section
# ==================================

if "processed_images" in st.session_state:

    st.divider()

    st.subheader(
        "OCR Extraction"
    )

    if st.button(
        "Run OCR",
        type="primary",
        use_container_width=True
    ):

        try:

            clear_after_ocr()

            full_text = ""

            with st.spinner(
                "Running EasyOCR..."
            ):

                for image_path in (
                    st.session_state[
                        "processed_images"
                    ]
                ):

                    response = run_ocr(
                        image_path
                    )

                    if "text" in response:

                        full_text += (
                            response["text"]
                            + "\n\n"
                        )

                    elif "ocr_text" in response:

                        full_text += (
                            response["ocr_text"]
                            + "\n\n"
                        )

            if not full_text.strip():
                raise ApiClientError(
                    "No text was detected in the processed document.",
                    error_code="ocr_failed"
                )

            st.session_state[
                "ocr_text"
            ] = full_text

            st.session_state[
                "ocr_metadata"
            ] = response

            st.success(
                "✅ OCR Completed"
            )

        except ApiClientError as e:

            if e.error_code == "backend_error":
                e.error_code = "ocr_failed"

            show_api_error(
                e,
                "OCR failed. Please try again."
            )

        except Exception as e:

            st.error(
                f"OCR failed: {e}"
            )

# ==================================
# OCR Result
# ==================================

if "ocr_text" in st.session_state:

    st.divider()

    st.subheader(
        "Extracted OCR Text"
    )

    st.text_area(
        label="OCR Result",
        value=st.session_state[
            "ocr_text"
        ],
        height=350
    )

    ocr_metadata = st.session_state.get(
        "ocr_metadata",
        {}
    )

    if ocr_metadata.get(
        "overlay_image"
    ):

        st.subheader(
            "Bounding Box Visualization"
        )

        st.image(
            ocr_metadata[
                "overlay_image"
            ],
            use_container_width=True
        )

    if ocr_metadata.get(
        "bounding_boxes"
    ):

        with st.expander(
            "OCR Bounding Boxes"
        ):
            st.json(
                ocr_metadata[
                    "bounding_boxes"
                ]
            )

    if ocr_metadata.get(
        "table_rows"
    ):

        with st.expander(
            "Detected Table Rows"
        ):
            st.json(
                ocr_metadata[
                    "table_rows"
                ]
            )

    if ocr_metadata.get(
        "tables"
    ):

        with st.expander(
            "Structured Tables"
        ):
            st.json(
                ocr_metadata[
                    "tables"
                ]
            )

    if ocr_metadata.get(
        "key_value_pairs"
    ):

        with st.expander(
            "Detected Key-Value Pairs"
        ):
            st.json(
                ocr_metadata[
                    "key_value_pairs"
                ]
            )

# ==================================
# OCR Correction
# ==================================

if "ocr_text" in st.session_state:

    st.divider()

    st.subheader(
        "LLM OCR Correction"
    )

    if st.button(
        "Correct OCR Text",
        type="primary",
        use_container_width=True
    ):

        try:

            clear_after_correction()

            with st.spinner(
                "Correcting OCR text..."
            ):

                response = correct_text(
                    st.session_state[
                        "ocr_text"
                    ]
                )

            corrected_text = response.get(
                "corrected_text",
                ""
            )

            if isinstance(
                corrected_text,
                dict
            ):
                corrected_text = corrected_text.get(
                    "corrected_text",
                    ""
                )

            if response.get(
                "warning"
            ):
                st.warning(
                    "Model loading failed. Using basic OCR cleanup instead."
                )

                st.caption(
                    f"Details: {response['warning']}"
                )

            st.session_state[
                "corrected_text"
            ] = corrected_text

            st.success(
                "✅ OCR Correction Complete"
            )

        except ApiClientError as e:

            show_api_error(
                e,
                "OCR correction failed. Please try again."
            )

        except Exception as e:

            st.error(
                f"OCR correction failed: {e}"
            )

# ==================================
# Corrected Text
# ==================================

if "corrected_text" in st.session_state:

    st.divider()

    st.subheader(
        "Corrected Text"
    )

    st.text_area(
        label="Corrected OCR Output",
        value=st.session_state[
            "corrected_text"
        ],
        height=350
    )

# ==================================
# Document Classification
# ==================================

if "corrected_text" in st.session_state:

    st.divider()

    st.subheader(
        "Document Classification"
    )

    if st.button(
        "Classify Document",
        type="primary",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "Classifying document..."
            ):

                response = classify_document(
                    st.session_state[
                        "corrected_text"
                    ]
                )

            st.session_state[
                "document_type"
            ] = response[
                "document_type"
            ]

            st.session_state[
                "classification_confidence"
            ] = response.get(
                "confidence",
                0.0
            )

            st.success(
                "✅ Classification Complete"
            )

        except ApiClientError as e:

            show_api_error(
                e,
                "Classification failed. Please try again."
            )

        except Exception as e:

            st.error(
                f"Classification failed: {e}"
            )

# ==================================
# Classification Result
# ==================================

if "document_type" in st.session_state:

    st.divider()

    st.subheader(
        "Classification Result"
    )

    st.success(
        f"📄 Document Type: "
        f"{st.session_state['document_type']}"
    )

    st.info(
        f"Confidence: "
        f"{st.session_state.get('classification_confidence', 0.0) * 100:.0f}%"
    )

# ==================================
# Information Extraction
# ==================================

if "corrected_text" in st.session_state:

    st.divider()

    st.subheader(
        "Information Extraction"
    )

    if st.button(
        "Extract Information",
        type="primary",
        use_container_width=True
    ):

        try:

            clear_after_extraction()

            with st.spinner(
                "Extracting information..."
            ):

                response = extract_information(
                    st.session_state[
                        "corrected_text"
                    ]
                )

            st.session_state[
                "extracted_data"
            ] = response

            st.success(
                "✅ Information Extraction Complete"
            )

        except ApiClientError as e:

            show_api_error(
                e,
                "Information extraction failed. Please try again."
            )

        except Exception as e:

            st.error(
                f"Information extraction failed: {e}"
            )

    # ==================================
    # Extracted Information
    # ==================================

    if "extracted_data" in st.session_state:
        st.divider()

        st.subheader(
            "Extracted Information"
        )

        st.json(
            st.session_state[
                "extracted_data"
            ]
        )

        data = st.session_state[
            "extracted_data"
        ]

        export_basename = get_export_basename()

        render_export_buttons(
            "Export Extracted Data",
            data,
            f"{export_basename}_extracted_data.json",
            f"{export_basename}_extracted_data.csv",
            to_single_row_csv(
                data
            )
        )

        st.subheader(
            "Key Information"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.write(
                "**Document Title:**",
                data.get(
                    "document_title",
                    ""
                )
            )

            st.write(
                "**Name:**",
                data.get(
                    "name",
                    ""
                )
            )

            st.write(
                "**Company Name:**",
                data.get(
                    "company_name",
                    ""
                )
            )

            st.write(
                "**Document ID:**",
                data.get(
                    "document_id",
                    ""
                )
            )

            st.write(
                "**Phone:**",
                data.get(
                    "phone",
                    ""
                )
            )

            st.write(
                "**Email:**",
                data.get(
                    "email",
                    ""
                )
            )

        with col2:
            st.write(
                "**Date:**",
                data.get(
                    "date",
                    ""
                )
            )

            st.write(
                "**Document Type:**",
                data.get(
                    "document_type",
                    ""
                )
            )

            st.write(
                "**Invoice Number:**",
                data.get(
                    "invoice_number",
                    ""
                )
            )

            st.write(
                "**Amount:**",
                data.get(
                    "total_amount",
                    ""
                )
            )

            st.write(
                "**PAN:**",
                data.get(
                    "pan",
                    ""
                )
            )

            st.write(
                "**GST:**",
                data.get(
                    "gst",
                    ""
                )
            )

            st.write(
                "**Aadhaar:**",
                data.get(
                    "aadhaar",
                    ""
                )
            )

            st.write(
                "**Address:**",
                data.get(
                    "address",
                    ""
                )
            )

        dynamic_fields = data.get(
            "dynamic_fields",
            {}
        )

        if dynamic_fields:

            st.subheader(
                "Document Specific Fields"
            )

            st.json(
                dynamic_fields
            )

# ==================================
# Validation
# ==================================

if "extracted_data" in st.session_state:

    st.divider()

    st.subheader(
        "Data Validation"
    )

    if st.button(
        "Validate Data",
        type="primary",
        use_container_width=True
    ):

        try:

            response = validate_document(
                st.session_state[
                    "extracted_data"
                ]
            )

            st.session_state[
                "validation_result"
            ] = response

            st.success(
                "✅ Validation Complete"
            )

        except ApiClientError as e:

            show_api_error(
                e,
                "Validation failed. Please try again."
            )

        except Exception as e:

            st.error(
                f"Validation failed: {e}"
            )

# ==================================
# Validation Results
# ==================================

if "validation_result" in st.session_state:

    st.divider()

    st.subheader(
        "Validation Results"
    )

    results = st.session_state[
        "validation_result"
    ]

    valid_count = sum(
        results.values()
    )

    total_count = len(results)

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Valid Fields",
            valid_count
        )

    with col2:
        st.metric(
            "Total Fields",
            total_count
        )

    validation_df = pd.DataFrame(
        [
            {
                "Field": key.upper(),
                "Is Valid": value,
                "Status":
                    "✅ Valid"
                    if value
                    else "❌ Invalid"
            }
            for key, value in
            results.items()
        ]
    )

    st.dataframe(
        validation_df,
        use_container_width=True
    )

    export_basename = get_export_basename()

    render_export_buttons(
        "Export Validation Result",
        results,
        f"{export_basename}_validation_result.json",
        f"{export_basename}_validation_result.csv",
        validation_df.to_csv(
            index=False
        )
    )

# ==================================
# Question Answering
# ==================================

if "corrected_text" in st.session_state:

    st.divider()

    st.subheader(
        "Document Question Answering"
    )

    question = st.text_input(
        "Ask a question about the document"
    )

    if st.button(
        "Get Answer",
        type="primary",
        use_container_width=True
    ):

        if not question:

            st.warning(
                "Please enter a question."
            )

        else:

            try:

                with st.spinner(
                    "Generating answer..."
                ):

                    response = ask_question(
                        st.session_state[
                            "corrected_text"
                        ],
                        question
                    )

                if (
                    "qa_history"
                    not in st.session_state
                ):
                    st.session_state[
                        "qa_history"
                    ] = []

                st.session_state[
                    "qa_history"
                ].append(
                    {
                        "question": question,
                        "answer": response[
                            "answer"
                        ]
                    }
                )

            except ApiClientError as e:

                show_api_error(
                    e,
                    "Question answering failed. Please try again."
                )

            except Exception as e:

                st.error(
                    f"Question answering failed: {e}"
                )

# ==================================
# QA History
# ==================================

if "qa_history" in st.session_state:

    st.divider()

    st.subheader(
        "Question & Answers"
    )

    for item in reversed(
        st.session_state[
            "qa_history"
        ]
    ):

        st.markdown(
            f"**Q:** {item['question']}"
        )

        st.success(
            item["answer"]
        )
# ==================================
# Pipeline Status
# ==================================

st.divider()
st.subheader("Pipeline Status")

if "upload_response" not in st.session_state:

    st.info(
        "📄 Upload a document to start processing."
    )

else:

    # show progress tracker
    ...

st.divider()

st.subheader("Pipeline Status")

pipeline_steps = [
    ("upload_response", "Upload"),
    ("processed_images", "Preprocessing"),
    ("ocr_text", "OCR Extraction"),
    ("corrected_text", "OCR Correction"),
    ("document_type", "Document Classification"),
    ("extracted_data", "Information Extraction"),
    ("validation_result", "Data Validation"),
    ("qa_history", "Question Answering")
]

completed_steps = 0
total_steps = len(pipeline_steps)

for session_key, label in pipeline_steps:

    if session_key in st.session_state:

        st.success(
            f"✅ {label} Complete"
        )

        completed_steps += 1

    else:

        st.info(
            f"⏳ {label} Pending"
        )

# ==================================
# Overall Progress
# ==================================

progress = completed_steps / total_steps

st.divider()

st.subheader("Overall Progress")

st.progress(progress)

st.caption(
    f"{completed_steps}/{total_steps} Steps Completed "
    f"({round(progress * 100)}%)"
)

# ==================================
# Project Complete
# ==================================

if completed_steps == total_steps:

    st.balloons()

    st.success(
        "Intelligent OCR Pipeline Completed Successfully!"
    )


