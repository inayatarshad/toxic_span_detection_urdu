This project focuses on toxic span detection in Urdu text using a transformer-based token classification approach. The goal is not only to identify whether a sentence contains toxic content, but also to locate and highlight the specific toxic words or spans within the text. In Urdu language, no work has been done prior in span detection. This is the first time, we propose toxicity span detection. For this purpose, the model is built using XLM-RoBERTa, a multilingual pretrained language model well-suited for low-resource languages such as Urdu. The system can be trained on annotated toxic span datasets and then used for inference to mark toxic segments in new sentences.

Task: Toxic span detection at word/token level in Urdu

Model: XLM-RoBERTa fine-tuned for token classification

Output: Highlighted toxic words/spans within each input sentence

Use case: Moderation, abusive language detection, and explainable toxicity analysis

Framework: HuggingFace Transformers with PyTorch training pipeline
