# Training Your GPT-2 Chatbot

This guide explains how to teach your GPT-2 chatbot new information and customize its responses using the fine-tuning capabilities.

## Overview

The GPT-2 chatbot can now learn from:
1. Custom text data (articles, books, documentation)
2. Conversation examples (question-answer pairs)
3. Interactive examples you provide

## Quick Start

### Training on a Text File

```bash
python train_chatbot.py --model "gpt2-small (124M)" --text_file your_text_file.txt --epochs 1 --save_path "trained_model.pt"
```

### Training on Conversation Examples

First, create a JSON file with examples:

```json
[
  {
    "user": "What is your favorite color?",
    "bot": "My favorite color is blue. I find it calming and peaceful."
  },
  {
    "user": "Tell me about quantum physics",
    "bot": "Quantum physics is the study of matter and energy at its most fundamental level..."
  }
]
```

Then train the model:

```bash
python train_chatbot.py --model "gpt2-small (124M)" --conversation_file examples.json --epochs 2 --save_path "trained_model.pt"
```

### Interactive Mode

Train and immediately test your model:

```bash
python train_chatbot.py --text_file your_data.txt --interactive
```

## Advanced Usage

### Using a Pre-trained Model

```python
from gpt_generate import GPT2Chatbot

# Initialize with custom weights
chatbot = GPT2Chatbot(model_name="gpt2-small (124M)")
chatbot.load_model("path/to/your/trained_model.pt")

# Chat with the fine-tuned model
response = chatbot.chat("Tell me what you learned")
print(response)
```

### Adding Examples Without Training

You can add examples to the conversation history without full training:

```python
examples = [
    {"user": "Who created you?", "bot": "I was created by you to help with your specific tasks."},
    {"user": "What's our project about?", "bot": "Our project focuses on natural language processing and custom AI assistants."}
]

chatbot.add_custom_examples(examples)
```

### Creating a Domain-Specific Assistant

To create a specialized assistant:

1. Collect domain-specific text (articles, documentation, books)
2. Create example conversations showing ideal responses
3. Fine-tune a smaller model first (faster training)
4. Test and refine with additional examples

## Training Tips

1. **Start small**: Use the smallest model (124M) for faster experimentation
2. **Quality data**: Curate high-quality examples that demonstrate the style and knowledge you want
3. **Few epochs**: Often 1-3 epochs is sufficient to avoid overfitting
4. **Save checkpoints**: Save models after training to compare different approaches
5. **Combine approaches**: Use both text training and conversation examples

## Troubleshooting

- **Out of memory errors**: Reduce batch size or use a smaller model
- **Poor quality responses**: Provide more diverse, high-quality examples
- **Training too slow**: Use a smaller model size for initial experiments

## Example Workflow

1. Create example conversations showing ideal responses
2. Train on a small model for quick iteration
3. Test responses and identify weaknesses
4. Add more targeted examples to address weaknesses
5. Retrain and evaluate
6. When satisfied, optionally train a larger model for better quality